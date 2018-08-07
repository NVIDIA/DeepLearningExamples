import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from torch.nn.modules import Module
from torch.autograd import Variable
from collections import OrderedDict


def flat_dist_call(tensors, call, extra_args=None):
    flat_dist_call.warn_on_half = True
    buckets = OrderedDict()
    for tensor in tensors:
        tp = tensor.type()
        if tp not in buckets:
            buckets[tp] = []
        buckets[tp].append(tensor)

    if flat_dist_call.warn_on_half:
        if torch.cuda.HalfTensor in buckets:
            print("WARNING: gloo dist backend for half parameters may be extremely slow." +
                  " It is recommended to use the NCCL backend in this case.")
            flat_dist_call.warn_on_half = False

    for tp in buckets:
        bucket = buckets[tp]
        coalesced = _flatten_dense_tensors(bucket)
        if extra_args is not None:
            call(coalesced, *extra_args)
        else:
            call(coalesced)
        if call is dist.all_reduce:
            coalesced /= dist.get_world_size()

        for buf, synced in zip(bucket, _unflatten_dense_tensors(coalesced, bucket)):
            buf.copy_(synced)

class DistributedDataParallel(Module):
    """
    :class:`apex.parallel.DistributedDataParallel` is a module wrapper that enables
    easy multiprocess distributed data parallel training, similar to ``torch.nn.parallel.DistributedDataParallel``.

    :class:`DistributedDataParallel` is designed to work with
    the launch utility script ``apex.parallel.multiproc.py``.
    When used with ``multiproc.py``, :class:`DistributedDataParallel`
    assigns 1 process to each of the available (visible) GPUs on the node.
    Parameters are broadcast across participating processes on initialization, and gradients are
    allreduced and averaged over processes during ``backward()``.

    :class:`DistributedDataParallel` is optimized for use with NCCL.  It achieves high performance by
    overlapping communication with computation during ``backward()`` and bucketing smaller gradient
    transfers to reduce the total number of transfers required.

    :class:`DistributedDataParallel` assumes that your script accepts the command line
    arguments "rank" and "world-size."  It also assumes that your script calls
    ``torch.cuda.set_device(args.rank)`` before creating the model.

    https://github.com/NVIDIA/apex/tree/master/examples/distributed shows detailed usage.
    https://github.com/NVIDIA/apex/tree/master/examples/imagenet shows another example
    that combines :class:`DistributedDataParallel` with mixed precision training.

    Args:
        module: Network definition to be run in multi-gpu/distributed mode.
        message_size (Default = 1e7): Minimum number of elements in a communication bucket.
        shared_param (Default = False): If your model uses shared parameters this must be True.  It will disable bucketing of parameters to avoid race conditions.

    """

    def __init__(self, module, message_size=10000000, shared_param=False):
        super(DistributedDataParallel, self).__init__()
        self.warn_on_half = True if dist._backend == dist.dist_backend.GLOO else False
        self.shared_param = shared_param
        self.message_size = message_size

        #reference to last iterations parameters to see if anything has changed
        self.param_refs = []

        self.reduction_stream = torch.cuda.Stream()

        self.module = module
        self.param_list = list(self.module.parameters())

        if dist._backend == dist.dist_backend.NCCL:
            for param in self.param_list:
                assert param.is_cuda, "NCCL backend only supports model parameters to be on GPU."

        self.record = []
        self.create_hooks()

        flat_dist_call([param.data for param in self.module.parameters()], dist.broadcast, (0,) )

    def create_hooks(self):
        #all reduce gradient hook
        def allreduce_params():
            if not self.needs_reduction:
                return
            self.needs_reduction = False

            #parameter ordering refresh
            if self.needs_refresh and not self.shared_param:
                t_record = torch.cuda.IntTensor(self.record)
                dist.broadcast(t_record, 0)
                self.record = [int(entry) for entry in t_record]
                self.needs_refresh = False

            grads = [param.grad.data for param in self.module.parameters() if param.grad is not None]
            flat_dist_call(grads, dist.all_reduce)

        def flush_buckets():
            if not self.needs_reduction:
                return
            self.needs_reduction = False

            grads = []
            for i in range(self.ready_end, len(self.param_state)):
                param = self.param_refs[self.record[i]]
                if param.grad is not None:
                    grads.append(param.grad.data)
            grads = [param.grad.data for param in self.ready_params] + grads

            if(len(grads)>0):
                orig_stream = torch.cuda.current_stream()
                with torch.cuda.stream(self.reduction_stream):
                    self.reduction_stream.wait_stream(orig_stream)
                    flat_dist_call(grads, dist.all_reduce)

            torch.cuda.current_stream().wait_stream(self.reduction_stream)

        for param_i, param in enumerate(list(self.module.parameters())):
            def wrapper(param_i):

                def allreduce_hook(*unused):
                    if self.needs_refresh:
                        self.record.append(param_i)
                        Variable._execution_engine.queue_callback(allreduce_params)
                    else:
                        Variable._execution_engine.queue_callback(flush_buckets)
                        self.comm_ready_buckets(self.record.index(param_i))


                if param.requires_grad:
                    param.register_hook(allreduce_hook)
            wrapper(param_i)


    def comm_ready_buckets(self, param_ind):

        if self.param_state[param_ind] != 0:
            raise RuntimeError("Error: Your model uses shared parameters, DDP flag shared_params must be set to True in initialization.")


        if self.param_state[self.ready_end] == 0:
            self.param_state[param_ind] = 1
            return


        while self.ready_end < len(self.param_state) and self.param_state[self.ready_end] == 1:
            self.ready_params.append(self.param_refs[self.record[self.ready_end]])
            self.ready_numel += self.ready_params[-1].numel()
            self.ready_end += 1


        if self.ready_numel < self.message_size:
            self.param_state[param_ind] = 1
            return

        grads = [param.grad.data for param in self.ready_params]

        bucket = []
        bucket_inds = []
        while grads:
            bucket.append(grads.pop(0))

            cumm_size = 0
            for ten in bucket:
                cumm_size += ten.numel()

            if cumm_size < self.message_size:
                continue

            evt = torch.cuda.Event()
            evt.record(torch.cuda.current_stream())
            evt.wait(stream=self.reduction_stream)

            with torch.cuda.stream(self.reduction_stream):
                flat_dist_call(bucket, dist.all_reduce)

            for i in range(self.ready_start, self.ready_start+len(bucket)):
                self.param_state[i] = 2
                self.ready_params.pop(0)

        self.param_state[param_ind] = 1

    def forward(self, *inputs, **kwargs):

        param_list = [param for param in list(self.module.parameters()) if param.requires_grad]


        #Force needs_refresh to True if there are shared params
        #this will force it to always, only call flush_buckets which is safe
        #for shared parameters in the model.
        #Parentheses are not necessary for correct order of operations, but make the intent clearer.
        if (not self.param_refs) or self.shared_param:
            self.needs_refresh = True
        else:
            self.needs_refresh = (
                (len(param_list) != len(self.param_refs)) or any(
                    [param1 is not param2 for param1, param2 in zip(param_list, self.param_refs)]))

        if  self.needs_refresh:
            self.record = []


        self.param_state = [0 for i in range(len(param_list))]
        self.param_refs = param_list
        self.needs_reduction = True

        self.ready_start = 0
        self.ready_end   = 0
        self.ready_params = []
        self.ready_numel = 0

        return self.module(*inputs, **kwargs)
