# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import itertools
import os
import sys
from absl import app, flags, logging
from apex import amp, parallel, optimizers as apex_optim

from dlrm.data.feature_spec import FeatureSpec
from dlrm.model.distributed import DistributedDlrm
from dlrm.utils import distributed as dist
from dlrm.utils.checkpointing.distributed import make_distributed_checkpoint_writer, make_distributed_checkpoint_loader
from dlrm.utils.distributed import get_gpu_batch_sizes, get_device_mapping, is_main_process, is_distributed

import datetime
from time import time

import dllogger
import numpy as np
import torch
from absl import app, flags

import dlrm.scripts.utils as utils
from dlrm.data.data_loader import get_data_loaders
from dlrm.data.utils import prefetcher, get_embedding_sizes

FLAGS = flags.FLAGS

# Basic run settings
flags.DEFINE_enum("mode", default='train', enum_values=['train', 'test', 'inference_benchmark'],
                  help="Select task to be performed")
flags.DEFINE_integer("seed", 12345, "Random seed")

# Training flags
flags.DEFINE_integer("batch_size", 65536, "Batch size used for training")
flags.DEFINE_integer("test_batch_size", 65536, "Batch size used for testing/validation")
flags.DEFINE_float("lr", 24, "Base learning rate")
flags.DEFINE_integer("epochs", 1, "Number of epochs to train for")
flags.DEFINE_integer("max_steps", None, "Stop training after doing this many optimization steps")

# Learning rate schedule flags
flags.DEFINE_integer("warmup_factor", 0, "Learning rate warmup factor. Must be a non-negative integer")
flags.DEFINE_integer("warmup_steps", 8000, "Number of warmup optimization steps")
flags.DEFINE_integer("decay_steps", 24000,
                     "Polynomial learning rate decay steps. If equal to 0 will not do any decaying")
flags.DEFINE_integer("decay_start_step", 48000,
                     "Optimization step after which to start decaying the learning rate, "
                     "if None will start decaying right after the warmup phase is completed")
flags.DEFINE_integer("decay_power", 2, "Polynomial learning rate decay power")
flags.DEFINE_float("decay_end_lr", 0, "LR after the decay ends")

# Model configuration
flags.DEFINE_enum("embedding_type", "custom_cuda",
                  ["joint", "custom_cuda", "multi_table", "joint_sparse", "joint_fused"],
                  help="The type of the embedding operation to use")
flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of embedding space for categorical features")
flags.DEFINE_list("top_mlp_sizes", [1024, 1024, 512, 256, 1], "Linear layer sizes for the top MLP")
flags.DEFINE_list("bottom_mlp_sizes", [512, 256, 128], "Linear layer sizes for the bottom MLP")
flags.DEFINE_enum("interaction_op", default="cuda_dot", enum_values=["cuda_dot", "dot", "cat"],
                  help="Type of interaction operation to perform.")

# Data configuration
flags.DEFINE_string("dataset", None, "Path to dataset directory")
flags.DEFINE_string("feature_spec", default="feature_spec.yaml",
                    help="Name of the feature spec file in the dataset directory")
flags.DEFINE_enum("dataset_type", default="parametric", enum_values=['synthetic_gpu', 'parametric'],
                  help='The type of the dataset to use')
flags.DEFINE_boolean("shuffle_batch_order", False, "Read batch in train dataset by random order", short_name="shuffle")

flags.DEFINE_integer("max_table_size", None,
                     "Maximum number of rows per embedding table, "
                     "by default equal to the number of unique values for each categorical variable")
flags.DEFINE_boolean("hash_indices", False,
                     "If True the model will compute `index := index % table size` "
                     "to ensure that the indices match table sizes")

# Synthetic data configuration
flags.DEFINE_integer("synthetic_dataset_num_entries", default=int(2 ** 15 * 1024),
                     help="Number of samples per epoch for the synthetic dataset")
flags.DEFINE_list("synthetic_dataset_table_sizes", default=','.join(26 * [str(10 ** 5)]),
                  help="Cardinalities of variables to use with the synthetic dataset.")
flags.DEFINE_integer("synthetic_dataset_numerical_features", default='13',
                     help="Number of numerical features to use with the synthetic dataset")
flags.DEFINE_boolean("synthetic_dataset_use_feature_spec", default=False,
                     help="Create a temporary synthetic dataset based on a real one. "
                          "Uses --dataset and --feature_spec"
                          "Overrides synthetic_dataset_table_sizes and synthetic_dataset_numerical_features."
                          "--synthetic_dataset_num_entries is still required")

# Checkpointing
flags.DEFINE_string("load_checkpoint_path", None, "Path from which to load a checkpoint")
flags.DEFINE_string("save_checkpoint_path", None, "Path to which to save the training checkpoints")

# Saving and logging flags
flags.DEFINE_string("log_path", "./log.json", "Destination for the log file with various results and statistics")
flags.DEFINE_integer("test_freq", None,
                     "Number of optimization steps between validations. If None will test after each epoch")
flags.DEFINE_float("test_after", 0, "Don't test the model unless this many epochs has been completed")
flags.DEFINE_integer("print_freq", 200, "Number of optimizations steps between printing training status to stdout")
flags.DEFINE_integer("benchmark_warmup_steps", 0,
                     "Number of initial iterations to exclude from throughput measurements")

# Machine setting flags
flags.DEFINE_string("base_device", "cuda", "Device to run the majority of the model operations")
flags.DEFINE_boolean("amp", False, "If True the script will use Automatic Mixed Precision")

flags.DEFINE_boolean("cuda_graphs", False, "Use CUDA Graphs")

# inference benchmark
flags.DEFINE_list("inference_benchmark_batch_sizes", default=[1, 64, 4096],
                  help="Batch sizes for inference throughput and latency measurements")
flags.DEFINE_integer("inference_benchmark_steps", 200,
                     "Number of steps for measuring inference latency and throughput")

# Miscellaneous
flags.DEFINE_float("auc_threshold", None, "Stop the training after achieving this AUC")
flags.DEFINE_boolean("optimized_mlp", True, "Use an optimized implementation of MLP from apex")
flags.DEFINE_enum("auc_device", default="GPU", enum_values=['GPU', 'CPU'],
                  help="Specifies where ROC AUC metric is calculated")

flags.DEFINE_string("backend", "nccl", "Backend to use for distributed training. Default nccl")
flags.DEFINE_boolean("bottom_features_ordered", False,
                     "Sort features from the bottom model, useful when using saved "
                     "checkpoint in different device configurations")
flags.DEFINE_boolean("freeze_mlps", False,
                     "For debug and benchmarking. Don't perform the weight update for MLPs.")
flags.DEFINE_boolean("freeze_embeddings", False,
                     "For debug and benchmarking. Don't perform the weight update for the embeddings.")
flags.DEFINE_boolean("Adam_embedding_optimizer", False, "Swaps embedding optimizer to Adam")
flags.DEFINE_boolean("Adam_MLP_optimizer", False, "Swaps MLP optimizer to Adam")


def validate_flags(cat_feature_count):
    if FLAGS.max_table_size is not None and not FLAGS.hash_indices:
        raise ValueError('Hash indices must be True when setting a max_table_size')

    if FLAGS.base_device == 'cpu':
        if FLAGS.embedding_type in ('joint_fused', 'joint_sparse'):
            print('WARNING: CUDA joint embeddings are not supported on CPU')
            FLAGS.embedding_type = 'joint'

        if FLAGS.amp:
            print('WARNING: Automatic mixed precision not supported on CPU')
            FLAGS.amp = False

        if FLAGS.optimized_mlp:
            print('WARNING: Optimized MLP is not supported on CPU')
            FLAGS.optimized_mlp = False

    if FLAGS.embedding_type == 'custom_cuda':
        if (not is_distributed()) and FLAGS.embedding_dim == 128 and cat_feature_count == 26:
            FLAGS.embedding_type = 'joint_fused'
        else:
            FLAGS.embedding_type = 'joint_sparse'

    if FLAGS.embedding_type == 'joint_fused' and FLAGS.embedding_dim != 128:
        print('WARNING: Joint fused can be used only with embedding_dim=128. Changed embedding type to joint_sparse.')
        FLAGS.embedding_type = 'joint_sparse'

    if FLAGS.dataset is None and (FLAGS.dataset_type != 'synthetic_gpu' or
                                  FLAGS.synthetic_dataset_use_feature_spec):
        raise ValueError('Dataset argument has to specify a path to the dataset')

    FLAGS.inference_benchmark_batch_sizes = [int(x) for x in FLAGS.inference_benchmark_batch_sizes]
    FLAGS.top_mlp_sizes = [int(x) for x in FLAGS.top_mlp_sizes]
    FLAGS.bottom_mlp_sizes = [int(x) for x in FLAGS.bottom_mlp_sizes]

    # TODO check that bottom_mlp ends in embedding_dim size


def load_feature_spec(flags):
    if flags.dataset_type == 'synthetic_gpu' and not flags.synthetic_dataset_use_feature_spec:
        num_numerical = flags.synthetic_dataset_numerical_features
        categorical_sizes = [int(s) for s in FLAGS.synthetic_dataset_table_sizes]
        return FeatureSpec.get_default_feature_spec(number_of_numerical_features=num_numerical,
                                                    categorical_feature_cardinalities=categorical_sizes)
    fspec_path = os.path.join(flags.dataset, flags.feature_spec)
    return FeatureSpec.from_yaml(fspec_path)


class CudaGraphWrapper:
    def __init__(self, model, train_step, parallelize,
                 zero_grad, cuda_graphs=False, warmup_steps=20):

        self.cuda_graphs = cuda_graphs
        self.warmup_iters = warmup_steps
        self.graph = None
        self.stream = None
        self.static_args = None

        self.model = model

        self._parallelize = parallelize
        self._train_step = train_step
        self._zero_grad = zero_grad

        self.loss = None
        self.step = -1

        if cuda_graphs:
            self.stream = torch.cuda.Stream()
        else:
            # if not using graphs, parallelize the model immediately
            # otherwise do this in the warmup phase under the graph stream
            self.model = self._parallelize(self.model)
            self.stream = torch.cuda.default_stream()

    def _copy_input_data(self, *train_step_args):
        if len(train_step_args) != len(self.static_args):
            raise ValueError(f'Expected {len(self.static_args)} arguments to train step'
                             f'Got: {len(train_step_args)}')

        for data, placeholder in zip(train_step_args, self.static_args):
            if placeholder is None:
                continue
            placeholder.copy_(data)

    def _cuda_graph_capture(self, *train_step_args):
        self._copy_input_data(*train_step_args)
        self.graph = torch.cuda.CUDAGraph()
        self._zero_grad(self.model)
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.loss = self._train_step(self.model, *self.static_args)
        return self.loss

    def _cuda_graph_replay(self, *train_step_args):
        self._copy_input_data(*train_step_args)
        self.graph.replay()

    def _warmup_step(self, *train_step_args):
        with torch.cuda.stream(self.stream):
            if self.step == 0:
                self.model = self._parallelize(self.model)
                self.static_args = list(train_step_args)
            else:
                self._copy_input_data(*train_step_args)

            self._zero_grad(self.model)
            self.loss = self._train_step(self.model, *self.static_args)
            return self.loss

    def train_step(self, *train_step_args):
        self.step += 1

        if not self.cuda_graphs:
            self._zero_grad(self.model)
            self.loss = self._train_step(self.model, *train_step_args)
            return self.loss

        if self.step == 0:
            self.stream.wait_stream(torch.cuda.current_stream())

        if self.step < self.warmup_iters:
            return self._warmup_step(*train_step_args)

        if self.graph is None:
            torch.cuda.synchronize()
            self._cuda_graph_capture(*train_step_args)

        self._cuda_graph_replay(*train_step_args)
        return self.loss


def inference_benchmark(*args, cuda_graphs=False, **kwargs):
    if cuda_graphs:
        return inference_benchmark_graphed(*args, **kwargs)
    else:
        return inference_benchmark_nongraphed(*args, **kwargs)


def inference_benchmark_nongraphed(model, data_loader, num_batches=100):
    model.eval()
    base_device = FLAGS.base_device
    latencies = []

    y_true = []
    y_score = []

    with torch.no_grad():
        for step, (numerical_features, categorical_features, click) in enumerate(data_loader):
            if step > num_batches:
                break

            step_start_time = time()

            numerical_features = numerical_features.to(base_device)
            if FLAGS.amp:
                numerical_features = numerical_features.half()

            categorical_features = categorical_features.to(device=base_device, dtype=torch.int64)

            inference_result = model(numerical_features, categorical_features).squeeze()
            torch.cuda.synchronize()
            step_time = time() - step_start_time

            if step >= FLAGS.benchmark_warmup_steps:
                latencies.append(step_time)

            y_true.append(click)
            y_score.append(inference_result.reshape([-1]).clone())

    y_true = torch.cat(y_true)
    y_score = torch.sigmoid(torch.cat(y_score)).float()
    auc = utils.roc_auc_score(y_true, y_score)
    print('auc: ', auc)

    return latencies


def inference_benchmark_graphed(model, data_loader, num_batches=100):
    model.eval()
    base_device = FLAGS.base_device
    latencies = []

    data_iter = iter(data_loader)
    numerical, categorical, _ = next(data_iter)

    # Warmup before capture
    s = torch.cuda.Stream()
    static_numerical = numerical.to(base_device)
    static_categorical = categorical.to(device=base_device, dtype=torch.int64)
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(10):
            if FLAGS.amp:
                numerical = static_numerical.half()
            else:
                numerical = static_numerical
            inference_result = model(numerical, static_categorical).squeeze()

    torch.cuda.synchronize()

    # Graph capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if FLAGS.amp:
            numerical = static_numerical.half()
        else:
            numerical = static_numerical
        inference_result = model(numerical, static_categorical).squeeze()

    torch.cuda.synchronize()
    # Inference
    y_true = []
    y_score = []

    with torch.no_grad():
        for step, (numerical_features, categorical_features, click) in enumerate(data_loader):
            if step > num_batches:
                break
            torch.cuda.synchronize()
            step_start_time = time()

            numerical_features = numerical_features.to(base_device)
            categorical_features = categorical_features.to(device=base_device, dtype=torch.int64)

            static_categorical.copy_(categorical_features)
            static_numerical.copy_(numerical_features)
            graph.replay()
            torch.cuda.synchronize()
            step_time = time() - step_start_time

            if step >= FLAGS.benchmark_warmup_steps:
                latencies.append(step_time)
            y_true.append(click)
            y_score.append(inference_result.reshape([-1]).clone())
    y_true = torch.cat(y_true)
    y_score = torch.sigmoid(torch.cat(y_score)).float()
    auc = utils.roc_auc_score(y_true, y_score)
    print('auc: ', auc)
    return latencies


def main(argv):
    torch.manual_seed(FLAGS.seed)

    use_gpu = "cpu" not in FLAGS.base_device.lower()
    rank, world_size, gpu = dist.init_distributed_mode(backend=FLAGS.backend, use_gpu=use_gpu)
    device = FLAGS.base_device

    feature_spec = load_feature_spec(FLAGS)

    cat_feature_count = len(get_embedding_sizes(feature_spec, None))
    validate_flags(cat_feature_count)

    if is_main_process():
        utils.init_logging(log_path=FLAGS.log_path)
        dllogger.log(data=FLAGS.flag_values_dict(), step='PARAMETER')

    FLAGS.set_default("test_batch_size", FLAGS.test_batch_size // world_size * world_size)

    feature_spec = load_feature_spec(FLAGS)
    world_embedding_sizes = get_embedding_sizes(feature_spec, max_table_size=FLAGS.max_table_size)
    world_categorical_feature_sizes = np.asarray(world_embedding_sizes)
    device_mapping = get_device_mapping(world_embedding_sizes, num_gpus=world_size)

    batch_sizes_per_gpu = get_gpu_batch_sizes(FLAGS.batch_size, num_gpus=world_size)
    batch_indices = tuple(np.cumsum([0] + list(batch_sizes_per_gpu)))  # todo what does this do

    # Embedding sizes for each GPU
    categorical_feature_sizes = world_categorical_feature_sizes[device_mapping['embedding'][rank]].tolist()
    num_numerical_features = feature_spec.get_number_of_numerical_features()

    bottom_mlp_sizes = FLAGS.bottom_mlp_sizes if rank == device_mapping['bottom_mlp'] else None

    data_loader_train, data_loader_test = get_data_loaders(FLAGS, device_mapping=device_mapping,
                                                           feature_spec=feature_spec)

    model = DistributedDlrm(
        vectors_per_gpu=device_mapping['vectors_per_gpu'],
        embedding_device_mapping=device_mapping['embedding'],
        embedding_type=FLAGS.embedding_type,
        embedding_dim=FLAGS.embedding_dim,
        world_num_categorical_features=len(world_categorical_feature_sizes),
        categorical_feature_sizes=categorical_feature_sizes,
        num_numerical_features=num_numerical_features,
        hash_indices=FLAGS.hash_indices,
        bottom_mlp_sizes=bottom_mlp_sizes,
        top_mlp_sizes=FLAGS.top_mlp_sizes,
        interaction_op=FLAGS.interaction_op,
        fp16=FLAGS.amp,
        use_cpp_mlp=FLAGS.optimized_mlp,
        bottom_features_ordered=FLAGS.bottom_features_ordered,
        device=device
    )

    dist.setup_distributed_print(is_main_process())

    # DDP introduces a gradient average through allreduce(mean), which doesn't apply to bottom model.
    # Compensate it with further scaling lr
    if FLAGS.Adam_embedding_optimizer:
        embedding_model_parallel_lr = FLAGS.lr
    else:
        embedding_model_parallel_lr = FLAGS.lr / world_size

    if FLAGS.Adam_MLP_optimizer:
        MLP_model_parallel_lr = FLAGS.lr
    else:
        MLP_model_parallel_lr = FLAGS.lr / world_size

    data_parallel_lr = FLAGS.lr

    if is_main_process():
        mlp_params = [
            {'params': list(model.top_model.parameters()), 'lr': data_parallel_lr},
            {'params': list(model.bottom_model.mlp.parameters()), 'lr': MLP_model_parallel_lr}
        ]
        mlp_lrs = [data_parallel_lr, MLP_model_parallel_lr]
    else:
        mlp_params = [
            {'params': list(model.top_model.parameters()), 'lr': data_parallel_lr}
        ]
        mlp_lrs = [data_parallel_lr]

    if FLAGS.Adam_MLP_optimizer:
        mlp_optimizer = apex_optim.FusedAdam(mlp_params)
    else:
        mlp_optimizer = apex_optim.FusedSGD(mlp_params)

    embedding_params = [{
        'params': list(model.bottom_model.embeddings.parameters()),
        'lr': embedding_model_parallel_lr
    }]
    embedding_lrs = [embedding_model_parallel_lr]

    if FLAGS.Adam_embedding_optimizer:
        embedding_optimizer = torch.optim.SparseAdam(embedding_params)
    else:
        embedding_optimizer = torch.optim.SGD(embedding_params)

    checkpoint_writer = make_distributed_checkpoint_writer(
        device_mapping=device_mapping,
        rank=rank,
        is_main_process=is_main_process(),
        config=FLAGS.flag_values_dict()
    )

    checkpoint_loader = make_distributed_checkpoint_loader(device_mapping=device_mapping, rank=rank)

    if FLAGS.load_checkpoint_path:
        checkpoint_loader.load_checkpoint(model, FLAGS.load_checkpoint_path)
        model.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=FLAGS.amp, growth_interval=int(1e9))

    def parallelize(model):
        if world_size <= 1:
            return model

        if use_gpu:
            model.top_model = parallel.DistributedDataParallel(model.top_model)
        else:  # Use other backend for CPU
            model.top_model = torch.nn.parallel.DistributedDataParallel(model.top_model)
        return model

    if FLAGS.mode == 'test':
        model = parallelize(model)
        auc, valid_loss = dist_evaluate(model, data_loader_test)

        results = {'best_auc': auc, 'best_validation_loss': valid_loss}
        if is_main_process():
            dllogger.log(data=results, step=tuple())
        return
    elif FLAGS.mode == 'inference_benchmark':
        if world_size > 1:
            raise ValueError('Inference benchmark only supports singleGPU mode.')

        results = {}

        if FLAGS.amp:
            # can use pure FP16 for inference
            model = model.half()

        for batch_size in FLAGS.inference_benchmark_batch_sizes:
            FLAGS.test_batch_size = batch_size
            _, data_loader_test = get_data_loaders(FLAGS, device_mapping=device_mapping, feature_spec=feature_spec)

            latencies = inference_benchmark(model=model, data_loader=data_loader_test,
                                            num_batches=FLAGS.inference_benchmark_steps,
                                            cuda_graphs=FLAGS.cuda_graphs)

            # drop the first 10 as a warmup
            latencies = latencies[10:]

            mean_latency = np.mean(latencies)
            mean_inference_throughput = batch_size / mean_latency
            subresult = {f'mean_inference_latency_batch_{batch_size}': mean_latency,
                         f'mean_inference_throughput_batch_{batch_size}': mean_inference_throughput}
            results.update(subresult)
        if is_main_process():
            dllogger.log(data=results, step=tuple())
        return

    if FLAGS.save_checkpoint_path and not FLAGS.bottom_features_ordered and is_main_process():
        logging.warning("Saving checkpoint without --bottom_features_ordered flag will result in "
                        "a device-order dependent model. Consider using --bottom_features_ordered "
                        "if you plan to load the checkpoint in different device configurations.")

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")

    # Print per 16384 * 2000 samples by default
    default_print_freq = 16384 * 2000 // FLAGS.batch_size
    print_freq = default_print_freq if FLAGS.print_freq is None else FLAGS.print_freq

    # last one will be dropped in the training loop
    steps_per_epoch = len(data_loader_train) - 1
    test_freq = FLAGS.test_freq if FLAGS.test_freq is not None else steps_per_epoch - 2

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{avg:.8f}'))
    metric_logger.add_meter('step_time', utils.SmoothedValue(window_size=1, fmt='{avg:.6f}'))
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    # Accumulating loss on GPU to avoid memcpyD2H every step
    moving_loss = torch.zeros(1, device=device)

    lr_scheduler = utils.LearningRateScheduler(optimizers=[mlp_optimizer, embedding_optimizer],
                                               base_lrs=[mlp_lrs, embedding_lrs],
                                               warmup_steps=FLAGS.warmup_steps,
                                               warmup_factor=FLAGS.warmup_factor,
                                               decay_start_step=FLAGS.decay_start_step,
                                               decay_steps=FLAGS.decay_steps,
                                               decay_power=FLAGS.decay_power,
                                               end_lr_factor=FLAGS.decay_end_lr / FLAGS.lr)

    def zero_grad(model):
        if FLAGS.Adam_embedding_optimizer or FLAGS.Adam_MLP_optimizer:
            model.zero_grad()
        else:
            # We don't need to accumulate gradient. Set grad to None is faster than optimizer.zero_grad()
            for param_group in itertools.chain(embedding_optimizer.param_groups, mlp_optimizer.param_groups):
                for param in param_group['params']:
                    param.grad = None

    def forward_backward(model, *args):

        numerical_features, categorical_features, click = args
        with torch.cuda.amp.autocast(enabled=FLAGS.amp):
            output = model(numerical_features, categorical_features, batch_sizes_per_gpu).squeeze()
            loss = loss_fn(output, click[batch_indices[rank]: batch_indices[rank + 1]])

        scaler.scale(loss).backward()

        return loss

    def weight_update():
        if not FLAGS.freeze_mlps:
            if FLAGS.Adam_MLP_optimizer:
                scale_MLP_gradients(mlp_optimizer, world_size)
            scaler.step(mlp_optimizer)

        if not FLAGS.freeze_embeddings:
            if FLAGS.Adam_embedding_optimizer:
                scale_embeddings_gradients(embedding_optimizer, world_size)
            scaler.unscale_(embedding_optimizer)
            embedding_optimizer.step()

        scaler.update()

    trainer = CudaGraphWrapper(model, forward_backward, parallelize, zero_grad,
                               cuda_graphs=FLAGS.cuda_graphs)

    data_stream = torch.cuda.Stream()
    timer = utils.StepTimer()

    best_validation_loss = 1e6
    best_auc = 0
    best_epoch = 0
    start_time = time()

    for epoch in range(FLAGS.epochs):
        epoch_start_time = time()

        batch_iter = prefetcher(iter(data_loader_train), data_stream)

        for step in range(len(data_loader_train)):
            numerical_features, categorical_features, click = next(batch_iter)
            timer.click(synchronize=(device == 'cuda'))

            global_step = steps_per_epoch * epoch + step

            if FLAGS.max_steps and global_step > FLAGS.max_steps:
                print(f"Reached max global steps of {FLAGS.max_steps}. Stopping.")
                break

            # One of the batches will be smaller because the dataset size
            # isn't necessarily a multiple of the batch size. #TODO isn't dropping here a change of behavior
            if click.shape[0] != FLAGS.batch_size:
                continue

            lr_scheduler.step()
            loss = trainer.train_step(numerical_features, categorical_features, click)

            # need to wait for the gradients before the weight update
            torch.cuda.current_stream().wait_stream(trainer.stream)
            weight_update()
            moving_loss += loss

            if timer.measured is None:
                # first iteration, no step time etc. to print
                continue

            if step == 0:
                print(f"Started epoch {epoch}...")
            elif step % print_freq == 0:
                # Averaging across a print_freq period to reduce the error.
                # An accurate timing needs synchronize which would slow things down.

                # only check for nan every print_freq steps
                if torch.any(torch.isnan(loss)):
                    print('NaN loss encountered.')
                    break

                if global_step < FLAGS.benchmark_warmup_steps:
                    metric_logger.update(
                        loss=moving_loss.item() / print_freq,
                        lr=mlp_optimizer.param_groups[0]["lr"])
                else:
                    metric_logger.update(
                        step_time=timer.measured,
                        loss=moving_loss.item() / print_freq,
                        lr=mlp_optimizer.param_groups[0]["lr"])

                eta_str = datetime.timedelta(seconds=int(metric_logger.step_time.global_avg * (steps_per_epoch - step)))
                metric_logger.print(header=f"Epoch:[{epoch}/{FLAGS.epochs}] [{step}/{steps_per_epoch}]  eta: {eta_str}")

                moving_loss = 0.

            if global_step % test_freq == 0 and global_step > 0 and global_step / steps_per_epoch >= FLAGS.test_after:
                auc, validation_loss = dist_evaluate(trainer.model, data_loader_test)

                if auc is None:
                    continue

                print(f"Epoch {epoch} step {step}. auc {auc:.6f}")
                stop_time = time()

                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch + ((step + 1) / steps_per_epoch)

                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss

                if FLAGS.auc_threshold and auc >= FLAGS.auc_threshold:
                    run_time_s = int(stop_time - start_time)
                    print(f"Hit target accuracy AUC {FLAGS.auc_threshold} at epoch "
                          f"{global_step / steps_per_epoch:.2f} in {run_time_s}s. ")
                    sys.exit()

        epoch_stop_time = time()
        epoch_time_s = epoch_stop_time - epoch_start_time
        print(f"Finished epoch {epoch} in {datetime.timedelta(seconds=int(epoch_time_s))}. ")

    avg_throughput = FLAGS.batch_size / metric_logger.step_time.avg

    if FLAGS.save_checkpoint_path:
        checkpoint_writer.save_checkpoint(model, FLAGS.save_checkpoint_path, epoch, step)

    results = {'best_auc': best_auc,
               'best_validation_loss': best_validation_loss,
               'training_loss' : metric_logger.meters['loss'].avg,
               'best_epoch': best_epoch,
               'average_train_throughput': avg_throughput}

    if is_main_process():
        dllogger.log(data=results, step=tuple())


def scale_MLP_gradients(mlp_optimizer: torch.optim.Optimizer, world_size: int):
    for param_group in mlp_optimizer.param_groups[1:]:  # Omitting top MLP
        for param in param_group['params']:
            param.grad.div_(world_size)


def scale_embeddings_gradients(embedding_optimizer: torch.optim.Optimizer, world_size: int):
    for param_group in embedding_optimizer.param_groups:
        for param in param_group['params']:
            if param.grad != None:
                param.grad.div_(world_size)


def dist_evaluate(model, data_loader):
    """Test distributed DLRM model

    Args:
        model (DistDLRM):
        data_loader (torch.utils.data.DataLoader):
    """
    model.eval()

    device = FLAGS.base_device
    world_size = dist.get_world_size()

    batch_sizes_per_gpu = [FLAGS.test_batch_size // world_size for _ in range(world_size)]
    test_batch_size = sum(batch_sizes_per_gpu)

    if FLAGS.test_batch_size != test_batch_size:
        print(f"Rounded test_batch_size to {test_batch_size}")

    # Test bach size could be big, make sure it prints
    default_print_freq = max(524288 * 100 // test_batch_size, 1)
    print_freq = default_print_freq if FLAGS.print_freq is None else FLAGS.print_freq

    steps_per_epoch = len(data_loader)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('step_time', utils.SmoothedValue(window_size=1, fmt='{avg:.4f}'))

    with torch.no_grad():
        timer = utils.StepTimer()

        # ROC can be computed per batch and then compute AUC globally, but I don't have the code.
        # So pack all the outputs and labels together to compute AUC. y_true and y_score naming follows sklearn
        y_true = []
        y_score = []
        data_stream = torch.cuda.Stream()

        batch_iter = prefetcher(iter(data_loader), data_stream)
        loss_fn = torch.nn.BCELoss(reduction="mean")

        timer.click(synchronize=(device=='cuda'))
        for step in range(len(data_loader)):
            numerical_features, categorical_features, click = next(batch_iter)
            torch.cuda.synchronize()

            last_batch_size = None
            if click.shape[0] != test_batch_size:  # last batch
                last_batch_size = click.shape[0]
                padding_size = test_batch_size - last_batch_size

                if numerical_features is not None:
                    padding_numerical = torch.empty(
                        padding_size, numerical_features.shape[1],
                        device=numerical_features.device, dtype=numerical_features.dtype)
                    numerical_features = torch.cat((numerical_features, padding_numerical), dim=0)

                if categorical_features is not None:
                    padding_categorical = torch.ones(
                        padding_size, categorical_features.shape[1],
                        device=categorical_features.device, dtype=categorical_features.dtype)
                    categorical_features = torch.cat((categorical_features, padding_categorical), dim=0)

            with torch.cuda.amp.autocast(enabled=FLAGS.amp):
                output = model(numerical_features, categorical_features, batch_sizes_per_gpu)
                output = output.squeeze()
                output = output.float()

            if world_size > 1:
                output_receive_buffer = torch.empty(test_batch_size, device=device)
                torch.distributed.all_gather(list(output_receive_buffer.split(batch_sizes_per_gpu)), output)
                output = output_receive_buffer

            if last_batch_size is not None:
                output = output[:last_batch_size]

            if FLAGS.auc_device == "CPU":
                click = click.cpu()
                output = output.cpu()

            y_true.append(click)
            y_score.append(output)

            timer.click(synchronize=(device == 'cuda'))

            if timer.measured is not None:
                metric_logger.update(step_time=timer.measured)
                if step % print_freq == 0 and step > 0:
                    metric_logger.print(header=f"Test: [{step}/{steps_per_epoch}]")

        if is_main_process():
            y_true = torch.cat(y_true)
            y_score = torch.sigmoid(torch.cat(y_score)).float()
            auc = utils.roc_auc_score(y_true, y_score)
            loss = loss_fn(y_score, y_true).item()
            print(f'test loss: {loss:.8f}', )
        else:
            auc = None
            loss = None

        if world_size > 1:
            torch.distributed.barrier()

    model.train()

    return auc, loss


if __name__ == '__main__':
    app.run(main)
