# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from nvidia import dali
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.mxnet import DALIClassificationIterator
import horovod.mxnet as hvd


def add_dali_args(parser):
    group = parser.add_argument_group('DALI data backend', 'entire group applies only to dali data backend')
    group.add_argument('--dali-separ-val', action='store_true',
                      help='each process will perform independent validation on whole val-set')
    group.add_argument('--dali-threads', type=int, default=3, help="number of threads" +\
                       "per GPU for DALI")
    group.add_argument('--dali-validation-threads', type=int, default=10, help="number of threads" +\
                       "per GPU for DALI for validation")
    group.add_argument('--dali-prefetch-queue', type=int, default=2, help="DALI prefetch queue depth")
    group.add_argument('--dali-nvjpeg-memory-padding', type=int, default=64, help="Memory padding value for nvJPEG (in MB)")
    group.add_argument('--dali-fuse-decoder', type=int, default=1, help="0 or 1 whether to fuse decoder or not")
    return parser


class HybridTrainPipe(Pipeline):
    def __init__(self, args, batch_size, num_threads, device_id, rec_path, idx_path,
                 shard_id, num_shards, crop_shape, nvjpeg_padding, prefetch_queue=3,
                 output_layout=types.NCHW, pad_output=True, dtype='float16', dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id, prefetch_queue_depth = prefetch_queue)
        self.input = ops.MXNetReader(path=[rec_path], index_path=[idx_path],
                                     random_shuffle=True, shard_id=shard_id, num_shards=num_shards)

        if dali_cpu:
            dali_device = "cpu"
            if args.dali_fuse_decoder:
                self.decode = ops.HostDecoderRandomCrop(device=dali_device, output_type=types.RGB)
            else:
                self.decode = ops.HostDecoder(device=dali_device, output_type=types.RGB)
        else:
            dali_device = "gpu"
            if args.dali_fuse_decoder:
                self.decode = ops.nvJPEGDecoderRandomCrop(device="mixed", output_type=types.RGB,
                                                          device_memory_padding=nvjpeg_padding, host_memory_padding=nvjpeg_padding)
            else:
                self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB,
                                                device_memory_padding=nvjpeg_padding, host_memory_padding=nvjpeg_padding)

        if args.dali_fuse_decoder:
            self.resize = ops.Resize(device=dali_device, resize_x=crop_shape[1], resize_y=crop_shape[0])
        else:
            self.resize = ops.RandomResizedCrop(device=dali_device, size=crop_shape)

        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT16 if dtype == 'float16' else types.FLOAT,
                                            output_layout=output_layout, crop=crop_shape, pad_output=pad_output,
                                            image_type=types.RGB, mean=args.rgb_mean, std=args.rgb_std)
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")

        images = self.decode(self.jpegs)
        images = self.resize(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, args, batch_size, num_threads, device_id, rec_path, idx_path,
                 shard_id, num_shards, crop_shape, nvjpeg_padding, prefetch_queue=3, resize_shp=None,
                 output_layout=types.NCHW, pad_output=True, dtype='float16', dali_cpu=False):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id, prefetch_queue_depth=prefetch_queue)
        self.input = ops.MXNetReader(path=[rec_path], index_path=[idx_path],
                                     random_shuffle=False, shard_id=shard_id, num_shards=num_shards)

        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoder(device=dali_device, output_type=types.RGB)
        else:
            dali_device = "gpu"
            self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB,
                                            device_memory_padding=nvjpeg_padding,
                                            host_memory_padding=nvjpeg_padding)
        self.resize = ops.Resize(device=dali_device, resize_shorter=resize_shp) if resize_shp else None
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT16 if dtype == 'float16' else types.FLOAT,
                                            output_layout=output_layout, crop=crop_shape, pad_output=pad_output,
                                            image_type=types.RGB, mean=args.rgb_mean, std=args.rgb_std)

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        if self.resize:
            images = self.resize(images)
        output = self.cmnp(images.gpu())
        return [output, self.labels]


def get_rec_iter(args, kv=None, dali_cpu=False):
    gpus = args.gpus
    num_threads = args.dali_threads
    num_validation_threads = args.dali_validation_threads
    pad_output = (args.image_shape[0] == 4)

    # the input_layout w.r.t. the model is the output_layout of the image pipeline
    output_layout = types.NHWC if args.input_layout == 'NHWC' else types.NCHW

    if 'horovod' in args.kv_store:
        rank = hvd.rank()
        nWrk = hvd.size()
    else:
        rank = kv.rank if kv else 0
        nWrk = kv.num_workers if kv else 1

    batch_size = args.batch_size // nWrk // len(gpus)

    trainpipes = [HybridTrainPipe(args           = args,
                                  batch_size     = batch_size,
                                  num_threads    = num_threads,
                                  device_id      = gpu_id,
                                  rec_path       = args.data_train,
                                  idx_path       = args.data_train_idx,
                                  shard_id       = gpus.index(gpu_id) + len(gpus)*rank,
                                  num_shards     = len(gpus)*nWrk,
                                  crop_shape     = args.image_shape[1:],
                                  output_layout  = output_layout,
                                  dtype          = args.dtype,
                                  pad_output     = pad_output,
                                  dali_cpu       = dali_cpu,
                                  nvjpeg_padding = args.dali_nvjpeg_memory_padding * 1024 * 1024,
                                  prefetch_queue = args.dali_prefetch_queue) for gpu_id in gpus]

    if args.data_val:
        valpipes = [HybridValPipe(args           = args,
                                  batch_size     = batch_size,
                                  num_threads    = num_validation_threads,
                                  device_id      = gpu_id,
                                  rec_path       = args.data_val,
                                  idx_path       = args.data_val_idx,
                                  shard_id       = 0 if args.dali_separ_val
                                                      else gpus.index(gpu_id) + len(gpus)*rank,
                                  num_shards     = 1 if args.dali_separ_val else len(gpus)*nWrk,
                                  crop_shape     = args.image_shape[1:],
                                  resize_shp     = args.data_val_resize,
                                  output_layout  = output_layout,
                                  dtype          = args.dtype,
                                  pad_output     = pad_output,
                                  dali_cpu       = dali_cpu,
                                  nvjpeg_padding = args.dali_nvjpeg_memory_padding * 1024 * 1024,
                                  prefetch_queue = args.dali_prefetch_queue) for gpu_id in gpus] if args.data_val else None
    trainpipes[0].build()
    if args.data_val:
        valpipes[0].build()
        worker_val_examples = valpipes[0].epoch_size("Reader")
        if not args.dali_separ_val:
            worker_val_examples = worker_val_examples // nWrk
            if rank < valpipes[0].epoch_size("Reader") % nWrk:
                worker_val_examples += 1

    if args.num_examples < trainpipes[0].epoch_size("Reader"):
        warnings.warn("{} training examples will be used, although full training set contains {} examples".format(args.num_examples, trainpipes[0].epoch_size("Reader")))
    dali_train_iter = DALIClassificationIterator(trainpipes, args.num_examples // nWrk)

    if args.data_val:
        dali_val_iter = DALIClassificationIterator(valpipes, worker_val_examples, fill_last_batch = False) if args.data_val else None
    else:
        dali_val_iter = None

    return dali_train_iter, dali_val_iter
