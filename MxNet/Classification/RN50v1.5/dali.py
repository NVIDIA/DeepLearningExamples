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


def add_dali_args(parser):
    group = parser.add_argument_group('DALI', 'pipeline and augumentation')
    group.add_argument('--use-dali', action='store_true',
                      help='use dalli pipeline and augunetation')
    group.add_argument('--separ-val', action='store_true',
                      help='each process will perform independent validation on whole val-set')
    group.add_argument('--dali-threads', type=int, default=3, help="number of threads" +\
                       "per GPU for DALI")
    group.add_argument('--validation-dali-threads', type=int, default=10, help="number of threads" +\
                       "per GPU for DALI for validation")
    group.add_argument('--dali-prefetch-queue', type=int, default=3, help="DALI prefetch queue depth")
    group.add_argument('--dali-nvjpeg-memory-padding', type=int, default=16, help="Memory padding value for nvJPEG (in MB)")
    return parser


_mean_pixel = [255 * x for x in (0.485, 0.456, 0.406)]
_std_pixel  = [255 * x for x in (0.229, 0.224, 0.225)]

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, rec_path, idx_path,
                 shard_id, num_shards, crop_shape,
                 nvjpeg_padding, prefetch_queue=3,
                 output_layout=types.NCHW, pad_output=True, dtype='float16'):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id, prefetch_queue_depth = prefetch_queue)
        self.input = ops.MXNetReader(path = [rec_path], index_path=[idx_path],
                                     random_shuffle=True, shard_id=shard_id, num_shards=num_shards)

        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB,
                                        device_memory_padding = nvjpeg_padding,
                                        host_memory_padding = nvjpeg_padding)
        self.rrc = ops.RandomResizedCrop(device = "gpu", size = crop_shape)
        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT16 if dtype == 'float16' else types.FLOAT,
                                            output_layout = output_layout,
                                            crop = crop_shape,
                                            pad_output = pad_output,
                                            image_type = types.RGB,
                                            mean = _mean_pixel,
                                            std =  _std_pixel)
        self.coin = ops.CoinFlip(probability = 0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name = "Reader")

        images = self.decode(self.jpegs)
        images = self.rrc(images)
        output = self.cmnp(images, mirror = rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, rec_path, idx_path,
                 shard_id, num_shards, crop_shape,
                 nvjpeg_padding, prefetch_queue=3,
                 resize_shp=None,
                 output_layout=types.NCHW, pad_output=True, dtype='float16'):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id, prefetch_queue_depth = prefetch_queue)
        self.input = ops.MXNetReader(path = [rec_path], index_path=[idx_path],
                                     random_shuffle=False, shard_id=shard_id, num_shards=num_shards)
        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB,
                                        device_memory_padding = nvjpeg_padding,
                                        host_memory_padding = nvjpeg_padding)
        self.resize = ops.Resize(device = "gpu", resize_shorter=resize_shp) if resize_shp else None
        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT16 if dtype == 'float16' else types.FLOAT,
                                            output_layout = output_layout,
                                            crop = crop_shape,
                                            pad_output = pad_output,
                                            image_type = types.RGB,
                                            mean = _mean_pixel,
                                            std =  _std_pixel)

    def define_graph(self):
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        if self.resize:
            images = self.resize(images)
        output = self.cmnp(images)
        return [output, self.labels]


def get_rec_iter(args, kv=None):
    # resize is default base length of shorter edge for dataset;
    # all images will be reshaped to this size
    resize = int(args.resize)
    # target shape is final shape of images pipelined to network;
    # all images will be cropped to this size
    target_shape = tuple([int(l) for l in args.image_shape.split(',')])
    pad_output = target_shape[0] == 4
    gpus = list(map(int, filter(None, args.gpus.split(',')))) # filter to not encount eventually empty strings
    batch_size = args.batch_size//len(gpus)
    num_threads = args.dali_threads
    num_validation_threads = args.validation_dali_threads
    #db_folder = "/data/imagenet/train-480-val-256-recordio/"

    # the input_layout w.r.t. the model is the output_layout of the image pipeline
    output_layout = types.NHWC if args.input_layout == 'NHWC' else types.NCHW

    rank = kv.rank if kv else 0
    nWrk = kv.num_workers if kv else 1

    trainpipes = [HybridTrainPipe(batch_size     = batch_size,
                                  num_threads    = num_threads,
                                  device_id      = gpu_id,
                                  rec_path       = args.data_train,
                                  idx_path       = args.data_train_idx,
                                  shard_id       = gpus.index(gpu_id) + len(gpus)*rank,
                                  num_shards     = len(gpus)*nWrk,
                                  crop_shape     = target_shape[1:],
                                  output_layout  = output_layout,
                                  pad_output     = pad_output,
                                  dtype          = args.dtype,
                                  nvjpeg_padding = args.dali_nvjpeg_memory_padding * 1024 * 1024,
                                  prefetch_queue = args.dali_prefetch_queue) for gpu_id in gpus]

    valpipes = [HybridValPipe(batch_size     = batch_size,
                              num_threads    = num_validation_threads,
                              device_id      = gpu_id,
                              rec_path       = args.data_val,
                              idx_path       = args.data_val_idx,
                              shard_id       = 0 if args.separ_val
                                                 else gpus.index(gpu_id) + len(gpus)*rank,
                              num_shards     = 1 if args.separ_val else len(gpus)*nWrk,
                              crop_shape     = target_shape[1:],
                              resize_shp     = resize,
                              output_layout  = output_layout,
                              pad_output     = pad_output,
                              dtype          = args.dtype,
                              nvjpeg_padding = args.dali_nvjpeg_memory_padding * 1024 * 1024,
                              prefetch_queue = args.dali_prefetch_queue) for gpu_id in gpus] if args.data_val else None
    trainpipes[0].build()
    if args.data_val:
        valpipes[0].build()

    if args.num_examples < trainpipes[0].epoch_size("Reader"):
        warnings.warn("{} training examples will be used, although full training set contains {} examples".format(args.num_examples, trainpipes[0].epoch_size("Reader")))
    dali_train_iter = DALIClassificationIterator(trainpipes, args.num_examples // nWrk)
    dali_val_iter = DALIClassificationIterator(valpipes, valpipes[0].epoch_size("Reader") // (1 if args.separ_val else nWrk), fill_last_batch = False) if args.data_val else None
    return dali_train_iter, dali_val_iter

