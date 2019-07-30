import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

DATA_BACKEND_CHOICES = ['pytorch']
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    DATA_BACKEND_CHOICES.append('dali-gpu')
    DATA_BACKEND_CHOICES.append('dali-cpu')
except ImportError:
    print("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id)
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1

        self.input = ops.FileReader(
                file_root = data_dir,
                shard_id = local_rank,
                num_shards = world_size,
                random_shuffle = True)

        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoderRandomCrop(device=dali_device, output_type=types.RGB,
                                                    random_aspect_ratio=[0.75, 4./3.],
                                                    random_area=[0.08, 1.0],
                                                    num_attempts=100)
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.nvJPEGDecoderRandomCrop(device="mixed", output_type=types.RGB, device_memory_padding=211025920, host_memory_padding=140544512,
                                                      random_aspect_ratio=[0.75, 4./3.],
                                                      random_area=[0.08, 1.0],
                                                      num_attempts=100)

        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT,
                                            output_layout = types.NCHW,
                                            crop = (crop, crop),
                                            image_type = types.RGB,
                                            mean = [0.485 * 255,0.456 * 255,0.406 * 255],
                                            std = [0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability = 0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror = rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id)
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1

        self.input = ops.FileReader(
                file_root = data_dir,
                shard_id = local_rank,
                num_shards = world_size,
                random_shuffle = False)

        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
        self.res = ops.Resize(device = "gpu", resize_shorter = size)
        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                output_dtype = types.FLOAT,
                output_layout = types.NCHW,
                crop = (crop, crop),
                image_type = types.RGB,
                mean = [0.485 * 255,0.456 * 255,0.406 * 255],
                std = [0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


class DALIWrapper(object):
    def gen_wrapper(dalipipeline, num_classes, one_hot):
        for data in dalipipeline:
            input = data[0]["data"]
            target = data[0]["label"].squeeze().cuda().long()
            if one_hot:
                target = expand(num_classes, torch.float, target)
            yield input, target
        dalipipeline.reset()

    def __init__(self, dalipipeline, num_classes, one_hot):
        self.dalipipeline = dalipipeline
        self.num_classes =  num_classes
        self.one_hot = one_hot

    def __iter__(self):
        return DALIWrapper.gen_wrapper(self.dalipipeline, self.num_classes, self.one_hot)

def get_dali_train_loader(dali_cpu=False):
    def gdtl(data_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False):
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1

        traindir = os.path.join(data_path, 'train')

        pipe = HybridTrainPipe(batch_size=batch_size, num_threads=workers,
                device_id = local_rank,
                data_dir = traindir, crop = 224, dali_cpu=dali_cpu)

        pipe.build()
        train_loader = DALIClassificationIterator(pipe, size = int(pipe.epoch_size("Reader") / world_size))

        return DALIWrapper(train_loader, num_classes, one_hot), int(pipe.epoch_size("Reader") / (world_size * batch_size))

    return gdtl


def get_dali_val_loader():
    def gdvl(data_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False):
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            local_rank = 0
            world_size = 1

        valdir = os.path.join(data_path, 'val')

        pipe = HybridValPipe(batch_size=batch_size, num_threads=workers,
                device_id = local_rank,
                data_dir = valdir,
                crop = 224, size = 256)
        pipe.build()
        val_loader = DALIClassificationIterator(pipe, size = int(pipe.epoch_size("Reader") / world_size))

        return DALIWrapper(val_loader, num_classes, one_hot), int(pipe.epoch_size("Reader") / (world_size * batch_size))
    return gdvl


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


def expand(num_classes, dtype, tensor):
    e = torch.zeros(tensor.size(0), num_classes, dtype=dtype, device=torch.device('cuda'))
    e = e.scatter(1, tensor.unsqueeze(1), 1.0)
    return e

class PrefetchedWrapper(object):
    def prefetched_loader(loader, num_classes, fp16, one_hot):
        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        if fp16:
            mean = mean.half()
            std = std.half()

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(async=True)
                next_target = next_target.cuda(async=True)
                if fp16:
                    next_input = next_input.half()
                    if one_hot:
                        next_target = expand(num_classes, torch.half, next_target)
                else:
                    next_input = next_input.float()
                    if one_hot:
                        next_target = expand(num_classes, torch.float, next_target)

                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader, num_classes, fp16, one_hot):
        self.dataloader = dataloader
        self.fp16 = fp16
        self.epoch = 0
        self.one_hot = one_hot
        self.num_classes = num_classes

    def __iter__(self):
        if (self.dataloader.sampler is not None and
            isinstance(self.dataloader.sampler,
                       torch.utils.data.distributed.DistributedSampler)):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader, self.num_classes, self.fp16, self.one_hot)

def get_pytorch_train_loader(data_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False):
    traindir = os.path.join(data_path, 'train')
    train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                ]))

    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate, drop_last=True)

    return PrefetchedWrapper(train_loader, num_classes, fp16, one_hot), len(train_loader)

def get_pytorch_val_loader(data_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False):
    valdir = os.path.join(data_path, 'val')
    val_dataset = datasets.ImageFolder(
            valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                ]))

    if torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True,
            collate_fn=fast_collate)

    return PrefetchedWrapper(val_loader, num_classes, fp16, one_hot), len(val_loader)
