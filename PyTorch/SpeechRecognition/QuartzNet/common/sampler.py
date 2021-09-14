import torch
import numpy as np

from torch.utils.data.sampler import Sampler


class DistributedSampler(Sampler):
    def __init__(self, dataset, batch_size, world_size, rank):
        """
        Constructor for the DistributedSampler.
        :param dataset: dataset
        :param batch_size: local batch size
        :param world_size: number of distributed workers
        :param rank: rank of the current process
        """
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0

        self.batch_size = batch_size
        self.global_batch_size = batch_size * world_size

        self.data_len = len(self.dataset)

        self.num_samples = self.data_len // self.global_batch_size \
            * self.global_batch_size

    def distribute_batches(self, indices):
        """
        Assigns batches to workers.
        Consecutive ranks are getting consecutive batches.
        :param indices: torch.tensor with batch indices
        """
        assert len(indices) == self.num_samples

        indices = indices.view(-1, self.batch_size)
        indices = indices[self.rank::self.world_size].contiguous()
        indices = indices.view(-1)
        indices = indices.tolist()

        assert len(indices) == self.num_samples // self.world_size
        return indices

    def reshuffle_batches(self, indices, rng):
        """
        Permutes global batches
        :param indices: torch.tensor with batch indices
        :param rng: instance of torch.Generator
        """
        indices = indices.view(-1, self.global_batch_size)
        num_batches = indices.shape[0]
        order = torch.randperm(num_batches, generator=rng)
        indices = indices[order, :]
        indices = indices.view(-1)
        return indices

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        # generate permutation
        indices = torch.randperm(self.data_len, generator=rng)

        # make indices evenly divisible by (batch_size * world_size)
        indices = indices[:self.num_samples]

        # assign batches to workers
        indices = self.distribute_batches(indices)
        return iter(indices)

    def set_epoch(self, epoch):
        """
        Sets current epoch index.
        Epoch index is used to seed RNG in __iter__() function.
        :param epoch: index of current epoch
        """
        self.epoch = epoch

    def __len__(self):
        return self.num_samples // self.world_size


class BucketingSampler(DistributedSampler):
    def __init__(self, dataset, batch_size, num_buckets, world_size, rank):
        """
        Bucketing sampler with approx. equally-sized buckets.
        :param dataset: dataset
        :param batch_size: local batch size
        :param seeds: list of seeds, one seed for each training epoch
        :param num_buckets: number of buckets
        :param world_size: number of distributed workers
        :param rank: rank of the current process
        """
        super().__init__(dataset, batch_size, world_size, rank)

        self.num_buckets = num_buckets
        len_ids = np.argsort([sample['duration'] for sample in dataset.samples])
        self.buckets = [torch.from_numpy(t)
                        for t in np.array_split(len_ids, num_buckets)]
        global_bs = self.global_batch_size

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        global_bsz = self.global_batch_size

        indices = []
        for bid in range(self.num_buckets):
            # random shuffle within current bucket
            perm = torch.randperm(len(self.buckets[bid]), generator=g)
            bucket_indices = self.buckets[bid][perm]

            # add samples from current bucket to indices for current epoch
            indices.append(bucket_indices)

        indices = torch.cat(indices)

        # make indices evenly divisible by global batch size
        length = len(indices) // global_bsz * global_bsz
        indices = indices[:length]

        assert len(indices) % self.global_batch_size == 0

        # perform global reshuffle of all global batches
        indices = self.reshuffle_batches(indices, g)
        # distribute batches to individual workers
        indices = self.distribute_batches(indices)
        return iter(indices)
