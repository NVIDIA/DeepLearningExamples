import torch
from torch.utils.data.sampler import Sampler
from seq2seq.utils import get_world_size, get_rank


class BucketingSampler(Sampler):
    """
    Distributed data sampler supporting bucketing by sequence length.
    """
    def __init__(self, dataset, batch_size, bucketing=True, world_size=None,
                 rank=None):
        """
        Constructor for the BucketingSampler.

        :param dataset: dataset
        :param batch_size: batch size
        :param bucketing: if True enables bucketing by sequence length
        :param world_size: number of processes participating in distributed
            training
        :param rank: rank of the current process within world_size
        """
        if world_size is None:
            world_size = get_world_size()
        if rank is None:
            rank = get_rank()

        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0
        self.bucketing = bucketing

        self.batch_size = batch_size
        self.global_batch_size = batch_size * world_size

        self.data_len = len(self.dataset)
        self.num_samples = self.data_len // self.global_batch_size \
            * self.global_batch_size

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # generate permutation
        indices = torch.randperm(self.data_len, generator=g)
        # make indices evenly divisible by (batch_size * world_size)
        indices = indices[:self.num_samples]

        # splits the dataset into chunks of 'batches_in_shard' global batches
        # each, sorts by (src + tgt) sequence length within each chunk,
        # reshuffles all global batches
        if self.bucketing:
            batches_in_shard = 80
            shard_size = self.global_batch_size * batches_in_shard
            nshards = (self.num_samples + shard_size - 1) // shard_size

            lengths = self.dataset.lengths[indices]

            shards = [indices[i * shard_size:(i+1) * shard_size] for i in range(nshards)]
            len_shards = [lengths[i * shard_size:(i+1) * shard_size] for i in range(nshards)]

            indices = []
            for len_shard in len_shards:
                _, ind = len_shard.sort()
                indices.append(ind)

            output = tuple(shard[idx] for shard, idx in zip(shards, indices))
            indices = torch.cat(output)

            # global reshuffle
            indices = indices.view(-1, self.global_batch_size)
            order = torch.randperm(indices.shape[0], generator=g)
            indices = indices[order, :]
            indices = indices.view(-1)

        assert len(indices) == self.num_samples

        # build indices for each individual worker
        # consecutive ranks are getting consecutive batches,
        # default pytorch DistributedSampler assigns strided batches
        # with offset = length / world_size
        indices = indices.view(-1, self.batch_size)
        indices = indices[self.rank::self.world_size].contiguous()
        indices = indices.view(-1)
        indices = indices.tolist()

        assert len(indices) == self.num_samples // self.world_size

        return iter(indices)

    def __len__(self):
        return self.num_samples // self.world_size

    def set_epoch(self, epoch):
        """
        Sets current epoch index. This value is used to seed RNGs in __iter__()
        function.

        :param epoch: index of current epoch
        """
        self.epoch = epoch
