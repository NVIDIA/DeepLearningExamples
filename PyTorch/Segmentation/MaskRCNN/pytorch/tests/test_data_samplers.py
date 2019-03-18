# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import itertools
import random
import unittest

from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.sampler import RandomSampler

from maskrcnn_benchmark.data.samplers import GroupedBatchSampler
from maskrcnn_benchmark.data.samplers import IterationBasedBatchSampler


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class TestGroupedBatchSampler(unittest.TestCase):
    def test_respect_order_simple(self):
        drop_uneven = False
        dataset = [i for i in range(40)]
        group_ids = [i // 10 for i in dataset]
        sampler = SequentialSampler(dataset)
        for batch_size in [1, 3, 5, 6]:
            batch_sampler = GroupedBatchSampler(
                sampler, group_ids, batch_size, drop_uneven
            )
            result = list(batch_sampler)
            merged_result = list(itertools.chain.from_iterable(result))
            self.assertEqual(merged_result, dataset)

    def test_respect_order(self):
        drop_uneven = False
        dataset = [i for i in range(10)]
        group_ids = [0, 0, 1, 0, 1, 1, 0, 1, 1, 0]
        sampler = SequentialSampler(dataset)

        expected = [
            [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
            [[0, 1, 3], [2, 4, 5], [6, 9], [7, 8]],
            [[0, 1, 3, 6], [2, 4, 5, 7], [8], [9]],
        ]

        for idx, batch_size in enumerate([1, 3, 4]):
            batch_sampler = GroupedBatchSampler(
                sampler, group_ids, batch_size, drop_uneven
            )
            result = list(batch_sampler)
            self.assertEqual(result, expected[idx])

    def test_respect_order_drop_uneven(self):
        batch_size = 3
        drop_uneven = True
        dataset = [i for i in range(10)]
        group_ids = [0, 0, 1, 0, 1, 1, 0, 1, 1, 0]
        sampler = SequentialSampler(dataset)
        batch_sampler = GroupedBatchSampler(sampler, group_ids, batch_size, drop_uneven)

        result = list(batch_sampler)

        expected = [[0, 1, 3], [2, 4, 5]]
        self.assertEqual(result, expected)

    def test_subset_sampler(self):
        batch_size = 3
        drop_uneven = False
        dataset = [i for i in range(10)]
        group_ids = [0, 0, 1, 0, 1, 1, 0, 1, 1, 0]
        sampler = SubsetSampler([0, 3, 5, 6, 7, 8])

        batch_sampler = GroupedBatchSampler(sampler, group_ids, batch_size, drop_uneven)
        result = list(batch_sampler)

        expected = [[0, 3, 6], [5, 7, 8]]
        self.assertEqual(result, expected)

    def test_permute_subset_sampler(self):
        batch_size = 3
        drop_uneven = False
        dataset = [i for i in range(10)]
        group_ids = [0, 0, 1, 0, 1, 1, 0, 1, 1, 0]
        sampler = SubsetSampler([5, 0, 6, 1, 3, 8])

        batch_sampler = GroupedBatchSampler(sampler, group_ids, batch_size, drop_uneven)
        result = list(batch_sampler)

        expected = [[5, 8], [0, 6, 1], [3]]
        self.assertEqual(result, expected)

    def test_permute_subset_sampler_drop_uneven(self):
        batch_size = 3
        drop_uneven = True
        dataset = [i for i in range(10)]
        group_ids = [0, 0, 1, 0, 1, 1, 0, 1, 1, 0]
        sampler = SubsetSampler([5, 0, 6, 1, 3, 8])

        batch_sampler = GroupedBatchSampler(sampler, group_ids, batch_size, drop_uneven)
        result = list(batch_sampler)

        expected = [[0, 6, 1]]
        self.assertEqual(result, expected)

    def test_len(self):
        batch_size = 3
        drop_uneven = True
        dataset = [i for i in range(10)]
        group_ids = [random.randint(0, 1) for _ in dataset]
        sampler = RandomSampler(dataset)

        batch_sampler = GroupedBatchSampler(sampler, group_ids, batch_size, drop_uneven)
        result = list(batch_sampler)
        self.assertEqual(len(result), len(batch_sampler))
        self.assertEqual(len(result), len(batch_sampler))

        batch_sampler = GroupedBatchSampler(sampler, group_ids, batch_size, drop_uneven)
        batch_sampler_len = len(batch_sampler)
        result = list(batch_sampler)
        self.assertEqual(len(result), batch_sampler_len)
        self.assertEqual(len(result), len(batch_sampler))


class TestIterationBasedBatchSampler(unittest.TestCase):
    def test_number_of_iters_and_elements(self):
        for batch_size in [2, 3, 4]:
            for num_iterations in [4, 10, 20]:
                for drop_last in [False, True]:
                    dataset = [i for i in range(10)]
                    sampler = SequentialSampler(dataset)
                    batch_sampler = BatchSampler(
                        sampler, batch_size, drop_last=drop_last
                    )

                    iter_sampler = IterationBasedBatchSampler(
                        batch_sampler, num_iterations
                    )
                    assert len(iter_sampler) == num_iterations
                    for i, batch in enumerate(iter_sampler):
                        start = (i % len(batch_sampler)) * batch_size
                        end = min(start + batch_size, len(dataset))
                        expected = [x for x in range(start, end)]
                        self.assertEqual(batch, expected)


if __name__ == "__main__":
    unittest.main()
