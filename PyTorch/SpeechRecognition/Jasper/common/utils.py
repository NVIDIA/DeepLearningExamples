import numpy as np


class BenchmarkStats:
    """ Tracks statistics used for benchmarking. """
    def __init__(self):
        self.utts = []
        self.times = []
        self.losses = []

    def update(self, utts, times, losses):
        self.utts.append(utts)
        self.times.append(times)
        self.losses.append(losses)

    def get(self, n_epochs):
        throughput = sum(self.utts[-n_epochs:]) / sum(self.times[-n_epochs:])

        return {'throughput': throughput, 'benchmark_epochs_num': n_epochs,
                'loss': np.mean(self.losses[-n_epochs:])}
