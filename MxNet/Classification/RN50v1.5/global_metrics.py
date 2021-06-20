import numpy as np


class CompositeMeter:
    def __init__(self):
        self.register = {}

    def register_metric(self, name, metric):
        self.register[name] = metric

    def _validate(self, metric_name):
        if metric_name not in self.register:
            raise ValueError('{} is not registered metric'.format(metric_name))

    def update_metric(self, metric_name, value):
        self._validate(metric_name)
        self.register[metric_name].update(value)

    def update_dict(self, dict_metric):
        for name, val in dict_metric.items():
            if name in self.register.keys():
                self.update_metric(name, val)

    def get(self, metric_name=None):
        if metric_name is not None:
            self._validate(metric_name)
            return self.register[metric_name].get()
        res_dict = {name: metric.get() for name, metric in self.register.items()}
        return res_dict


class MaxMeter:
    def __init__(self):
        self.max = None
        self.n = 0

    def reset(self):
        self.max = None
        self.n = 0

    def update(self, val):
        if self.max is None:
            self.max = val
        else:
            self.max = max(self.max, val)

    def get(self):
        return self.max


class MinMeter:
    def __init__(self):
        self.min = None
        self.n = 0

    def reset(self):
        self.min = None
        self.n = 0

    def update(self, val):
        if self.min is None:
            self.min = val
        else:
            self.min = min(self.min, val)

    def get(self):
        return self.min


class AvgMeter:
    def __init__(self):
        self.sum = 0
        self.n = 0

    def reset(self):
        self.sum = 0
        self.n = 0

    def update(self, val):
        self.sum += val
        self.n += 1

    def get(self):
        return self.sum / self.n


class PercentileMeter:
    def __init__(self, q):
        self.data = []
        self.q = q

    def reset(self):
        self.data = []

    def update(self, data):
        self.data.extend(data)

    def get(self):
        return np.percentile(self.data, self.q)
