import random
import json
from collections import OrderedDict


class IterationMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.last = 0

    def record(self, val, n = 1):
        self.last = val

    def get_val(self):
        return None

    def get_last(self):
        return self.last


class EpochMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0

    def record(self, val, n = 1):
        self.val = val

    def get_val(self):
        return self.val

    def get_last(self):
        return None


class AverageMeter(object):
    def __init__(self, ret_last=True, ret_val=True):
        self.reset()
        self.ret_last = ret_last
        self.ret_val = ret_val

    def reset(self):
        self.n = 0
        self.val = 0
        self.last = 0

    def record(self, val, n = 1):
        self.last = val
        self.n += n
        self.val += val * n

    def get_val(self):
        if self.ret_val:
            if self.n == 0:
                return 0.0
            return self.val / self.n
        else:
            return None

    def get_last(self):
        if self.ret_last:
            return self.last
        else:
            return None


class RunningMeter(object):
    def __init__(self, decay):
        self.decay = decay

    def reset(self):
        self.val = 0
        self.last = 0

    def record(self, val, n = 1):
        self.last = val
        decay = 1 - ((1 - self.decay) ** n)
        self.val = (1 - decay) * self.val + decay * val

    def get_val(self):
        return self.val

    def get_last(self):
        return self.last


class Logger(object):
    def __init__(self, print_interval, backends, verbose=False):
        self.epoch = -1
        self.iteration = -1
        self.val_iteration = -1
        self.metrics = OrderedDict()
        self.backends = backends
        self.print_interval = print_interval
        self.verbose = verbose

    def log_run_tag(self, name, val):
        for b in self.backends:
            b.log_run_tag(name, val)

    def register_metric(self, metric_name, meter, log_level=0):
        if self.verbose:
            print("Registering metric: {}".format(metric_name))
        self.metrics[metric_name] = {'meter' : meter, 'level' : log_level}

    def log_metric(self, metric_name, val, n=1):
        self.metrics[metric_name]['meter'].record(val, n=n)

    def start_iteration(self, val=False):
        if val:
            self.val_iteration += 1
        else:
            self.iteration += 1

    def end_iteration(self, val=False):
        it = self.val_iteration if val else self.iteration
        if (it % self.print_interval == 0):
            for b in self.backends:
                if val:
                    b.log_iteration_metric('val.it', it)
                else:
                    b.log_iteration_metric('it', it)

                f = lambda l: filter(lambda m : m['level'] <= b.level)
                for n, m in [(n, m) for n, m in self.metrics.items() if m['level'] <= b.level and n.startswith('val') == val]:
                    mv = m['meter'].get_last()
                    if mv is not None:
                        b.log_iteration_metric(n, mv)

                b.log_end_iteration()

    def start_epoch(self):
        self.epoch += 1
        self.iteration = 0
        self.val_iteration = 0

        for b in self.backends:
            b.log_epoch_metric('ep', self.epoch)

        for n, m in [(n, m) for n, m in self.metrics.items() if m['level'] <= b.level]:
            m['meter'].reset()

    def end_epoch(self):
        for b in self.backends:
            for n, m in [(n, m) for n, m in self.metrics.items() if m['level'] <= b.level]:
                mv = m['meter'].get_val()
                if mv is not None:
                    b.log_epoch_metric(n, mv)
            b.log_end_epoch()

    def end(self):
        for b in self.backends:
            b.end()

    def iteration_generator_wrapper(self, gen, val = False):
        for g in gen:
            self.start_iteration(val = val)
            yield g
            self.end_iteration(val = val)

    def epoch_generator_wrapper(self, gen):
        for g in gen:
            self.start_epoch()
            yield g
            self.end_epoch()


class JsonBackend(object):
    def __init__(self, filename, log_level=0):
        self.level = log_level
        self.filename = filename
        self.json_log = OrderedDict([
                ('run'  , OrderedDict()),
                ('epoch', OrderedDict()),
                ('iter' , OrderedDict()),
                ('event', OrderedDict()),
                ])

    def log_run_tag(self, name, val):
        self.json_log['run'][name] = val

    def log_end_epoch(self):
        pass

    def log_end_iteration(self):
        pass

    def log_epoch_metric(self, name, val):
        if not name in self.json_log['epoch'].keys():
            self.json_log['epoch'][name] = []

        self.json_log['epoch'][name].append(val)

        if name != 'ep':
            if name in self.json_log['iter'].keys():
                self.json_log['iter'][name].append([])
        else:
            if not 'it' in self.json_log['iter'].keys():
                self.json_log['iter']['it'] = []

            self.json_log['iter']['it'].append([])

    def log_iteration_metric(self, name, val):
        if not (name in self.json_log['iter'].keys()):
            self.json_log['iter'][name] = [[]]

        self.json_log['iter'][name][-1].append(val)

    def end(self):
        print(json.dump(self.json_log, open(self.filename, 'w')))


class StdOut1LBackend(object):
    def __init__(self, iters, val_iters, epochs, log_level=0):
        self.level = log_level
        self.iteration = 0
        self.total_iterations = iters
        self.total_val_iterations = val_iters
        self.epoch = 0
        self.total_epochs = epochs
        self.iteration_metrics = {}
        self.epoch_metrics = {}
        self.mode = 'train'

    def log_run_tag(self, name, val):
        print("{} : {}".format(name, val))

    def log_end_epoch(self):
        print("Summary Epoch: {}/{};\t{}".format(
            self.epoch, self.total_epochs,
            "\t".join(["{} : {:.3f}".format(m,v) for m, v in self.epoch_metrics.items()])))

        self.epoch_metrics = {}

    def log_end_iteration(self):
        md = "Validation" if self.mode == 'val' else ""
        ti = self.total_val_iterations if self.mode == 'val' else self.total_iterations
        print("Epoch: {}/{} {} Iteration: {}/{};\t{}".format(
            self.epoch, self.total_epochs, md, self.iteration, ti,
            "\t".join(["{} : {:.3f}".format(m,v) for m, v in self.iteration_metrics.items()])))

        self.iteration_metrics = {}

    def log_epoch_metric(self, name, value):
        if name == 'ep':
            self.epoch = value
            self.iteration = 0
        else:
            self.epoch_metrics[name] = value

    def log_iteration_metric(self, name, value):
        if name == 'it' or name == 'val.it':
            self.mode = 'train' if name == 'it' else 'val'
            self.iteration = value
        else:
            self.iteration_metrics[name] = value

    def end(self):
        pass



class StdOutBackend(object):
    def __init__(self, iters, epochs, log_level=0):
        self.level = log_level
        self.iteration = 0
        self.epoch = 0

    def log_run_tag(self, name, val):
        print("{} : {}".format(name, val))

    def log_end_epoch(self):
        pass

    def log_end_iteration(self):
        pass

    def log_epoch_metric(self, name, value):
        if name == 'ep':
            self.epoch = value
            self.iteration = 0
        else:
            print("Summary Epoch: {};  {} = {:.3f}".format(self.epoch, name, value))

    def log_iteration_metric(self, name, value):
        if name == 'it' or name == 'val.it':
            self.iteration = value
        else:
            print("Epoch: {} Iteration: {};  {} = {:.3f}".format(self.epoch, self.iteration, name, value))

    def end(self):
        pass


