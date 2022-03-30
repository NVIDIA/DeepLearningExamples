import timer
from collections import defaultdict


class Metrics(defaultdict):

    # TODO Where to measure - gpu:0 or all gpus?

    def __init__(self, tb_keys=[], benchmark_epochs=10):
        super().__init__(float)

        # dll_tb_keys=['loss_gen', 'loss_discrim', 'loss_mel', 'took']:

        self.tb_keys = tb_keys  #_ = {'dll': dll_keys, 'tb': tb_keys, 'dll+tb': dll_tb_keys}
        self.iter_start_time = None
        self.iter_metrics = defaultdict(float)
        self.epoch_start_time = None
        self.epoch_metrics = defaultdict(float)
        self.benchmark_epochs = benchmark_epochs

    def start_epoch(self, epoch, start_timer=True):
        self.epoch = epoch
        if start_timer:
            self.epoch_start_time = time.time()

    def start_iter(self, iter, start_timer=True):
        self.iter = iter
        self.accum_steps = 0
        self.step_metrics.clear()
        if start_timer:
            self.iter_start_time = time.time()

    def update_iter(self, ...):
        # do stuff
        pass

    def accumulate(self, scope='step'):
        tgt = {'step': self.step_metrics, 'epoch': self.epoch_metrics}[scope]

        for k, v in self.items():
            tgt[k] += v

        self.clear()

    def update_iter(self, metrics={}, stop_timer=True):

        is not self.started_iter:
            return

        self.accumulate(metrics)
        self.accumulate(self.iter_metrics, scope='epoch')

        if stop_timer:
            self.iter_metrics['took'] = time.time() - self.iter_time_start

    def update_epoch(self, stop_timer=True):

        #            tb_total_steps=None,
        #            subset='train_avg',
        #            data=OrderedDict([
        #                ('loss', epoch_loss[-1]),
        #                ('mel_loss', epoch_mel_loss[-1]),
        #                ('frames/s', epoch_num_frames[-1] / epoch_time[-1]),
        #                ('took', epoch_time[-1])]),
        #            )

        if stop_timer:
            self.['epoch_time'] = time.time() - self.epoch_time_start


        if steps % args.stdout_interval == 0:
            # with torch.no_grad():
            #     mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

            took = time.time() - self.start_b


        self.sws['train'].add_scalar("gen_loss_total", loss_gen_all.item(), steps)
        self.sws['train'].add_scalar("mel_spec_error", mel_error.item(), steps)

        for key, val in meta.items():

            sw_name = 'train'
            for name_ in keys_mpd + keys_msd:
                if name_ in key:
                    sw_name = 'train_' + name_

            key = key.replace('loss_', 'loss/')
            key = re.sub('mpd\d+', 'mpd-msd', key)
            key = re.sub('msd\d+', 'mpd-msd', key)

            self.sws[sw_name].add_scalar(key, val / h.batch_size, steps)

    def iter_metrics(self, target='dll+tb'):
        return {self.iter_metrics[k] for k in self.keys_[target]}

    def foo

Steps : 40, Gen Loss Total : 57.993, Mel-Spec. Error : 47.374, s/b : 1.013

                logger.log((epoch, epoch_iter, num_iters),
                           tb_total_steps=total_iter,
                           subset='train',
                           data=OrderedDict([
                               ('loss', iter_loss),
                               ('mel_loss', iter_mel_loss),
                               ('frames/s', iter_num_frames / iter_time),      
                               ('took', iter_time),
                               ('lrate', optimizer.param_groups[0]['lr'])]),   
                           )



class Meter:
    def __init__(self, sink_type, scope, downstream=None, end_points=None, verbosity=dllogger.Verbosity.DEFAULT):
        self.verbosity = verbosity
        self.sink_type = sink_type
        self.scope = scope
        self.downstream = downstream

        self.end_points = end_points or []

    def start(self):
        ds = None if self.downstream is None else self.downstream.sink
        end_pt_fn = lambda x: list(map(lambda f: f(x), self.end_points))  # call all endpoint functions
        self.sink = self.sink_type(end_pt_fn, ds)

    def end(self):
        self.sink.close()

    def send(self, data):
        self.sink.send(data)

    def meters(self):
        if self.downstream is not None:
            downstream_meters = self.downstream.meters()
        else:
            downstream_meters = []
        return [self] + downstream_meters

    def add_end_point(self, new_endpoint):
        self.end_points.append(new_endpoint)

    def __or__(self, other):
        """for easy chaining of meters"""
        if self.downstream is None:
            self.downstream = other
        else:
            self.downstream | other

        return self
