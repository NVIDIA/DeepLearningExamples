# Copyright 2017-2018 The Apache Software Foundation
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -----------------------------------------------------------------------
#
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

""" train fit utility """
import logging
import math
import glob
import os
import random
import sys
import time
import re
from itertools import starmap

import signal
import pickle

import dllogger
import horovod.mxnet as hvd
import mxnet as mx
import mxnet.contrib.amp as amp
import numpy as np
from mxnet import autograd as ag
from mxnet import gluon

import data
from benchmarking import BenchmarkingDataIter
from global_metrics import CompositeMeter, MaxMeter, MinMeter, AvgMeter, PercentileMeter


class PartitionSignalHandler():
    def __init__(self, sync_freq: int = 10):
        self.step = 0
        self.freq = sync_freq

        self.t = mx.nd.array([0])

        signal.signal(signal.SIGUSR1, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def sync(self) -> bool:
        if self.step % self.freq == 0:
            new_sync = hvd.allreduce(self.t, average=False)
            if new_sync[0] > 0:
                self.t[0] = 1
        self.step += 1

        return self.should_end()

    def should_end(self) -> bool:
        return bool(self.t[0] > 0)

    def _signal_handler(self, signum, frame):
        print("Signal received")
        self.t[0] = 1


def add_fit_args(parser):
    def int_list(x):
        return list(map(int, x.split(',')))

    def float_list(x):
        return list(map(float, x.split(',')))

    train = parser.add_argument_group('Training')
    train.add_argument('--mode', default='train_val', choices=('train_val', 'train', 'val', 'pred'),
                       help='mode')
    train.add_argument('--seed', type=int, default=None,
                       help='random seed')
    train.add_argument('--gpus', type=int_list, default=[0],
                       help='list of gpus to run, e.g. 0 or 0,2,5')
    train.add_argument('--kv-store', type=str, default='device', choices=('device', 'horovod'),
                       help='key-value store type')
    train.add_argument('--dtype', type=str, default='float16', choices=('float32', 'float16'),
                       help='precision')
    train.add_argument('--amp', action='store_true',
                       help='If enabled, turn on AMP (Automatic Mixed Precision)')
    train.add_argument('--batch-size', type=int, default=192,
                       help='the batch size')
    train.add_argument('--num-epochs', type=int, default=90,
                       help='number of epochs')
    train.add_argument('--run-epochs', type=int, default=-1,
                       help='number of epochs to run in single run')
    train.add_argument('--lr', type=float, default=0.1,
                       help='initial learning rate')
    train.add_argument('--lr-schedule', choices=('multistep', 'cosine'), default='cosine',
                       help='learning rate schedule')
    train.add_argument('--lr-factor', type=float, default=0.256,
                       help='the ratio to reduce lr on each step')
    train.add_argument('--lr-steps', type=float_list, default=[],
                       help='the epochs to reduce the lr, e.g. 30,60')
    train.add_argument('--warmup-epochs', type=int, default=5,
                       help='the epochs to ramp-up lr to scaled large-batch value')
    train.add_argument('--optimizer', type=str, default='sgd',
                       help='the optimizer type')
    train.add_argument('--mom', type=float, default=0.875,
                       help='momentum for sgd')
    train.add_argument('--wd', type=float, default=1 / 32768,
                       help='weight decay for sgd')
    train.add_argument('--label-smoothing', type=float, default=0.1,
                       help='label smoothing factor')
    train.add_argument('--mixup', type=float, default=0,
                       help='alpha parameter for mixup (if 0 then mixup is not applied)')
    train.add_argument('--disp-batches', type=int, default=20,
                       help='show progress for every n batches')
    train.add_argument('--model-prefix', type=str, default='model',
                       help='model checkpoint prefix')
    train.add_argument('--save-frequency', type=int, default=-1,
                       help='frequency of saving model in epochs (--model-prefix must be specified). '
                            'If -1 then save only best model. If 0 then do not save anything.')
    train.add_argument('--begin-epoch', type=int, default=0,
                       help='start the model from an epoch')
    train.add_argument('--load', help='checkpoint to load')
    train.add_argument('--test-io', action='store_true',
                       help='test reading speed without training')
    train.add_argument('--test-io-mode', default='train', choices=('train', 'val'),
                       help='data to test')
    train.add_argument('--log', type=str, default='log.log',
                       help='file where to save the log from the experiment')
    train.add_argument('--dllogger-log', type=str, default='dllogger_log.log',
                       help='file where to save the dllogger log from the experiment')
    train.add_argument('--workspace', type=str, default='./',
                       help='path to directory where results will be stored')
    train.add_argument('--logdir', type=str, default=None,
                       help="path to directory where logs will be stored")
    train.add_argument('--no-metrics', action='store_true',
                       help='do not calculate evaluation metrics (for benchmarking)')
    train.add_argument('--benchmark-iters', type=int, default=None,
                       help='run only benchmark-iters iterations from each epoch')
    return train


def get_epoch_size(args, kv):
    return math.ceil(args.num_examples / args.batch_size)


def get_lr_scheduler(args):
    def multistep_schedule(x):
        lr = args.lr * \
            (args.lr_factor ** (len(list(filter(lambda step: step <= x, args.lr_steps)))))
        warmup_coeff = min(1, x / args.warmup_epochs)
        return warmup_coeff * lr

    def cosine_schedule(x):
        steps = args.lr_steps
        if not steps or steps[0] > args.warmup_epochs:
            steps = [args.warmup_epochs] + steps
        elif not steps or steps[0] != 0:
            steps = [0] + steps

        if steps[-1] != args.num_epochs:
            steps.append(args.num_epochs)

        if x < args.warmup_epochs:
            return args.lr * x / args.warmup_epochs

        for i, (step, next_step) in enumerate(zip(steps, steps[1:])):
            if next_step > x:
                return args.lr * 0.5 * (1 + math.cos(math.pi * (x - step) / (next_step - step))) * (args.lr_factor ** i)
        return 0

    schedules = {
        'multistep': multistep_schedule,
        'cosine': cosine_schedule,
    }
    return schedules[args.lr_schedule]


def load_model(args, model):
    file = list(glob.glob(
        f"{args.workspace}/{args.model_prefix}_*.params"))
    if len(file) == 0:
        return -1

    file = [x for x in sorted(file) if "best.params" not in x]

    if len(file) == 0:
        return -1

    file = file[-1]

    epoch = re.match(f".*{args.model_prefix}_([0-9]*)\.params", file)
    if epoch is None:
        return -1

    epoch = int(epoch.group(1))
    model.load_parameters(file)
    logging.info('Loaded model {}'.format(file))
    return epoch


def save_checkpoint(net, epoch, top1, best_acc, model_prefix, workspace, save_frequency, kvstore, force_save=False):
    if model_prefix is None or save_frequency == 0 or ('horovod' in kvstore and hvd.rank() != 0):
        return
    if (save_frequency > 0 and (epoch + 1) % save_frequency == 0) or force_save:
        fname = '{}_{:04}.params'.format(model_prefix, epoch)
        fname = os.path.join(workspace, fname)
        net.save_parameters(fname)
        logging.info('[Epoch {}] Saving checkpoint to {} with Accuracy: {:.4f}'.format(
            epoch, fname, top1))

    if top1 > best_acc:
        fname = os.path.join(workspace, f'{model_prefix}_best.params')
        net.save_parameters(fname)
        logging.info('[Epoch {}] Saving checkpoint to {} with Accuracy: {:.4f}'.format(
            epoch, fname, top1))


def model_pred(args, model, image):
    from imagenet_classes import classes
    output = model(image.reshape(-1, *image.shape)
                   )[0].softmax().as_in_context(mx.cpu())
    top = output.argsort(is_ascend=False)[:10]
    for i, ind in enumerate(top):
        ind = int(ind.asscalar())
        logging.info('{:2d}. {:5.2f}% -> {}'.format(i + 1,
                     output[ind].asscalar() * 100, classes[ind]))


def reduce_metrics(args, metrics, kvstore):
    if 'horovod' not in kvstore or not metrics[0] or hvd.size() == 1:
        return metrics

    m = mx.ndarray.array(metrics[1], ctx=mx.gpu(args.gpus[0]))
    reduced = hvd.allreduce(m)
    values = reduced.as_in_context(mx.cpu()).asnumpy().tolist()
    return (metrics[0], values)


def model_score(args, net, val_data, metric, kvstore):
    if val_data is None:
        logging.info('Omitting validation: no data')
        return [], []

    if not isinstance(metric, mx.metric.EvalMetric):
        metric = mx.metric.create(metric)
    metric.reset()

    val_data.reset()

    total_batch_size = val_data.batch_size * val_data._num_gpus * \
        (hvd.size() if 'horovod' in kvstore else 1)

    durations = []
    tic = time.time()
    outputs = []
    for batches in val_data:
        # synchronize to previous iteration
        for o in outputs:
            o.wait_to_read()

        data = [b.data[0] for b in batches]
        label = [b.label[0][:len(b.data[0]) - b.pad]
                 for b in batches if len(b.data[0]) != b.pad]
        outputs = [net(X) for X, b in zip(data, batches)]
        outputs = [o[:len(b.data[0]) - b.pad]
                   for o, b in zip(outputs, batches) if len(b.data[0]) != b.pad]
        metric.update(label, outputs)

        durations.append(time.time() - tic)
        tic = time.time()

    metric = reduce_metrics(args, metric.get_global(), kvstore)
    durations = durations[min(len(durations) // 10, 100):]
    duration_stats = {
        'ips': total_batch_size / np.mean(durations),
        'latency_avg': np.mean(durations),
    }

    return metric, duration_stats, durations


class ScalarMetric(mx.metric.Loss):
    def update(self, _, scalar):
        self.sum_metric += scalar
        self.global_sum_metric += scalar
        self.num_inst += 1
        self.global_num_inst += 1


def label_smoothing(labels, classes, eta):
    return labels.one_hot(classes, on_value=1 - eta + eta / classes, off_value=eta / classes)


def model_fit(args, net, train_data, eval_metric, optimizer,
              optimizer_params, lr_scheduler, eval_data, global_metrics, kvstore, kv,
              begin_epoch, num_epoch, run_epoch, model_prefix):
    if not isinstance(eval_metric, mx.metric.EvalMetric):
        eval_metric = mx.metric.create(eval_metric)
    loss_metric = ScalarMetric()

    if 'horovod' in kvstore:
        trainer = hvd.DistributedTrainer(
            net.collect_params(), optimizer, optimizer_params)
    else:
        trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params,
                                kvstore=kv, update_on_kvstore=False)

    if args.amp:
        amp.init_trainer(trainer)

    partition_handler = PartitionSignalHandler(1)

    sparse_label_loss = (args.label_smoothing == 0 and args.mixup == 0)
    loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=sparse_label_loss)
    loss.hybridize(static_shape=True, static_alloc=True)

    local_batch_size = train_data.batch_size
    total_batch_size = local_batch_size * train_data._num_gpus * \
        (hvd.size() if 'horovod' in kvstore else 1)
    durations = []

    epoch_size = get_epoch_size(args, kv)
    run_epoch = num_epoch if (run_epoch == -1) else (begin_epoch + run_epoch)

    def transform_data(images, labels):
        if args.mixup != 0:
            coeffs = mx.nd.array(np.random.beta(args.mixup, args.mixup, size=images.shape[0])).as_in_context(
                images.context)
            image_coeffs = coeffs.astype(
                images.dtype, copy=False).reshape(*coeffs.shape, 1, 1, 1)
            ret_images = image_coeffs * images + \
                (1 - image_coeffs) * images[::-1]

            ret_labels = label_smoothing(
                labels, args.num_classes, args.label_smoothing)
            label_coeffs = coeffs.reshape(*coeffs.shape, 1)
            ret_labels = label_coeffs * ret_labels + \
                (1 - label_coeffs) * ret_labels[::-1]
        else:
            ret_images = images
            if not sparse_label_loss:
                ret_labels = label_smoothing(
                    labels, args.num_classes, args.label_smoothing)
            else:
                ret_labels = labels

        return ret_images, ret_labels

    i = -1
    best_accuracy = -1
    for epoch in range(begin_epoch, min(run_epoch, num_epoch)):
        tic = time.time()
        btic = time.time()
        etic = time.time()

        train_data.reset()
        eval_metric.reset()
        loss_metric.reset()

        logging.info('Starting epoch {}'.format(epoch))
        outputs = []
        if not partition_handler.should_end():
            for i, batches in enumerate(train_data):
                # synchronize to previous iteration
                # for o in outputs:
                #    o.wait_to_read()

                trainer.set_learning_rate(lr_scheduler(epoch + i / epoch_size))

                data = [b.data[0] for b in batches]
                label = [b.label[0].as_in_context(
                    b.data[0].context) for b in batches]
                orig_label = label

                data, label = zip(*starmap(transform_data, zip(data, label)))

                outputs = []
                Ls = []
                with ag.record():
                    for x, y in zip(data, label):
                        z = net(x)
                        L = loss(z, y)
                        # store the loss and do backward after we have done forward
                        # on all GPUs for better speed on multiple GPUs.
                        Ls.append(L)
                        outputs.append(z)

                    if args.amp:
                        with amp.scale_loss(Ls, trainer) as scaled_loss:
                            ag.backward(scaled_loss)
                    else:
                        ag.backward(Ls)

                if 'horovod' in kvstore:
                    trainer.step(local_batch_size)
                else:
                    trainer.step(total_batch_size)

                loss_metric.update(..., np.mean(
                    [l.asnumpy() for l in Ls]).item())

                if args.disp_batches and not (i + 1) % args.disp_batches:
                    dllogger_it_data = {
                        'train.loss': loss_metric.get()[1],
                        'train.ips': args.disp_batches * total_batch_size / (time.time() - btic),
                        'train.lr': trainer.learning_rate
                    }
                    dllogger.log((epoch, i), data=dllogger_it_data)

                    loss_metric.reset_local()
                    btic = time.time()

                durations.append(time.time() - tic)
                tic = time.time()
        else:
            break

        durations = durations[min(len(durations) // 10, 100):]
        dllogger_epoch_data = {
            'train.loss': loss_metric.get_global()[1],
            'train.ips': total_batch_size / np.mean(durations)
        }

        should_break = partition_handler.sync()
        if args.mode == 'train_val':
            logging.info('Validating epoch {}'.format(epoch))
            score, duration_stats, _ = model_score(
                args, net, eval_data, eval_metric, kvstore)

            dllogger_epoch_data.update(
                starmap(lambda key, val: (
                    'val.{}'.format(key), val), zip(*score))
            )
            dllogger_epoch_data.update(
                starmap(lambda key, val: ('val.{}'.format(key), val),
                        duration_stats.items())
            )

            score = dict(zip(*score))
            accuracy = score.get('accuracy', -1)
            save_checkpoint(net, epoch, accuracy, best_accuracy,
                            model_prefix, args.workspace,
                            args.save_frequency if args.mode == "train_val" else -1,
                            kvstore, force_save=should_break)
            best_accuracy = max(best_accuracy, accuracy)
        global_metrics.update_dict(dllogger_epoch_data)
        dllogger.log(step=(epoch,), data=dllogger_epoch_data)


def fit(args, model, data_loader):
    """
    train a model
    args : argparse returns
    model : the the neural network model
    data_loader : function that returns the train and val data iterators
    """

    start_time = time.time()

    # select gpu for horovod process
    if 'horovod' in args.kv_store:
        args.gpus = [args.gpus[hvd.local_rank()]]

    if args.amp:
        amp.init()

    if args.seed is not None:
        logging.info('Setting seeds to {}'.format(args.seed))
        random.seed(args.seed)
        np.random.seed(args.seed)
        mx.random.seed(args.seed)

    # kvstore
    if 'horovod' in args.kv_store:
        kv = None
        rank = hvd.rank()
        num_workers = hvd.size()
    else:
        kv = mx.kvstore.create(args.kv_store)
        rank = kv.rank
        num_workers = kv.num_workers

    if args.test_io:
        train, val = data_loader(args, kv)

        if args.test_io_mode == 'train':
            data_iter = train
        else:
            data_iter = val

        tic = time.time()
        for i, batch in enumerate(data_iter):
            if isinstance(batch, list):
                for b in batch:
                    for j in b.data:
                        j.wait_to_read()
            else:
                for j in batch.data:
                    j.wait_to_read()
            if (i + 1) % args.disp_batches == 0:
                logging.info('Batch [{}]\tSpeed: {:.2f} samples/sec'.format(
                    i, args.disp_batches * args.batch_size / (time.time() - tic)))
                tic = time.time()
        return

    start_epoch = load_model(args, model) + 1
    if start_epoch == 0:
        # all initializers should be specified in the model definition.
        # if not, this will raise an error
        model.initialize(mx.init.Initializer())
    logging.info(f"starting epoch {start_epoch}")

    # devices for training
    devs = list(map(mx.gpu, args.gpus))
    model.collect_params().reset_ctx(devs)

    if args.mode == 'pred':
        logging.info('Infering image {}'.format(args.data_pred))
        model_pred(args, model, data.load_image(args, args.data_pred, devs[0]))
        return

    # learning rate
    lr_scheduler = get_lr_scheduler(args)

    optimizer_params = {
        'learning_rate': 0,
        'wd': args.wd,
        'multi_precision': True,
    }

    # Only a limited number of optimizers have 'momentum' property
    has_momentum = {'sgd', 'dcasgd', 'nag', 'signum', 'lbsgd'}
    if args.optimizer in has_momentum:
        optimizer_params['momentum'] = args.mom

    # evaluation metrices
    if not args.no_metrics:
        eval_metrics = ['accuracy']
        eval_metrics.append(mx.metric.create(
            'top_k_accuracy', top_k=5))
    else:
        eval_metrics = []

    train, val = data_loader(args, kv)
    train = BenchmarkingDataIter(train, args.benchmark_iters)
    if val is not None:
        val = BenchmarkingDataIter(val, args.benchmark_iters)

    if 'horovod' in args.kv_store:
        # Fetch and broadcast parameters
        params = model.collect_params()
        if params is not None:
            hvd.broadcast_parameters(params, root_rank=0)
        ctx = mx.gpu(hvd.local_rank())
        tensor1 = mx.nd.zeros(shape=(1,), dtype='float32', ctx=ctx)
        tensor2 = mx.nd.zeros(shape=(1,), dtype='float32', ctx=ctx)
        tensor1, tensor2 = hvd.grouped_allreduce([tensor1,tensor2])
        
    global_metrics = CompositeMeter()
    if args.mode in ['train_val', 'train']:
        global_metrics.register_metric('train.loss', MinMeter())
        global_metrics.register_metric('train.ips', AvgMeter())

    if args.mode in ['train_val', 'val']:
        global_metrics.register_metric('val.accuracy', MaxMeter())
        global_metrics.register_metric('val.top_k_accuracy_5', MaxMeter())
        global_metrics.register_metric('val.ips', AvgMeter())
        global_metrics.register_metric('val.latency_avg', AvgMeter())

    if args.mode in ['val']:
        global_metrics.register_metric('val.latency_50', PercentileMeter(50))
        global_metrics.register_metric('val.latency_90', PercentileMeter(90))
        global_metrics.register_metric('val.latency_95', PercentileMeter(95))
        global_metrics.register_metric('val.latency_99', PercentileMeter(99))
        global_metrics.register_metric('val.latency_100', PercentileMeter(100))

    # run
    if args.mode in ['train_val', 'train']:
        model_fit(
            args,
            model,
            train,
            begin_epoch=start_epoch,
            num_epoch=args.num_epochs,
            run_epoch=args.run_epochs,
            eval_data=val,
            eval_metric=eval_metrics,
            global_metrics=global_metrics,
            kvstore=args.kv_store,
            kv=kv,
            optimizer=args.optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            model_prefix=args.model_prefix,
        )
    elif args.mode == 'val':
        for epoch in range(args.num_epochs):  # loop for benchmarking
            score, duration_stats, durations = model_score(
                args, model, val, eval_metrics, args.kv_store)
            dllogger_data = dict(starmap(lambda key, val: (
                'val.{}'.format(key), val), zip(*score)))
            dllogger_data.update(
                starmap(lambda key, val: ('val.{}'.format(key), val),
                        duration_stats.items())
            )
            global_metrics.update_dict(dllogger_data)
            for percentile in [50, 90, 95, 99, 100]:
                metric_name = 'val.latency_{}'.format(percentile)
                dllogger_data[metric_name] = np.percentile(
                    durations, percentile)
                global_metrics.update_metric(metric_name, durations)
            dllogger.log(step=(epoch,), data=dllogger_data)
    else:
        raise ValueError('Wrong mode')

    mx.nd.waitall()
    dllogger.log(tuple(), data=global_metrics.get())
