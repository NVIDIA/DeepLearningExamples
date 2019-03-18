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


from dllogger import LOGGER, CompactBackend, StdOutBackend, MLPerfBackend, JsonBackend, Scope, AverageMeter, tags
from argparse import ArgumentParser
import random

@LOGGER.timed_function("train")
def train():
    for i in range(0, 10):
    #for i in LOGGER.epoch_generator_wrapper(range(0, 10)):
        LOGGER.epoch_start()
        LOGGER.log("epoch_nr", i)
        LOGGER.log("epochs2", 2 * i)
        train_epoch(i)
        LOGGER.epoch_stop()


@LOGGER.timed_function("train_epoch", "epoch")
def train_epoch(epoch):
    for i in range(epoch*30, (epoch+1)*30, 2):
    #for i in LOGGER.iteration_generator_wrapper(range(epoch*10, (epoch+1)*10, 2)):
        LOGGER.iteration_start()
        LOGGER.log("loss", i*epoch)
        LOGGER.iteration_stop()
    if epoch % 3 == 1:
        with LOGGER.timed_block("eval"):
            LOGGER.log("accuracy", i * epoch)
            LOGGER.log_event(key="ep divisible by 3", value=epoch)


def main():
    LOGGER.set_model_name('ResNet')
    LOGGER.set_backends([
            StdOutBackend(log_file='std.out',
                logging_scope=Scope.TRAIN_ITER),
            CompactBackend(log_file=None,
                logging_scope=Scope.TRAIN_ITER, iteration_interval=5),
            JsonBackend(log_file='dummy.json',
                logging_scope=Scope.TRAIN_ITER, iteration_interval=4)
            ])

    parser = ArgumentParser()
    parser.add_argument('--dummy', type=str, default='default_dummy_value')
    args = parser.parse_args()

    LOGGER.log_hardware()
    LOGGER.log_args(args)

    LOGGER.log(tags.RUN_INIT)
    LOGGER.register_metric('loss', meter=AverageMeter(), metric_scope=Scope.TRAIN_ITER)
    LOGGER.register_metric('epoch_nr', metric_scope=Scope.EPOCH)
    LOGGER.register_metric('epochs2')

    with LOGGER.timed_block(tags.SETUP_BLOCK):
        print("This is setup.")

    with LOGGER.timed_block(tags.PREPROC_BLOCK):
        print("This is preprocessing.")

    with LOGGER.timed_block(tags.RUN_BLOCK):
        print("This is run.")
        train()
        print("This is the end.")

    LOGGER.log(tags.RUN_FINAL)

    LOGGER.finish()

def main2():
    LOGGER.set_backends([
            CompactBackend(log_file=None,
                logging_scope=Scope.TRAIN_ITER),
            StdOutBackend(log_file='std.out',
                logging_scope=Scope.EPOCH, iteration_interval=4),
            JsonBackend(log_file='dummy.json',
                logging_scope=Scope.TRAIN_ITER, iteration_interval=1)
            ])
    LOGGER.log_hardware()

    data_x = range(0,10)
    data_y = [3.*x + 2. for x in data_x]

    data = list(zip(data_x, data_y))

    LOGGER.register_metric('l', AverageMeter(), metric_scope=Scope.TRAIN_ITER)
    LOGGER.register_metric('a', metric_scope=Scope.TRAIN_ITER)
    LOGGER.register_metric('b', metric_scope=Scope.TRAIN_ITER)

    LOGGER.info('RUN_INIT')

    model_a = 1.
    model_b = 0.

    def model(ma, mb, x):
        return ma*x+mb

    def loss(y, t):
        return (y-t)**2

    def update_a(ma, mb, x, t):
        return ma - 0.001 * 2*x*(ma*x+mb-t)

    def update_b(ma, mb, x, t):
        return mb - 0.001 * 2*(ma*x+mb-t)

    for e in range(0, 5):
        LOGGER.epoch_start()
        for (x, t) in data:
            LOGGER.iteration_start()
            y = model(model_a, model_b, x)
            model_a = update_a(model_a, model_b, x, t)
            model_b = update_b(model_a, model_b, x, t)
            l = loss(y, t)
            LOGGER.info('b', model_b)
            LOGGER.debug('a', model_a)
            LOGGER.warning('l', l)
            #LOGGER.log('a', model_a)
            LOGGER.iteration_stop()
        LOGGER.epoch_stop()

    #for e in LOGGER.epoch_generator_wrapper(range(0, 10)):
    #    for (x, t) in LOGGER.iteration_generator_wrapper(random.sample(data, len(data))):
    #        y = model(model_a, model_b, x)
    #        model_a = update_a(model_a, model_b, x, t)
    #        model_b = update_b(model_a, model_b, x, t)
    #        l = loss(y, t)
    #        LOGGER.debug('a', model_a)
    #        LOGGER.info('b', model_b)
    #        LOGGER.warning('l', l)


    LOGGER.finish()

    print("FINAL: {}*x+{}".format(model_a, model_b))


if __name__ == '__main__':
    main()
