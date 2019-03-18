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

from absl import flags
from time import time

import tensorflow as tf

from object_detection import model_hparams
from object_detection import model_lib


flags.DEFINE_string('checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
                    '`checkpoint_dir` is not provided, benchmark is running on random model')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config file.')
flags.DEFINE_integer('warmup_iters', 100, 'Number of iterations skipped during benchmark')
flags.DEFINE_integer('benchmark_iters', 300, 'Number of iterations measured by benchmark')
flags.DEFINE_integer('batch_size', 1, 'Number of inputs processed paralelly')
FLAGS = flags.FLAGS

flags.mark_flag_as_required('pipeline_config_path')


def build_estimator():
    session_config = tf.ConfigProto()
    config = tf.estimator.RunConfig(session_config=session_config)

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(None),
        pipeline_config_path=FLAGS.pipeline_config_path)
    estimator = train_and_eval_dict['estimator']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    return  estimator, eval_input_fns[0]


def build_benchmark_input_fn(input_fn):
    def benchmark_input_fn(params={}):
        params['batch_size'] = FLAGS.batch_size
        return input_fn(params).repeat().take(FLAGS.warmup_iters + FLAGS.benchmark_iters)
    return benchmark_input_fn


class TimingHook(tf.train.SessionRunHook):
    def __init__(self):
        super(TimingHook, self).__init__()
        self.times = []

    def before_run(self, *args, **kwargs):
        super(TimingHook, self).before_run(*args, **kwargs)
        self.start_time = time()

    def log_progress(self):
        print(len(self.times) - FLAGS.warmup_iters, '/', FLAGS.benchmark_iters, ' '*10, end='\r')

    def after_run(self, *args, **kwargs):
        super(TimingHook, self).after_run(*args, **kwargs)
        self.times.append(time() - self.start_time)
        self.log_progress()

    def collect_result(self):
        return FLAGS.batch_size * FLAGS.benchmark_iters / sum(self.times[FLAGS.benchmark_iters:])

    def end(self, *args, **kwargs):
        super(TimingHook, self).end(*args, **kwargs)
        print()
        print('Benchmark result:', self.collect_result(), 'img/s')


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    estimator, eval_input_fn = build_estimator()

    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir) \
                    if FLAGS.checkpoint_dir \
                    else None
    results = estimator.predict(
        input_fn=build_benchmark_input_fn(eval_input_fn),
        checkpoint_path=checkpoint_path,
        hooks=[ TimingHook() ],
        yield_single_examples=False
    )
    list(results)

if __name__ == '__main__':
    tf.app.run()
