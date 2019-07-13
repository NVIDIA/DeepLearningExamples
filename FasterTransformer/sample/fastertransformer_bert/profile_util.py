# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

from tensorflow.python.client import device_lib
import time
import contextlib
from tensorflow.python.client import timeline
import os
import tensorflow as tf


class Profiler():
    def __init__(self, profile_name_pref):
        self.profile_name_pref = profile_name_pref
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()
        self.ctr = 0
        self.time_avg = 0

    @contextlib.contextmanager
    def prof_run(self):
        start = time.time()
        yield
        end = time.time()
        self.time_avg = (self.time_avg * self.ctr + end - start)/(self.ctr + 1)

        fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        file_name = self.profile_name_pref + '_' + str(self.ctr) + '.json'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w') as f:
            f.write(chrome_trace)
        self.ctr += 1


def run_profile(graph_fn, jit_xla, num_iter, profiler=None, init_checkpoint=None, check_result=True, dryrun_iter=1):
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = jit_xla

    fetches = graph_fn()

    with tf.Session(config=config) as sess:
        # init
        if init_checkpoint is None:
            sess.run(tf.global_variables_initializer())
        else:
            saver = tf.train.Saver()
            saver.restore(sess, init_checkpoint)
        
        # dry run
        for _ in range(dryrun_iter):
            sess.run(fetches)
        
        res = []
        if profiler is None:
            start_time = time.time()
            if check_result:
                for _ in range(num_iter):
                    res.append(sess.run(fetches))
            else:
                for _ in range(num_iter):
                    sess.run(fetches)
            end_time = time.time()
            time_avg = (end_time - start_time)/num_iter

        else:
            if check_result:
                for _ in range(num_iter):
                    with profiler.prof_run():
                        res.append(sess.run(fetches, options=profiler.run_options, run_metadata=profiler.run_metadata))
            else:
                for _ in range(num_iter):
                    with profiler.prof_run():
                        sess.run(fetches, options=profiler.run_options, run_metadata=profiler.run_metadata)
            time_avg = profiler.time_avg

    return time_avg, res
