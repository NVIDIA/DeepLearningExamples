# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# usage example
# export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
# python profile_transformer_inference.py --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --tf_profile=false --output_dir=mrpc_output --profiling_output_file=time_elapsed --xla=false --floatx=float32
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.client import device_lib
import time
import contextlib
from tensorflow.python.client import timeline
import os
import tensorflow as tf
import fast_infer_util as fiu
import numpy as np
import profile_util
import sys
import my_modeling
bert_submodule = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bert')
sys.path.insert(0, bert_submodule)
import run_classifier
import optimization

flags = tf.flags
FLAGS = flags.FLAGS


# stacked transformer encoders
class TransformerModel(object):
    def __init__(self,
                 config,
                 is_training,
                 input_tensor,
                 attention_mask,
                 transformer_model_fn,
                 scope=None):
        config = my_modeling.copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = my_modeling.get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        with tf.variable_scope(scope, default_name="bert"):
            with tf.variable_scope("encoder"):
                # Run the stacked transformer.
                # `sequence_output` shape = [batch_size, seq_length, hidden_size].
                self.all_encoder_layers = transformer_model_fn(
                    input_tensor=input_tensor,
                    attention_mask=attention_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=my_modeling.get_activation(
                        config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True)

            self.sequence_output = self.all_encoder_layers[-1]
            with tf.variable_scope("pooler"):
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.layers.dense(
                    first_token_tensor,
                    config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=my_modeling.create_initializer(config.initializer_range))

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        return self.sequence_output


def model_fn_builder(bert_config, transformer_model_fn):

    def model_fn(input_tensor, attention_mask):  # pylint: disable=unused-argument
        model = TransformerModel(
            config=bert_config,
            is_training=False,
            input_tensor=input_tensor,
            attention_mask=attention_mask,
            transformer_model_fn=transformer_model_fn)
        seq_output = model.get_sequence_output()
        return seq_output

    return model_fn


def profile_model(config, jit_xla, num_iter):
    # initialize data
    input_data = np.random.randn(
        FLAGS.predict_batch_size, FLAGS.max_seq_length, config.hidden_size)
    attention_mask = np.random.randint(2, size=(
        FLAGS.predict_batch_size, FLAGS.max_seq_length))
    attention_mask = np.repeat(
        attention_mask[:, np.newaxis, :], FLAGS.max_seq_length, axis=1)

    model_fn_tf = model_fn_builder(config, my_modeling.transformer_model)
    model_fn_ft = model_fn_builder(config, fiu.fast_transformer_model_trans)

    def graph_fn_builder(model_fn):
        def graph_fn():
            input_tensor = tf.constant(input_data, dtype=FLAGS.floatx)
            mask_tensor = tf.constant(attention_mask, dtype=FLAGS.floatx)

            output_var = model_fn(input_tensor, mask_tensor)
            # for saving memcopy time
            return tf.reduce_mean(output_var)
        return graph_fn

    if FLAGS.tf_profile:
        tf.logging.info("***** Running tensorflow transformer*****")
        p1 = profile_util.Profiler(os.path.join(
            FLAGS.output_dir, 'prof/bert_origin'))
        t1, r1 = profile_util.run_profile(graph_fn_builder(
            model_fn_tf), jit_xla, num_iter, p1, init_checkpoint=FLAGS.init_checkpoint)
        tf.reset_default_graph()
        tf.logging.info("***** Running fast transformer*****")
        p2 = profile_util.Profiler(os.path.join(
            FLAGS.output_dir, 'prof/bert_fastinfer'))
        t2, r2 = profile_util.run_profile(graph_fn_builder(
            model_fn_ft), jit_xla, num_iter, p2, init_checkpoint=FLAGS.init_checkpoint)

    else:
        tf.logging.info("***** Running tensorflow transformer*****")
        t1, r1 = profile_util.run_profile(graph_fn_builder(
            model_fn_tf), jit_xla, num_iter, check_result=False, init_checkpoint=FLAGS.init_checkpoint)
        tf.reset_default_graph()
        tf.logging.info("***** Running fast transformer*****")
        t2, r2 = profile_util.run_profile(graph_fn_builder(
            model_fn_ft), jit_xla, num_iter, check_result=False, init_checkpoint=FLAGS.init_checkpoint)

    # check errors
    print('average time (seconds) elasped original tensorflow:', t1)
    print('average time (seconds) elasped fast transformer:', t2)

    if len(r1) + len(r2) > 0:
        check_res = np.asarray([np.allclose(
            r1[i], r2[i], atol=1e-4, rtol=0) for i in range(num_iter)])
        if check_res.all():
            print('Pass')
            print(np.mean(r1))
            print(np.mean(r2))
        else:
            for i in np.where(np.logical_not(check_res))[0]:
                diff = np.fabs(r1[i] - r2[i])
                idx = np.unravel_index(diff.argmax(), diff.shape)
                print('Failed iter:', i, "max diff:",
                      diff[idx], idx, r1[i][idx], r2[i][idx])

    return t1, t2


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    batch_size = [8]
    seq_length = [128]
    num_hidden_layers = [12]
    attention_heads_num_size = [(12, 64)]
    num_iter = 20
    interval = 0

    # collect results of both original bert and fast transformer
    jit_xla = tf.OptimizerOptions.ON_1 if FLAGS.xla else 0
    config = my_modeling.BertConfig(vocab_size=0)
    tf.gfile.MakeDirs(FLAGS.output_dir)
    local_device_protos = device_lib.list_local_devices()
    with open(os.path.join(FLAGS.output_dir, FLAGS.profiling_output_file), 'w') as f:
        for x in local_device_protos:
            if x.device_type == 'GPU':
                f.write(x.physical_device_desc + '\n')
        f.write(str(FLAGS.floatx) + '\t' + 'XLA: ' + str(FLAGS.xla) + '\n')
        f.write('batch_size\tseq_length\thidden_layers\tattention_heads\tattention_head_size\tTensorflow\tFasterTransformer\n')
        for bs in batch_size:
            FLAGS.predict_batch_size = bs
            for sl in seq_length:
                FLAGS.max_seq_length = sl
                for hidden_layers in num_hidden_layers:
                    config.num_hidden_layers = hidden_layers
                    for head_num, head_size in attention_heads_num_size:
                        config.num_attention_heads = head_num
                        config.hidden_size = head_num * head_size

                        time.sleep(interval)
                        t1, t2 = profile_model(config, jit_xla, num_iter)
                        tmp = [FLAGS.predict_batch_size, FLAGS.max_seq_length, hidden_layers, head_num, head_size,
                               '{:.6}'.format(t1), '{:.6}'.format(t2)]
                        f.write('\t'.join([str(x) for x in tmp]) + '\n')


if __name__ == "__main__":
    flags.mark_flag_as_required("output_dir")
    flags.DEFINE_string("profiling_output_file", None,
                        "The output file for profiling results.")
    flags.mark_flag_as_required("profiling_output_file")
    flags.DEFINE_string("floatx", "float32", "float32 or float16")
    flags.mark_flag_as_required("floatx")
    flags.DEFINE_bool("xla", False, "whether to turn on XLA")
    flags.mark_flag_as_required("xla")
    flags.DEFINE_bool("tf_profile", False,
                      "whether to use tensorflow profiling")
    tf.app.run()
