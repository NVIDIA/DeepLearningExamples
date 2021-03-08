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
#python ckpt_quantization.py --init_checkpoint=squad_model/QAT_noresidualQuant/model.ckpt-5474 --quantized_checkpoint=squad_model/QAT_noresidualQuant_quantized/model.ckpt
import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.python.ops import io_ops
from tensorflow.python.training.saver import BaseSaverBuilder
import os
import re

build_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../lib')

transformer_op_module = tf.load_op_library(
    os.path.join(build_path, 'libtf_weight_quantize.so'))

ACTIVATION_AMAX_NUM = 80

def checkpoint_quantization(in_checkpoint_file, out_checkpoint_file):
    var_list = checkpoint_utils.list_variables(tf.flags.FLAGS.init_checkpoint)
    def init_graph():
        restore_vars = []
        layer_num = 0
        regex = re.compile('layer_\d+')
        amaxTotalNum = 0
        for name, shape in var_list:
            var = checkpoint_utils.load_variable(tf.flags.FLAGS.init_checkpoint, name)
            if "intermediate/dense/kernel" in name and amaxTotalNum == 0:
                amaxTotalNum = ACTIVATION_AMAX_NUM + 9*shape[0]
                print(amaxTotalNum, shape[0])
            recon_dtype = var.dtype
            restore_vars.append(tf.get_variable(name, shape=shape, dtype=var.dtype))
            tmp = regex.findall(name)
            if len(tmp) < 1:
                continue
            num_tmp = int(tmp[0].replace("layer_", ""))
            if layer_num < num_tmp:
                layer_num = num_tmp
        layer_num = layer_num + 1
        #add new var for amax
        for i in range(layer_num):
            tf.get_variable("bert/encoder/layer_{}/amaxList".format(i), shape=[amaxTotalNum], dtype=tf.float32)
        return layer_num, amaxTotalNum, restore_vars


    layer_num, amaxTotalNum, restore_vars = init_graph()
    restorer = tf.train.Saver(restore_vars)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        restorer.restore(sess, in_checkpoint_file)
        kernel_name_list = ["attention/self/query", "attention/self/key", "attention/self/value", "attention/output/dense", "intermediate/dense", "output/dense"]

                            #input_scale, 0
        amax_name_list =   ["attention/self/query/tensor_quantizer_1",
                            #Qbias_scale, 1
                            "attention/self/tensor_quantizer",
                            #Kbias_scale, 2
                            "attention/self/tensor_quantizer_1",
                            #Vbias_scale, 3
                            "attention/self/tensor_quantizer_2",
                            #Softmax_scale, 4
                            "attention/self/tensor_quantizer_3",
                            #bmm2_scale, 5
                            "attention/output/dense/tensor_quantizer_1",
                            #ProjBiasNorm_scale, 6
                            "intermediate/dense/tensor_quantizer_1",
                            #F1Bias_scale, 7
                            "output/dense/tensor_quantizer_1",
                            ]

        factor = 1000000.0
        for i in range(layer_num):
            amaxList = np.zeros([amaxTotalNum])
            amax_id = 0
            for amax_name in amax_name_list:
                quant_max = tf.get_default_graph().get_tensor_by_name("bert/encoder/layer_{}/{}/quant_max:0".format(i, amax_name)).eval()
                quant_min = tf.get_default_graph().get_tensor_by_name("bert/encoder/layer_{}/{}/quant_min:0".format(i, amax_name)).eval()
                if abs(quant_max) > abs(quant_min):
                    amax = int(abs(quant_max)*factor)/factor
                else:
                    amax = int(abs(quant_min)*factor)/factor
                amaxList[amax_id] = amax
                amax_id += 1
                amaxList[amax_id] = amax/127.0
                amax_id += 1
                amaxList[amax_id] = amax/127.0/127.0
                amax_id += 1
                amaxList[amax_id] = 127.0/amax
                amax_id += 1
            if i != layer_num - 1:
                quant_max = tf.get_default_graph().get_tensor_by_name("bert/encoder/layer_{}/{}/quant_max:0".format(i+1, amax_name_list[0])).eval()
                quant_min = tf.get_default_graph().get_tensor_by_name("bert/encoder/layer_{}/{}/quant_min:0".format(i+1, amax_name_list[0])).eval()
                if abs(quant_max) > abs(quant_min):
                    amax = int(abs(quant_max)*factor)/factor
                else:
                    amax = int(abs(quant_min)*factor)/factor
            else:
                quant_max = tf.get_default_graph().get_tensor_by_name("bert/tensor_quantizer/quant_max:0").eval()
                quant_min = tf.get_default_graph().get_tensor_by_name("bert/tensor_quantizer/quant_min:0").eval()
                if abs(quant_max) > abs(quant_min):
                    amax = int(abs(quant_max)*factor)/factor
                else:
                    amax = int(abs(quant_min)*factor)/factor

            amaxList[amax_id] = amax
            amax_id += 1
            amaxList[amax_id] = amax/127.0
            amax_id += 1
            amaxList[amax_id] = amax/127.0/127.0
            amax_id += 1
            amaxList[amax_id] = 127.0/amax
            amax_id += 1    

            print("done process layer_{} activation amax".format(i))

            #kernel amax starts from ACTIVATION_AMAX_NUM
            amax_id = ACTIVATION_AMAX_NUM
            for kernel_id, kernel_name in enumerate(kernel_name_list):
                kernel = tf.get_default_graph().get_tensor_by_name("bert/encoder/layer_{}/{}/kernel:0".format(i, kernel_name))
                quant_max2 = tf.get_default_graph().get_tensor_by_name("bert/encoder/layer_{}/{}/tensor_quantizer/quant_max:0".format(i, kernel_name))
                quant_min2 = tf.get_default_graph().get_tensor_by_name("bert/encoder/layer_{}/{}/tensor_quantizer/quant_min:0".format(i, kernel_name))
                kernel_processed, quant_max_processed = transformer_op_module.weight_quantize(kernel, quant_max2, quant_min2)
                kernel_processed_, quant_max_processed_ = sess.run([kernel_processed, quant_max_processed])
                sess.run(tf.assign(kernel, kernel_processed_)) 
                for e in quant_max_processed_:
                    amaxList[amax_id] = e
                    amax_id += 1
            amaxL = tf.get_default_graph().get_tensor_by_name("bert/encoder/layer_{}/amaxList:0".format(i))
            sess.run(tf.assign(amaxL, amaxList))
            
            print("done process layer_{} kernel weight".format(i))

        saver.save(sess, out_checkpoint_file)

if __name__ == '__main__':
    tf.flags.DEFINE_string("quantized_checkpoint", None, "quantized checkpoint file")
    tf.flags.DEFINE_string("init_checkpoint", None, "initial checkpoint file")
    quantized_checkpoint_folder = "/".join(tf.flags.FLAGS.quantized_checkpoint.split("/")[:-1])
    if not os.path.exists(quantized_checkpoint_folder):
        os.system("mkdir -p " + quantized_checkpoint_folder)
    checkpoint_quantization(tf.flags.FLAGS.init_checkpoint, tf.flags.FLAGS.quantized_checkpoint)
