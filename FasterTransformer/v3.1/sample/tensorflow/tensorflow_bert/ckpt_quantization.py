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
INT8O_GEMM_NUM = 8

def checkpoint_quantization(in_checkpoint_file, out_checkpoint_file, per_channel_quantization):
    var_list = checkpoint_utils.list_variables(tf.flags.FLAGS.init_checkpoint)
    def init_graph():
        restore_vars = []
        layer_num = 0
        regex = re.compile('layer_\d+')
        amaxTotalNum = 0
        for name, shape in var_list:
            var = checkpoint_utils.load_variable(tf.flags.FLAGS.init_checkpoint, name)
            if "intermediate/dense/kernel" in name and amaxTotalNum == 0:
                amaxTotalNum = ACTIVATION_AMAX_NUM + 9*shape[0] + INT8O_GEMM_NUM
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
        amax_name_list =   ["attention/self/query/input_quantizer",
                            #Q_aftergemm_scale, 1
                            "attention/self/query/aftergemm_quantizer",
                            #Qbias_scale, 2
                            "attention/self/matmul_q_input_quantizer",
                            #K_aftergemm_scale, 3
                            "attention/self/key/aftergemm_quantizer",
                            #Kbias_scale, 4
                            "attention/self/matmul_k_input_quantizer",
                            #V_aftergemm_scale, 5
                            "attention/self/value/aftergemm_quantizer",
                            #Vbias_scale, 6
                            "attention/self/matmul_v_input_quantizer",
                            #bmm1_scale, 7
                            "attention/self/softmax_input_quantizer",
                            #Softmax_scale, 8
                            "attention/self/matmul_a_input_quantizer",
                            #bmm2_scale, 9
                            "attention/output/dense/input_quantizer",
                            #Proj_aftergemm_scale, 10
                            "attention/output/dense/aftergemm_quantizer",
                            #ProjBiasNorm_scale, 11
                            "intermediate/dense/input_quantizer",
                            #FC1_aftergemm_scale, 12
                            "intermediate/dense/aftergemm_quantizer",
                            #F1Bias_scale, 13
                            "output/dense/input_quantizer",
                            #FC2_aftergemm_scale, 14
                            "output/dense/aftergemm_quantizer",
                            #F2Bias_scale, 15
                            "special_F2Bias_scale", 
                            ]

        int8O_gemm_weight_amax_list = [0 for i in range(INT8O_GEMM_NUM)]
                                  #Q_aftergemm
        int8O_gemm_weight_list = ["attention/self/query", 
                                  #K_aftergemm
                                  "attention/self/key", 
                                  #V_aftergemm
                                  "attention/self/value", 
                                  #bmm1_aftergemm
                                  "attention/self/matmul_k_input_quantizer",
                                  #bmm2_aftergemm                                  
                                  "attention/self/matmul_v_input_quantizer", 
                                  #Proj_aftergemm
                                  "attention/output/dense", 
                                  #FC1_aftergemm
                                  "intermediate/dense", 
                                  #FC2_aftergemm
                                  "output/dense"]
        
        int8O_gemm_input_amax_list = [0 for i in range(INT8O_GEMM_NUM)]
                                 #Q_aftergemm
        int8O_gemm_input_list = ["attention/self/query/input_quantizer",
                                 #K_aftergemm
                                 "attention/self/key/input_quantizer",
                                 #V_aftergemm
                                 "attention/self/value/input_quantizer", 
                                 #bmm1_aftergemm
                                 "attention/self/matmul_q_input_quantizer", 
                                 #bmm2_aftergemm
                                 "attention/self/matmul_a_input_quantizer",
                                 #Proj_aftergemm
                                 "attention/output/dense/input_quantizer",
                                 #FC1_aftergemm
                                 "intermediate/dense/input_quantizer", 
                                 #FC2_aftergemm
                                 "output/dense/input_quantizer"]
        
        int8O_gemm_output_amax_list = [0 for i in range(INT8O_GEMM_NUM)]
                                  #Q_aftergemm
        int8O_gemm_output_list = ["attention/self/query/aftergemm_quantizer",
                                  #K_aftergemm
                                  "attention/self/key/aftergemm_quantizer",
                                  #V_aftergemm
                                  "attention/self/value/aftergemm_quantizer",
                                  #bmm1_aftergemm
                                  "attention/self/softmax_input_quantizer", 
                                  #bmm2_aftergemm
                                  "attention/output/dense/input_quantizer",
                                  #Proj_aftergemm
                                  "attention/output/dense/aftergemm_quantizer",
                                  #FC1_aftergemm
                                  "intermediate/dense/aftergemm_quantizer", 
                                  #FC2_aftergemm
                                  "output/dense/aftergemm_quantizer"]
        
        factor = 1000000.0
        for i in range(layer_num):
            amaxList = np.zeros([amaxTotalNum])
            amax_id = 0
            for amax_name in amax_name_list:
                if amax_name == "special_F2Bias_scale":
                    if i != layer_num - 1:
                        name = "bert/encoder/layer_{}/{}/quant_max:0".format(i+1, amax_name_list[0])
                        quant_max = checkpoint_utils.load_variable(tf.flags.FLAGS.init_checkpoint, name)    
                        name = "bert/encoder/layer_{}/{}/quant_min:0".format(i+1, amax_name_list[0])
                        quant_min = checkpoint_utils.load_variable(tf.flags.FLAGS.init_checkpoint, name)                     
                        if abs(quant_max) > abs(quant_min):
                            amax = abs(quant_max)#int(abs(quant_max)*factor)/factor
                        else:
                            amax = abs(quant_min)#int(abs(quant_min)*factor)/factor
                    else:
                
                        #not used, placeholder
                        amax = 1.0

                    amaxList[amax_id] = amax
                    amax_id += 1
                    amaxList[amax_id] = amax/127.0
                    amax_id += 1
                    amaxList[amax_id] = amax/127.0/127.0
                    amax_id += 1
                    amaxList[amax_id] = 127.0/amax
                    amax_id += 1
                    continue
                
                name = "bert/encoder/layer_{}/{}/quant_max:0".format(i, amax_name)
                quant_max = checkpoint_utils.load_variable(tf.flags.FLAGS.init_checkpoint, name)
                name = "bert/encoder/layer_{}/{}/quant_min:0".format(i, amax_name)
                quant_min = checkpoint_utils.load_variable(tf.flags.FLAGS.init_checkpoint, name)
                
                if abs(quant_max) > abs(quant_min):
                    amax = abs(quant_max)#int(abs(quant_max)*factor)/factor
                else:
                    amax = abs(quant_min)#int(abs(quant_min)*factor)/factor
                    
                if amax_name in int8O_gemm_input_list:
                    int8O_gemm_input_amax_list[int8O_gemm_input_list.index(amax_name)] = amax
                    if amax_name == "attention/self/query/input_quantizer":
                        int8O_gemm_input_amax_list[int8O_gemm_input_list.index("attention/self/key/input_quantizer")] = amax
                        int8O_gemm_input_amax_list[int8O_gemm_input_list.index("attention/self/value/input_quantizer")] = amax
                        
                if amax_name in int8O_gemm_output_list:
                    int8O_gemm_output_amax_list[int8O_gemm_output_list.index(amax_name)] = amax
                    
                if amax_name in int8O_gemm_weight_list:
                    int8O_gemm_weight_amax_list[int8O_gemm_weight_list.index(amax_name)] = amax      
                    
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

                name = "bert/encoder/layer_{}/{}/kernel_quantizer/quant_max:0".format(i, kernel_name)
                quant_max2 = tf.convert_to_tensor(checkpoint_utils.load_variable(tf.flags.FLAGS.init_checkpoint, name))
                
                name = "bert/encoder/layer_{}/{}/kernel_quantizer/quant_min:0".format(i, kernel_name)
                quant_min2 = tf.convert_to_tensor(checkpoint_utils.load_variable(tf.flags.FLAGS.init_checkpoint, name))
                
                kernel_processed, quant_max_processed = transformer_op_module.weight_quantize(kernel, quant_max2, quant_min2, per_channel_quantization = per_channel_quantization)
                kernel_processed_, quant_max_processed_ = sess.run([kernel_processed, quant_max_processed])
                sess.run(tf.assign(kernel, kernel_processed_)) 
                if kernel_name in int8O_gemm_weight_list:
                    int8O_gemm_weight_amax_list[int8O_gemm_weight_list.index(kernel_name)] = quant_max_processed_[0]
                for e in quant_max_processed_:
                    amaxList[amax_id] = e
                    amax_id += 1
            #for int8O gemm deQuant
            for j in range(INT8O_GEMM_NUM):
                amaxList[amax_id] = (int8O_gemm_input_amax_list[j]*int8O_gemm_weight_amax_list[j])/(127.0*int8O_gemm_output_amax_list[j])
                amax_id += 1
            amaxL = tf.get_default_graph().get_tensor_by_name("bert/encoder/layer_{}/amaxList:0".format(i))
            sess.run(tf.assign(amaxL, amaxList))
            
            print("done process layer_{} kernel weight".format(i))

        saver.save(sess, out_checkpoint_file)

if __name__ == '__main__':
    tf.flags.DEFINE_string("quantized_checkpoint", None, "quantized checkpoint file")
    tf.flags.DEFINE_string("init_checkpoint", None, "initial checkpoint file")
    tf.flags.DEFINE_integer("int8_mode", 1, "int8 mode in FasterTransformer, default as 1")
    if tf.flags.FLAGS.int8_mode == 1:
        per_channel_quantization = True
    elif tf.flags.FLAGS.int8_mode == 2:
        per_channel_quantization = False
    else:
        raise ValueError("wrong int8_mode argument")
    quantized_checkpoint_folder = "/".join(tf.flags.FLAGS.quantized_checkpoint.split("/")[:-1])
    if not os.path.exists(quantized_checkpoint_folder):
        os.system("mkdir -p " + quantized_checkpoint_folder)
    checkpoint_quantization(tf.flags.FLAGS.init_checkpoint, tf.flags.FLAGS.quantized_checkpoint, per_channel_quantization)
