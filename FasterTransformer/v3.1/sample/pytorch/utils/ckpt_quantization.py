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

import sys
import argparse
import re
import numpy as np
import torch

ACTIVATION_AMAX_NUM = 80
INT8O_GEMM_NUM = 8


def checkpoint_quantization(init_dict, is_per_channel, module_path='./', ths_path='./lib/libths_fastertransformer.so', verbose=True):
    print("Quantizing checkpoint ...")
    try:
        sys.path.insert(0, module_path)
        from th_fastertransformer import weight_quantize
    except:
        torch.classes.load_library(ths_path)
        weight_quantize = torch.ops.fastertransformer.weight_quantize

    def init_graph():
        layer_num = 0
        regex = re.compile('layer.\d+')
        amaxTotalNum = 0
        for name, tensor_value in init_dict.items():
            if "intermediate.dense.weight" in name and amaxTotalNum == 0:
                amaxTotalNum = ACTIVATION_AMAX_NUM + 9 * tensor_value.size(1) + INT8O_GEMM_NUM
                if verbose:
                    print("amaxTotalNum", amaxTotalNum)
                    print("Hidden size:", tensor_value.size(1))
            tmp = regex.findall(name)
            if len(tmp) < 1:
                continue
            num_tmp = int(tmp[0].replace("layer.", ""))
            if layer_num < num_tmp:
                layer_num = num_tmp
        layer_num = layer_num + 1
        #add new var for amax
        for i in range(layer_num):
            init_dict["bert.encoder.layer.{}.amaxList".format(i)] = torch.zeros((amaxTotalNum,), dtype=torch.float32)
        return layer_num, amaxTotalNum
    layer_num, amaxTotalNum = init_graph()

    kernel_name_list = ["attention.self.query",
                        "attention.self.key",
                        "attention.self.value",
                        "attention.output.dense",
                        "intermediate.dense",
                        "output.dense"]

    amax_name_list =   ["attention.self.query._input_quantizer",
                        "attention.self.query._aftergemm_quantizer",
                        "attention.self.matmul_q_input_quantizer",
                        "attention.self.key._aftergemm_quantizer",
                        "attention.self.matmul_k_input_quantizer",
                        "attention.self.value._aftergemm_quantizer",
                        "attention.self.matmul_v_input_quantizer",
                        "attention.self.softmax_input_quantizer",
                        "attention.self.matmul_a_input_quantizer",
                        "attention.output.dense._input_quantizer",
                        "attention.output.dense._aftergemm_quantizer",
                        "intermediate.dense._input_quantizer",
                        "intermediate.dense._aftergemm_quantizer",
                        "output.dense._input_quantizer",
                        "output.dense._aftergemm_quantizer",
                        "special_F2Bias_scale",
                        ]

    int8O_gemm_weight_amax_list = [0 for i in range(INT8O_GEMM_NUM)]
    int8O_gemm_weight_list = ["attention.self.query", 
                              "attention.self.key", 
                              "attention.self.value", 
                              "attention.self.matmul_k_input_quantizer",
                              "attention.self.matmul_v_input_quantizer", 
                              "attention.output.dense", 
                              "intermediate.dense", 
                              "output.dense"]

    int8O_gemm_input_amax_list = [0 for i in range(INT8O_GEMM_NUM)]
    int8O_gemm_input_list = ["attention.self.query._input_quantizer",
                             "attention.self.key._input_quantizer",
                             "attention.self.value._input_quantizer", 
                             "attention.self.matmul_q_input_quantizer", 
                             "attention.self.matmul_a_input_quantizer",
                             "attention.output.dense._input_quantizer",
                             "intermediate.dense._input_quantizer", 
                             "output.dense._input_quantizer"]
    
    int8O_gemm_output_amax_list = [0 for i in range(INT8O_GEMM_NUM)]
    int8O_gemm_output_list = ["attention.self.query._aftergemm_quantizer",
                              "attention.self.key._aftergemm_quantizer",
                              "attention.self.value._aftergemm_quantizer",
                              "attention.self.softmax_input_quantizer", 
                              "attention.output.dense._input_quantizer",
                              "attention.output.dense._aftergemm_quantizer",
                              "intermediate.dense._aftergemm_quantizer", 
                              "output.dense._aftergemm_quantizer"]

    same_value_tuple_list = [("attention.self.query._input_quantizer",
                              "attention.self.key._input_quantizer",
                              "attention.self.value._input_quantizer",
                              "attention.output.add_residual_input_quantizer"),
                             ("intermediate.dense._input_quantizer",
                              "output.add_residual_input_quantizer")]

    factor = 1000000.0
    for i in range(layer_num):
        amaxList = np.zeros([amaxTotalNum]).astype(np.float32)
        amax_id = 0
        # verify some quantizers have same value. input_quantizer is per-tensor quantization
        for same_value_tuple in same_value_tuple_list:
            tmp_v = init_dict["bert.encoder.layer.{}.{}._amax".format(i, same_value_tuple[0])].numpy()
            for same_value_name in same_value_tuple:
                tmp_v_2 = init_dict["bert.encoder.layer.{}.{}._amax".format(i, same_value_name)].numpy()
                assert(np.allclose(tmp_v, tmp_v_2))

        for amax_name in amax_name_list:
            if amax_name == "special_F2Bias_scale":
                if i != layer_num - 1:
                    quant_max = init_dict["bert.encoder.layer.{}.{}._amax".format(i+1, amax_name_list[0])].item()
                    amax = abs(quant_max)
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

            quant_max = init_dict["bert.encoder.layer.{}.{}._amax".format(i, amax_name)].item()
            amax = abs(quant_max)#round(abs(quant_max)*factor)/factor
            if amax_name in int8O_gemm_input_list:
                int8O_gemm_input_amax_list[int8O_gemm_input_list.index(amax_name)] = amax
                if amax_name == "attention.self.query._input_quantizer":
                    int8O_gemm_input_amax_list[int8O_gemm_input_list.index("attention.self.key._input_quantizer")] = amax
                    int8O_gemm_input_amax_list[int8O_gemm_input_list.index("attention.self.value._input_quantizer")] = amax
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
            # if verbose:
            #     print(i, amax_name)
            #     print('quant_max:', quant_max)
            #     print('amax:', amax)
        if verbose:
            print("done process layer_{} activation amax".format(i))

        #kernel amax starts from ACTIVATION_AMAX_NUM
        assert amax_id == 64
        amax_id = ACTIVATION_AMAX_NUM
        for kernel_id, kernel_name in enumerate(kernel_name_list):
            kernel = init_dict["bert.encoder.layer.{}.{}.weight".format(i, kernel_name)].transpose(-1, -2).contiguous()
            quant_max2 = init_dict["bert.encoder.layer.{}.{}._weight_quantizer._amax".format(i, kernel_name)]
            amax2 = abs(quant_max2)
            kernel_processed, quant_max_processed = weight_quantize(kernel, amax2, -amax2, is_per_channel)
            init_dict["bert.encoder.layer.{}.{}.weight".format(i, kernel_name)] = kernel_processed
            if kernel_name in int8O_gemm_weight_list:
                int8O_gemm_weight_amax_list[int8O_gemm_weight_list.index(kernel_name)] = quant_max_processed[0]
            for e in quant_max_processed:
                amaxList[amax_id] = e
                amax_id += 1
            # if verbose:
            #     print(i, kernel_name)
            #     print('kernel:', kernel)
            #     print('quant_max2:', quant_max2)
            #     print('quant_max_processed_:', quant_max_processed)
            
        #for int8O gemm deQuant
        for j in range(INT8O_GEMM_NUM):
            amaxList[amax_id] = (int8O_gemm_input_amax_list[j]*int8O_gemm_weight_amax_list[j])/(127.0*int8O_gemm_output_amax_list[j])
            amax_id += 1

        init_dict["bert.encoder.layer.{}.amaxList".format(i)] = torch.tensor(amaxList, dtype=torch.float32)
        if verbose:
            print("done process layer_{} kernel weight".format(i))

    print("Quantizing checkpoint done.")
    return init_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module_path', type=str, default='./',
                        help='directory containing the th_fastertransformer dynamic lib')
    parser.add_argument('--ths_path', type=str, default='./lib/libths_fastertransformer.so',
                        help='path of the ths_fastertransformer dynamic lib file')
    parser.add_argument('--init_ckpt', type=str,
                        help='checkpoint to be processed')
    parser.add_argument('--quantized_ckpt', type=str,
                        help='quantized checkpoint')
    parser.add_argument('--int8_mode', type=int,
                        help='int8 mode in FasterTransformer.')
    args = parser.parse_args()
    if args.int8_mode == 1:
        per_channel_quantization = True
    elif args.int8_mode == 2:
        per_channel_quantization = False
    else:
        raise ValueError("wrong int8_mode argument")
    init_dict = torch.load(args.init_ckpt, map_location='cpu')
    init_dict = checkpoint_quantization(init_dict, per_channel_quantization, args.module_path, args.ths_path)
    torch.save(init_dict, args.quantized_ckpt)
    print("Saving quantized checkpoint done.")
