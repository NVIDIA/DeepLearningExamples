# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
# ==============================================================================
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.compat.v1.saved_model import tag_constants, signature_constants


def export_model(model_dir, prec, tf_trt_model_dir=None):
    model = tf.saved_model.load(model_dir)
    input_shape = [1, 384]
    dummy_input = tf.constant(tf.zeros(input_shape, dtype=tf.int32))
    x = [
        tf.constant(dummy_input, name='input_word_ids'),
        tf.constant(dummy_input, name='input_mask'),
        tf.constant(dummy_input, name='input_type_ids'),
    ]
    _ = model(x)

    trt_prec = trt.TrtPrecisionMode.FP32 if prec == "fp32" else trt.TrtPrecisionMode.FP16
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=model_dir,
        conversion_params=trt.TrtConversionParams(precision_mode=trt_prec),
    )
    converter.convert()
    tf_trt_model_dir = tf_trt_model_dir or f'/tmp/tf-trt_model_{prec}'
    converter.save(tf_trt_model_dir)
    print(f"TF-TRT model saved at {tf_trt_model_dir}")

class SavedModel:
    def __init__(self, model_dir, precision):
        self.saved_model_loaded = tf.saved_model.load(model_dir, tags=[tag_constants.SERVING])
        self.graph_func = self.saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self.precision = tf.float16 if precision == "amp" else tf.float32

    def __call__(self, x, **kwargs):
        return self.infer_step(x)

    @tf.function
    def infer_step(self, x):
        output = self.graph_func(**x)
        return output['start_positions'], output['end_positions']

class TFTRTModel:
    def __init__(self, model_dir, precision):
        temp_tftrt_dir = f"/tmp/tf-trt_model_{precision}"
        export_model(model_dir, precision, temp_tftrt_dir)
        saved_model_loaded = tf.saved_model.load(temp_tftrt_dir, tags=[tag_constants.SERVING])
        print(f"TF-TRT model loaded from {temp_tftrt_dir}")
        self.graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self.precision = tf.float16 if precision == "amp" else tf.float32

    def __call__(self, x, **kwargs):
        return self.infer_step(x)

    @tf.function
    def infer_step(self, x):
        output = self.graph_func(**x)
        return output['start_positions'], output['end_positions']
