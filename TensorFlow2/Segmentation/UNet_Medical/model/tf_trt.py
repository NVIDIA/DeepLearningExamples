import os
from operator import itemgetter

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.compat.v1.saved_model import tag_constants, signature_constants


def export_model(model_dir, prec, tf_trt_model_dir=None):
    model = tf.keras.models.load_model(os.path.join(model_dir, f'saved_model_{prec}'))
    input_shape = [1, 572, 572, 1]
    dummy_input = tf.constant(tf.zeros(input_shape, dtype=tf.float32 if prec=="fp32" else tf.float16))
    _ = model(dummy_input, training=False)

    trt_prec = trt.TrtPrecisionMode.FP32 if prec == "fp32" else trt.TrtPrecisionMode.FP16
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=os.path.join(model_dir, f'saved_model_{prec}'),
        conversion_params=trt.TrtConversionParams(precision_mode=trt_prec),
    )
    converter.convert()
    tf_trt_model_dir = tf_trt_model_dir or f'/tmp/tf-trt_model_{prec}'
    converter.save(tf_trt_model_dir)
    print(f"TF-TRT model saved at {tf_trt_model_dir}")


def _force_gpu_resync(func):
    p = tf.constant(0.)  # Create small tensor to force GPU resync

    def wrapper(*args, **kwargs):
        rslt = func(*args, **kwargs)
        (p + 1.).numpy()  # Sync the GPU
        return rslt

    return wrapper


class TFTRTModel:
    def __init__(self, model_dir, precision, output_tensor_name="output_1"):
        temp_tftrt_dir = f"/tmp/tf-trt_model_{precision}"
        export_model(model_dir, precision, temp_tftrt_dir)
        saved_model_loaded = tf.saved_model.load(temp_tftrt_dir, tags=[tag_constants.SERVING])
        print(f"TF-TRT model loaded from {temp_tftrt_dir}")
        self.graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self.output_tensor_name = output_tensor_name
        self.precision = tf.float16 if precision == "amp" else tf.float32

    def __call__(self, x, **kwargs):
        return self.infer_step(x)

    #@_force_gpu_resync
    @tf.function(jit_compile=False)
    def infer_step(self, batch_x):
        if batch_x.dtype != self.precision:
            batch_x = tf.cast(batch_x, self.precision)
        output = self.graph_func(batch_x)
        return itemgetter(self.output_tensor_name)(output)
