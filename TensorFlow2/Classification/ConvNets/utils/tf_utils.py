import tensorflow as tf
import numpy as np
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

def get_num_params(model, readable_format=True):
    """Return number of parameters and flops."""
    nparams = np.sum([
        np.prod(v.get_shape().as_list())
        for v in model.trainable_weights
    ])
    if readable_format:
        nparams = float(nparams) * 1e-6
    return nparams
  


def get_num_flops(model, input_shape, readable_format=True):

    if hasattr(model,'model'):
        model = model.model

    forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + input_shape)])

    graph_info = profile(forward_pass.get_concrete_function().graph,
                            options=ProfileOptionBuilder.float_operation())

    # The //2 is necessary since `profile` counts multiply and accumulate
    # as two flops, here we report the total number of multiply accumulate ops
    flops = graph_info.total_float_ops // 2
    if readable_format:
        flops = float(flops) * 1e-9
    return flops