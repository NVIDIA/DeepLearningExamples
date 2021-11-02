import logging

import tensorflow as tf
from utils import data_utils

LOGGER = logging.getLogger(__name__)

NCLASSES = 1001
WIDTH = 224
HEIGHT = 224
NCHANNELS = 3
INPUT_FORMAT = "NHWC"
COMPUTE_FORMAT = "NHWC"


def get_model(
    *,
    model_dir: str,
    arch: str = "resnet50",
    precision: str = "fp32",
    use_xla: bool = True,
    use_tf_amp: bool = False,
    use_dali: bool = False,
    gpu_memory_fraction=0.7,
):
    from runtime import Runner
    from utils import hvd_wrapper as hvd

    hvd.init()

    try:
        dtype = {"fp16": tf.float16, "fp32": tf.float32}[precision.lower()]
    except KeyError:
        raise ValueError(f"Uknown precision {precision}. Allowed values: fp16|fp32")

    LOGGER.info(
        f"Creating model arch={arch} precision={precision} xla={use_xla}"
        f"tf_amp={use_tf_amp}, dali={use_dali}, gpu_memory_frac={gpu_memory_fraction}"
    )

    runner = Runner(
        n_classes=NCLASSES,
        architecture=arch,
        input_format=INPUT_FORMAT,
        compute_format=COMPUTE_FORMAT,
        dtype=dtype,
        n_channels=NCHANNELS,
        height=HEIGHT,
        width=WIDTH,
        use_xla=use_xla,
        use_tf_amp=use_tf_amp,
        use_dali=use_dali,
        gpu_memory_fraction=gpu_memory_fraction,
        gpu_id=0,
        model_dir=model_dir,
    )

    # removed params not used in inference
    estimator_params = {"use_final_conv": False}  # TODO: Why not moved to model constructor?
    estimator = runner._get_estimator(
        mode="inference",
        run_params=estimator_params,
        use_xla=use_xla,
        use_dali=use_dali,
        gpu_memory_fraction=gpu_memory_fraction,
    )
    return estimator


def get_serving_input_receiver_fn(
    batch_size: int = None,
    input_dtype: str = "fp32",
    width: int = WIDTH,
    height: int = HEIGHT,
    nchannels: int = NCHANNELS,
):
    input_dtype = tf.float16 if input_dtype and "16" in input_dtype else tf.float32
    serving_input_receiver_fn = data_utils.get_serving_input_receiver_fn(
        batch_size=batch_size,
        height=height,
        width=width,
        num_channels=nchannels,
        data_format=INPUT_FORMAT,
        dtype=input_dtype,
    )
    return serving_input_receiver_fn
