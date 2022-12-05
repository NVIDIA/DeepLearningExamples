import itertools

import numpy as np
import tensorflow as tf
from scipy import signal


def get_window_slices(image_size, roi_size, overlap, strategy):
    dim_starts = []
    for image_x, roi_x in zip(image_size, roi_size):
        interval = roi_x if roi_x == image_x else int(roi_x * (1 - overlap))
        starts = list(range(0, image_x - roi_x + 1, interval))
        if strategy == "overlap_inside" and starts[-1] + roi_x < image_x:
            starts.append(image_x - roi_x)
        dim_starts.append(starts)
    slices = [(starts + (0,), roi_size + (-1,)) for starts in itertools.product(*dim_starts)]
    batched_window_slices = [((0,) + start, (1,) + roi_size) for start, roi_size in slices]
    return batched_window_slices


@tf.function
def gaussian_kernel(roi_size, sigma):
    gauss = signal.windows.gaussian(roi_size[0], std=sigma * roi_size[0])
    for s in roi_size[1:]:
        gauss = np.outer(gauss, signal.windows.gaussian(s, std=sigma * s))

    gauss = np.reshape(gauss, roi_size)
    gauss = np.power(gauss, 1 / len(roi_size))
    gauss /= gauss.max()

    return tf.convert_to_tensor(gauss, dtype=tf.float32)


def get_importance_kernel(roi_size, blend_mode, sigma):
    if blend_mode == "constant":
        return tf.ones(roi_size, dtype=tf.float32)
    elif blend_mode == "gaussian":
        return gaussian_kernel(roi_size, sigma)
    else:
        raise ValueError(f'Invalid blend mode: {blend_mode}. Use either "constant" or "gaussian".')


@tf.function
def run_model(x, model, importance_map, **kwargs):
    return tf.cast(model(x, **kwargs), dtype=tf.float32) * importance_map


def sliding_window_inference(
    inputs,
    roi_size,
    model,
    overlap,
    n_class,
    importance_map,
    strategy="overlap_inside",
    **kwargs,
):
    image_size = tuple(inputs.shape[1:-1])
    roi_size = tuple(roi_size)
    # Padding to make sure that the image size is at least roi size
    padded_image_size = tuple(max(image_size[i], roi_size[i]) for i in range(3))
    padding_size = [image_x - input_x for image_x, input_x in zip(image_size, padded_image_size)]
    paddings = [[0, 0]] + [[x // 2, x - x // 2] for x in padding_size] + [[0, 0]]
    input_padded = tf.pad(inputs, paddings)

    output_shape = (1, *padded_image_size, n_class)
    output_sum = tf.zeros(output_shape, dtype=tf.float32)
    output_weight_sum = tf.ones(output_shape, dtype=tf.float32)
    window_slices = get_window_slices(padded_image_size, roi_size, overlap, strategy)

    for window_slice in window_slices:
        window = tf.slice(input_padded, begin=window_slice[0], size=window_slice[1])
        pred = run_model(window, model, importance_map, **kwargs)
        padding = [
            [start, output_size - (start + size)] for start, size, output_size in zip(*window_slice, output_shape)
        ]
        padding = padding[:-1] + [[0, 0]]
        output_sum = output_sum + tf.pad(pred, padding)
        output_weight_sum = output_weight_sum + tf.pad(importance_map, padding)

    output = output_sum / output_weight_sum
    crop_slice = [slice(pad[0], pad[0] + input_x) for pad, input_x in zip(paddings, inputs.shape[:-1])]

    return output[crop_slice]
