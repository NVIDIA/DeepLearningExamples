import itertools

import numpy as np
import tensorflow as tf
from scipy import signal


def get_window_slices(image_size, roi_size, intervals, strategy):
    dim_starts = []
    for image_x, roi_x, interval in zip(image_size, roi_size, intervals):
        starts = list(range(0, image_x - roi_x + 1, interval))
        if strategy == "overlap_inside" and starts[-1] + roi_x < image_x:
            starts.append(image_x - roi_x)
        dim_starts.append(starts)
    slices = [(starts + (0,), roi_size + (-1,)) for starts in itertools.product(*dim_starts)]
    return slices


def batch_window_slices(slices, image_batch_size, batch_size):
    batched_window_slices = []
    for batch_start in range(0, image_batch_size, batch_size):
        batched_window_slices.extend(
            [
                ((batch_start,) + start, (min(batch_size, image_batch_size - batch_start),) + roi_size)
                for start, roi_size in slices
            ]
        )
    return batched_window_slices

@tf.function
def gaussian_kernel_tf_v2(roi_size, sigma):
        """
        adapted from: https://gist.github.com/blzq
        """
        kernel_size = roi_size[0]
        sigma = sigma * kernel_size
        gauss = tf.range(start = 0, limit = kernel_size, dtype = tf.float32) - (kernel_size - 1.0) / 2.0
        xx, yy, zz = tf.meshgrid(gauss, gauss, gauss)
        kernel = tf.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2.0 * sigma ** 2))
        kernel = tf.math.pow(kernel, 1/len(roi_size))
        kernel = kernel / tf.reduce_max(kernel)
        return kernel

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
        return gaussian_kernel_tf_v2(roi_size, sigma=sigma)
    else:
        raise ValueError(f'Invalid blend mode: {blend_mode}. Use either "constant" or "gaussian".')


@tf.function(experimental_relax_shapes=True)
def run_model(model, windows, importance_map, sw_batch_size, **kwargs):
    windows_merged = tf.reshape(windows, shape=(-1, *windows.shape[2:]))
    preds = tf.cast(model(windows_merged, **kwargs), dtype=tf.float32) * importance_map
    return tf.reshape(preds, shape=(sw_batch_size, -1, *preds.shape[1:]))


def sliding_window_inference(
    inputs,
    roi_size,
    sw_batch_size,
    model,
    overlap,
    n_class,
    blend_mode="gaussian",
    sigma=0.125,
    strategy="overlap_inside",
    **kwargs,
):
    """
    Sliding window inference based on implementation by monai library:
    https://docs.monai.io/en/latest/_modules/monai/inferers/utils.html#sliding_window_inference

    Args:
        inputs: tf.Tensor to process; should have batch dimension and be
            in channels-last format, therefore assuming NHWDC or NHWC format.
            Currently batch dimension MUST have size equal to 1 for NHWDC layout.
        roi_size: region-of-interest size i.e. the sliding window shape
        sw_batch_size: batch size for the stacked windows
        overlap: [0.0, 1.0] float, ratio of overlapping windows in one dimension.
            Can be equal to 1, then a stride 1 will be used.
        blend_mode: how to blend overlapping windows. Possible values {"constant", "gaussian"}.
        n_class: number of output channels.
        sigma: standard deviation for the gaussian blending kernel.
        strategy: strategy for dealing with unaligned edge window. Possible values:
            "pad" for padding the input image with zeroes to match the size or
            "overlap_inside" for reducing the length of the last stride.
        kwargs: additional parameters for the model call.

    Returns: Inferred tf.Tensor.
    """

    dim = int(tf.rank(inputs)) - 2
    batch_size = inputs.shape[0]
    assert dim in [2, 3], "Only 2D and 3D data are supported"
    assert dim != 3 or batch_size == 1, "Batch size of the 3D input has to be equal to one"
    assert len(roi_size) == dim, "Dimensionality of ROI size used by sliding window does not match the input dim"

    input_spatial_shape = list(inputs.shape[1:-1])

    roi_size = tuple(roi_size)
    image_size = tuple(max(input_spatial_shape[i], roi_size[i]) for i in range(dim))
    padding_size = [image_x - input_x for image_x, input_x in zip(image_size, input_spatial_shape)]

    intervals = []
    for i in range(dim):
        if roi_size[i] == image_size[i]:
            intervals.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            intervals.append(interval if interval > 0 else 1)

    if strategy == "pad":
        for i, (image_x, roi_x, interval) in enumerate(zip(image_size, roi_size, intervals)):
            if image_x % interval != roi_x % interval:
                padding_size[i] += interval - (image_x - roi_x) % interval
    paddings = [[0, 0]] + [[x // 2, x - x // 2] for x in padding_size] + [[0, 0]]
    input_padded = tf.pad(inputs, paddings)
    image_size = list(input_padded.shape[1:-1])

    importance_kernel = get_importance_kernel(roi_size, blend_mode, sigma=sigma)

    output_shape = (batch_size,) + tuple(image_size) + (n_class,)
    importance_map = tf.tile(
        tf.reshape(importance_kernel, shape=[1, *roi_size, 1]),
        multiples=[sw_batch_size] + [1] * dim + [output_shape[-1]],
    )
    output_sum = tf.zeros(output_shape, dtype=tf.float32)
    output_weight_sum = tf.zeros(output_shape, dtype=tf.float32)

    window_slices = get_window_slices(image_size, roi_size, intervals, strategy)
    if dim == 3:
        window_slices = batch_window_slices(window_slices, batch_size, 1)
    else:
        window_slices = batch_window_slices(window_slices, batch_size, sw_batch_size)
        sw_batch_size = 1

    for window_group_start in range(0, len(window_slices), sw_batch_size):
        slice_group = window_slices[window_group_start : window_group_start + sw_batch_size]
        windows = tf.stack([tf.slice(input_padded, begin=begin, size=size) for begin, size in slice_group])
        importance_map_part = importance_map[: windows.shape[0] * windows.shape[1]]
        preds = run_model(model, windows, importance_map_part, windows.shape[0], **kwargs)
        preds = tf.unstack(preds, axis=0)
        for s, pred in zip(slice_group, preds):
            padding = [[start, output_size - (start + size)] for start, size, output_size in zip(*s, output_shape)]
            padding = padding[:-1] + [[0, 0]]
            output_sum = output_sum + tf.pad(pred, padding)
            output_weight_sum = output_weight_sum + tf.pad(importance_map_part, padding)

    output = output_sum / output_weight_sum
    crop_slice = [slice(pad[0], pad[0] + input_x) for pad, input_x in zip(paddings, inputs.shape[:-1])]
    output_cropped = output[crop_slice]

    return output_cropped


if __name__ == "__main__":
    image_size = [7, 6]
    roi_size = [3, 2]
    intervals = [2, 2]
    assert get_window_slices(image_size, roi_size, intervals) == [
        (slice(0, 3), slice(0, 2)),
        (slice(0, 3), slice(2, 4)),
        (slice(0, 3), slice(4, 6)),
        (slice(2, 5), slice(0, 2)),
        (slice(2, 5), slice(2, 4)),
        (slice(2, 5), slice(4, 6)),
        (slice(4, 7), slice(0, 2)),
        (slice(4, 7), slice(2, 4)),
        (slice(4, 7), slice(4, 6)),
    ]

    # print(gaussian_kernel([4, 5, 6], sigma=0.125))

    import matplotlib.pyplot as plt
    import PIL

    img = np.asarray(PIL.Image.open("images/unet3d.png"))
    inputs = tf.expand_dims(tf.convert_to_tensor(img, dtype=tf.float32), axis=0)
    model = tf.identity
    result = sliding_window_inference(inputs, roi_size=(128, 128), sw_batch_size=1, overlap=0.5, model=model)
    plt.imsave("images/sw_unet3d.png", np.squeeze(result.numpy()))
