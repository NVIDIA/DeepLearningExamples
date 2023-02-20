"""
General Utility functions
"""
import os
import tensorflow as tf
from omegaconf import DictConfig
from .images_utils import image_to_mask_name


def create_directory(path):
    """
    Create Directory if it already does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def join_paths(*paths):
    """
    Concatenate multiple paths.
    """
    return os.path.normpath(os.path.sep.join(path.rstrip(r"\/") for path in paths))


def set_gpus(gpu_ids):
    """
    Change number of visible gpus for tensorflow.
    gpu_ids: Could be integer or list of integers.
    In case Integer: if integer value is -1 then use all available gpus.
    otherwise if positive number, then use given number of gpus.
    In case list of Integer: each integer will be considered as gpu id
    """
    all_gpus = tf.config.experimental.list_physical_devices('GPU')
    all_gpus_length = len(all_gpus)
    if isinstance(gpu_ids, int):
        if gpu_ids == -1:
            gpu_ids = range(all_gpus_length)
        else:
            gpu_ids = min(gpu_ids, all_gpus_length)
            gpu_ids = range(gpu_ids)

    selected_gpus = [all_gpus[gpu_id] for gpu_id in gpu_ids if gpu_id < all_gpus_length]

    try:
        tf.config.experimental.set_visible_devices(selected_gpus, 'GPU')
    except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)


def get_gpus_count():
    """
    Return length of available gpus.
    """
    return len(tf.config.experimental.list_logical_devices('GPU'))


def get_data_paths(cfg: DictConfig, mode: str, mask_available: bool):
    """
    Return list of absolute images/mask paths.
    There are two options you can either pass directory path or list.
    In case of directory, it should contain relative path of images/mask
    folder from project root path.
    In case of list of images, every element should contain absolute path
    for each image and mask.
    For prediction, you can set mask path to None if mask are not
    available for visualization.
    """

    # read images from directory
    if isinstance(cfg.DATASET[mode].IMAGES_PATH, str):
        # has only images name not full path
        images_paths = os.listdir(
            join_paths(
                cfg.WORK_DIR,
                cfg.DATASET[mode].IMAGES_PATH
            )
        )

        if mask_available:
            mask_paths = [
                image_to_mask_name(image_name) for image_name in images_paths
            ]
            # create full mask paths from folder
            mask_paths = [
                join_paths(
                    cfg.WORK_DIR,
                    cfg.DATASET[mode].MASK_PATH,
                    mask_name
                ) for mask_name in mask_paths
            ]

        # create full images paths from folder
        images_paths = [
            join_paths(
                cfg.WORK_DIR,
                cfg.DATASET[mode].IMAGES_PATH,
                image_name
            ) for image_name in images_paths
        ]
    else:
        # read images and mask from absolute paths given in list
        images_paths = list(cfg.DATASET[mode].IMAGES_PATH)
        if mask_available:
            mask_paths = list(cfg.DATASET[mode].MASK_PATH)

    if mask_available:
        return images_paths, mask_paths
    else:
        return images_paths,


def suppress_warnings():
    """
    Suppress TensorFlow warnings.
    """
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('dali').setLevel(logging.ERROR)
    os.environ["KMP_AFFINITY"] = "noverbose"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.autograph.set_verbosity(3)
