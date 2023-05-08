"""
Data generator
"""
import os
import tensorflow as tf
from omegaconf import DictConfig

from utils.general_utils import join_paths, get_gpus_count
from .tf_data_generator import DataGenerator as tf_data_generator

try:
    from .dali_data_generator import data_generator as dali_data_generator
except ModuleNotFoundError:
    print("NVIDIA DALI not installed, please install it."
          "\nNote: DALI is only available on Linux platform. For Window "
          "you can use TensorFlow generator for training.")


def get_data_generator(cfg: DictConfig,
                       mode: str,
                       strategy: tf.distribute.Strategy = None):
    """
    Creates and return data generator object based on given type.
    """
    if cfg.DATA_GENERATOR_TYPE == "TF_GENERATOR":
        print(f"Using TensorFlow generator for {mode} data")
        generator = tf_data_generator(cfg, mode)
    elif cfg.DATA_GENERATOR_TYPE == "DALI_GENERATOR":
        print(f"Using NVIDIA DALI generator for {mode} data")
        if cfg.USE_MULTI_GPUS.VALUE:
            generator = dali_data_generator(cfg, mode, strategy)
        else:
            generator = dali_data_generator(cfg, mode)
    else:
        raise ValueError(
            "Wrong generator type passed."
            "\nPossible options are TF_GENERATOR and DALI_GENERATOR"
        )
    return generator


def update_batch_size(cfg: DictConfig):
    """
    Scale up batch size to multi gpus in case of TensorFlow generator.
    """
    if cfg.DATA_GENERATOR_TYPE == "TF_GENERATOR" and cfg.USE_MULTI_GPUS.VALUE:
        # change batch size according to available gpus
        cfg.HYPER_PARAMETERS.BATCH_SIZE = \
            cfg.HYPER_PARAMETERS.BATCH_SIZE * get_gpus_count()


def get_batch_size(cfg: DictConfig):
    """
    Return batch size.
    In case of DALI generator scale up batch size to multi gpus.
    """
    if cfg.DATA_GENERATOR_TYPE == "DALI_GENERATOR" and cfg.USE_MULTI_GPUS.VALUE:
        # change batch size according to available gpus
        return cfg.HYPER_PARAMETERS.BATCH_SIZE * get_gpus_count()
    else:
        return cfg.HYPER_PARAMETERS.BATCH_SIZE


def get_iterations(cfg: DictConfig, mode: str):
    """
    Return steps per epoch
    """
    images_length = len(
        os.listdir(
            join_paths(
                cfg.WORK_DIR,
                cfg.DATASET[mode].IMAGES_PATH
            )
        )
    )

    if cfg.DATA_GENERATOR_TYPE == "TF_GENERATOR":
        training_steps = images_length // cfg.HYPER_PARAMETERS.BATCH_SIZE
    elif cfg.DATA_GENERATOR_TYPE == "DALI_GENERATOR":
        if cfg.USE_MULTI_GPUS.VALUE:
            training_steps = images_length // (
                    cfg.HYPER_PARAMETERS.BATCH_SIZE * get_gpus_count())
        else:
            training_steps = images_length // cfg.HYPER_PARAMETERS.BATCH_SIZE
    else:
        raise ValueError("Wrong generator type passed.")

    return training_steps
