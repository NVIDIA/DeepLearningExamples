"""
NVIDIA DALI data generator object.
"""
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf
import tensorflow as tf
from omegaconf import DictConfig

from utils.general_utils import get_data_paths, get_gpus_count


def data_generator_pipeline(cfg: DictConfig, mode: str, mask_available: bool):
    """
    Returns DALI data pipeline object.
    """
    data_paths = get_data_paths(cfg, mode, mask_available)  # get data paths
    images_paths = data_paths[0]
    if mask_available:
        mask_paths = data_paths[1]

    @pipeline_def(batch_size=cfg.HYPER_PARAMETERS.BATCH_SIZE)
    def single_gpu_pipeline(device):
        """
        Returns DALI data pipeline object for single GPU training.
        """
        device = 'mixed' if 'gpu' in device.lower() else 'cpu'

        pngs, _ = fn.readers.file(
            files=images_paths,
            random_shuffle=cfg.PREPROCESS_DATA.SHUFFLE[mode].VALUE,
            seed=cfg.SEED
        )
        images = fn.decoders.image(pngs, device=device, output_type=types.RGB)
        if cfg.PREPROCESS_DATA.RESIZE.VALUE:
            # TODO verify image resizing method
            images = fn.resize(
                images,
                size=[
                    cfg.PREPROCESS_DATA.RESIZE.HEIGHT,
                    cfg.PREPROCESS_DATA.RESIZE.WIDTH
                ]
            )
        if cfg.PREPROCESS_DATA.IMAGE_PREPROCESSING_TYPE == "normalize":
            images = fn.normalize(images, mean=0, stddev=255, )  # axes=(2,)

        if mask_available:
            labels, _ = fn.readers.file(
                files=mask_paths,
                random_shuffle=cfg.PREPROCESS_DATA.SHUFFLE[mode].VALUE,
                seed=cfg.SEED
            )
            labels = fn.decoders.image(
                labels,
                device=device,
                output_type=types.GRAY
            )
            if cfg.PREPROCESS_DATA.RESIZE.VALUE:
                # TODO verify image resizing method
                labels = fn.resize(
                    labels,
                    size=[
                        cfg.PREPROCESS_DATA.RESIZE.HEIGHT,
                        cfg.PREPROCESS_DATA.RESIZE.WIDTH
                    ]
                )
            if cfg.PREPROCESS_DATA.NORMALIZE_MASK.VALUE:
                labels = fn.normalize(
                    labels,
                    mean=0,
                    stddev=cfg.PREPROCESS_DATA.NORMALIZE_MASK.NORMALIZE_VALUE,
                )
            if cfg.OUTPUT.CLASSES == 1:
                labels = fn.cast(labels, dtype=types.FLOAT)
            else:
                labels = fn.squeeze(labels, axes=[2])
                labels = fn.one_hot(labels, num_classes=cfg.OUTPUT.CLASSES)

        if mask_available:
            return images, labels
        else:
            return images,

    @pipeline_def(batch_size=cfg.HYPER_PARAMETERS.BATCH_SIZE)
    def multi_gpu_pipeline(device, shard_id):
        """
        Returns DALI data pipeline object for multi GPU'S training.
        """
        device = 'mixed' if 'gpu' in device.lower() else 'cpu'
        shard_id = 1 if 'cpu' in device else shard_id
        num_shards = get_gpus_count()
        # num_shards should be <= #images
        num_shards = len(images_paths) if num_shards > len(images_paths) else num_shards

        pngs, _ = fn.readers.file(
            files=images_paths,
            random_shuffle=cfg.PREPROCESS_DATA.SHUFFLE[mode].VALUE,
            shard_id=shard_id,
            num_shards=num_shards,
            seed=cfg.SEED
        )
        images = fn.decoders.image(pngs, device=device, output_type=types.RGB)
        if cfg.PREPROCESS_DATA.RESIZE.VALUE:
            # TODO verify image resizing method
            images = fn.resize(
                images,
                size=[
                    cfg.PREPROCESS_DATA.RESIZE.HEIGHT,
                    cfg.PREPROCESS_DATA.RESIZE.WIDTH
                ]
            )
        if cfg.PREPROCESS_DATA.IMAGE_PREPROCESSING_TYPE == "normalize":
            images = fn.normalize(images, mean=0, stddev=255, )  # axes=(2,)

        if mask_available:
            labels, _ = fn.readers.file(
                files=mask_paths,
                random_shuffle=cfg.PREPROCESS_DATA.SHUFFLE[mode].VALUE,
                shard_id=shard_id,
                num_shards=num_shards,
                seed=cfg.SEED
            )
            labels = fn.decoders.image(
                labels,
                device=device,
                output_type=types.GRAY
            )
            if cfg.PREPROCESS_DATA.RESIZE.VALUE:
                # TODO verify image resizing method
                labels = fn.resize(
                    labels,
                    size=[
                        cfg.PREPROCESS_DATA.RESIZE.HEIGHT,
                        cfg.PREPROCESS_DATA.RESIZE.WIDTH
                    ]
                )
            if cfg.PREPROCESS_DATA.NORMALIZE_MASK.VALUE:
                labels = fn.normalize(
                    labels,
                    mean=0,
                    stddev=cfg.PREPROCESS_DATA.NORMALIZE_MASK.NORMALIZE_VALUE,
                )
            if cfg.OUTPUT.CLASSES == 1:
                labels = fn.cast(labels, dtype=types.FLOAT)
            else:
                labels = fn.squeeze(labels, axes=[2])
                labels = fn.one_hot(labels, num_classes=cfg.OUTPUT.CLASSES)

        if mask_available:
            return images, labels
        else:
            return images,

    if cfg.USE_MULTI_GPUS.VALUE:
        return multi_gpu_pipeline
    else:
        return single_gpu_pipeline


def get_data_shapes(cfg: DictConfig, mask_available: bool):
    """
    Returns shapes and dtypes of the outputs.
    """
    if mask_available:
        shapes = (
            (cfg.HYPER_PARAMETERS.BATCH_SIZE,
             cfg.INPUT.HEIGHT,
             cfg.INPUT.WIDTH,
             cfg.INPUT.CHANNELS),
            (cfg.HYPER_PARAMETERS.BATCH_SIZE,
             cfg.INPUT.HEIGHT,
             cfg.INPUT.WIDTH,
             cfg.OUTPUT.CLASSES)
        )
        dtypes = (
            tf.float32,
            tf.float32)
    else:
        shapes = (
            (cfg.HYPER_PARAMETERS.BATCH_SIZE,
             cfg.INPUT.HEIGHT,
             cfg.INPUT.WIDTH,
             cfg.INPUT.CHANNELS),
        )
        dtypes = (
            tf.float32,
        )
    return shapes, dtypes


def data_generator(cfg: DictConfig,
                   mode: str,
                   strategy: tf.distribute.Strategy = None):
    """
    Generate batches of data for model by reading images and their
    corresponding masks using NVIDIA DALI.
    Works for both single and mult GPU's. In case of multi gpu pass
    the strategy object too.
    There are two options you can either pass directory path or list.
    In case of directory, it should contain relative path of images/mask
    folder from project root path.
    In case of list of images, every element should contain absolute path
    for each image and mask.
    """

    # check mask are available or not
    mask_available = False if cfg.DATASET[mode].MASK_PATH is None or str(
        cfg.DATASET[mode].MASK_PATH).lower() == "none" else True

    # create dali data pipeline
    data_pipeline = data_generator_pipeline(cfg, mode, mask_available)

    shapes, dtypes = get_data_shapes(cfg, mask_available)

    if cfg.USE_MULTI_GPUS.VALUE:
        def bound_dataset(input_context):
            """
            In case of multi gpu training bound dataset to a device for distributed training.
            """
            with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
                device_id = input_context.input_pipeline_id
                return dali_tf.DALIDataset(
                    pipeline=data_pipeline(
                        device="gpu",
                        device_id=device_id,
                        shard_id=device_id,
                        num_threads=cfg.DATALOADER_WORKERS
                    ),
                    batch_size=cfg.HYPER_PARAMETERS.BATCH_SIZE,
                    output_shapes=shapes,
                    output_dtypes=dtypes,
                    device_id=device_id,
                )

        # distribute dataset
        input_options = tf.distribute.InputOptions(
            experimental_place_dataset_on_device=True,
            # for older dali versions use experimental_prefetch_to_device
            # for new dali versions use  experimental_fetch_to_device
            experimental_fetch_to_device=False,  # experimental_fetch_to_device
            experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA)

        # map dataset to given strategy and return it
        return strategy.distribute_datasets_from_function(bound_dataset, input_options)
    else:
        # single gpu pipeline
        pipeline = data_pipeline(
            batch_size=cfg.HYPER_PARAMETERS.BATCH_SIZE,
            num_threads=cfg.DATALOADER_WORKERS,
            device="gpu",
            device_id=0
        )

        # create dataset
        with tf.device('/gpu:0'):
            data_generator = dali_tf.DALIDataset(
                pipeline=pipeline,
                batch_size=cfg.HYPER_PARAMETERS.BATCH_SIZE,
                output_shapes=shapes,
                output_dtypes=dtypes,
                device_id=0)

        return data_generator
