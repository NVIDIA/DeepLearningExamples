"""
Tensorflow data generator class.
"""
import tensorflow as tf
import numpy as np
from omegaconf import DictConfig

from utils.general_utils import get_data_paths
from utils.images_utils import prepare_image, prepare_mask


class DataGenerator(tf.keras.utils.Sequence):
    """
    Generate batches of data for model by reading images and their
    corresponding masks using TensorFlow Sequence Generator.
    There are two options you can either pass directory path or list.
    In case of directory, it should contain relative path of images/mask
    folder from project root path.
    In case of list of images, every element should contain absolute path
    for each image and mask.
    Because this generator is also used for prediction, so during testing you can
    set mask path to None if mask are not available for visualization.
    """

    def __init__(self, cfg: DictConfig, mode: str):
        """
        Initialization
        """
        self.cfg = cfg
        self.mode = mode
        self.batch_size = self.cfg.HYPER_PARAMETERS.BATCH_SIZE
        # set seed for reproducibility
        np.random.seed(cfg.SEED)

        # check mask are available or not
        self.mask_available = False if cfg.DATASET[mode].MASK_PATH is None or str(
            cfg.DATASET[mode].MASK_PATH).lower() == "none" else True

        data_paths = get_data_paths(cfg, mode, self.mask_available)

        self.images_paths = data_paths[0]
        if self.mask_available:
            self.mask_paths = data_paths[1]

        # self.images_paths.sort()  # no need for sorting

        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        # Tensorflow problem: on_epoch_end is not being called at the end
        # of each epoch, so forcing on_epoch_end call
        self.on_epoch_end()
        return int(
            np.floor(
                len(self.images_paths) / self.batch_size
            )
        )

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.images_paths))
        if self.cfg.PREPROCESS_DATA.SHUFFLE[self.mode].VALUE:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[
                  index * self.batch_size:(index + 1) * self.batch_size
                  ]

        # Generate data
        return self.__data_generation(indexes)

    def __data_generation(self, indexes):
        """
        Generates batch data
        """

        # create empty array to store batch data
        batch_images = np.zeros(
            (
                self.cfg.HYPER_PARAMETERS.BATCH_SIZE,
                self.cfg.INPUT.HEIGHT,
                self.cfg.INPUT.WIDTH,
                self.cfg.INPUT.CHANNELS
            )
        ).astype(np.float32)

        if self.mask_available:
            batch_masks = np.zeros(
                (
                    self.cfg.HYPER_PARAMETERS.BATCH_SIZE,
                    self.cfg.INPUT.HEIGHT,
                    self.cfg.INPUT.WIDTH,
                    self.cfg.OUTPUT.CLASSES
                )
            ).astype(np.float32)

        for i, index in enumerate(indexes):
            # extract path from list
            img_path = self.images_paths[int(index)]
            if self.mask_available:
                mask_path = self.mask_paths[int(index)]

            # prepare image for model by resizing and preprocessing it
            image = prepare_image(
                img_path,
                self.cfg.PREPROCESS_DATA.RESIZE,
                self.cfg.PREPROCESS_DATA.IMAGE_PREPROCESSING_TYPE,
            )

            if self.mask_available:
                # prepare image for model by resizing and preprocessing it
                mask = prepare_mask(
                    mask_path,
                    self.cfg.PREPROCESS_DATA.RESIZE,
                    self.cfg.PREPROCESS_DATA.NORMALIZE_MASK,
                )

            # numpy to tensorflow conversion
            if self.mask_available:
                image, mask = tf.numpy_function(
                    self.tf_func,
                    [image, mask],
                    [tf.float32, tf.int32]
                )
            else:
                image = tf.numpy_function(
                    self.tf_func,
                    [image, ],
                    [tf.float32, ]
                )

            # set shape attributes which was lost during Tf conversion
            image.set_shape(
                [
                    self.cfg.INPUT.HEIGHT,
                    self.cfg.INPUT.WIDTH,
                    self.cfg.INPUT.CHANNELS
                ]
            )
            batch_images[i] = image

            if self.mask_available:
                # height x width --> height x width x output classes
                if self.cfg.OUTPUT.CLASSES == 1:
                    mask = tf.expand_dims(mask, axis=-1)
                else:
                    # convert mask into one hot vectors
                    mask = tf.one_hot(
                        mask,
                        self.cfg.OUTPUT.CLASSES,
                        dtype=tf.int32
                    )
                mask.set_shape(
                    [
                        self.cfg.INPUT.HEIGHT,
                        self.cfg.INPUT.WIDTH,
                        self.cfg.OUTPUT.CLASSES
                    ]
                )
                batch_masks[i] = mask

        if self.mask_available:
            return batch_images, batch_masks
        else:
            return batch_images,

    @staticmethod
    def tf_func(*args):
        return args
