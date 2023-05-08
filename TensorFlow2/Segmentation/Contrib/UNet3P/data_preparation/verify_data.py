"""
Verify for each image corresponding mask exist or not.
Check against both train and val data
"""
import os
import sys
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.append(os.path.abspath("./"))
from utils.general_utils import join_paths
from utils.images_utils import image_to_mask_name


def check_image_and_mask(cfg, mode):
    """
    Check and print names of those images whose mask are not found.
    """
    images_path = join_paths(
        cfg.WORK_DIR,
        cfg.DATASET[mode].IMAGES_PATH
    )
    mask_path = join_paths(
        cfg.WORK_DIR,
        cfg.DATASET[mode].MASK_PATH
    )

    all_images = os.listdir(images_path)

    both_found = True
    for image in tqdm(all_images):
        mask_name = image_to_mask_name(image)
        if not (
                os.path.exists(
                    join_paths(images_path, image)
                ) and
                os.path.exists(
                    join_paths(mask_path, mask_name)
                )
        ):
            print(f"{mask_name} did not found against {image}")
            both_found = False

    return both_found


def verify_data(cfg: DictConfig):
    """
    For both train and val data, check for each image its
    corresponding mask exist or not. If not then stop the program.
    """
    assert check_image_and_mask(cfg, "TRAIN"), \
        "Train images and mask should be same in length"

    assert check_image_and_mask(cfg, "VAL"), \
        "Validation images and mask should be same in length"
