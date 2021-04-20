import logging
from pathlib import Path

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


def get_dataloader_fn(
    *, data_dir: str, batch_size: int = 1, width: int = 224, height: int = 224, images_num: int = None,
    precision: str = "fp32", classes: int = 1000
):
    def _dataloader():
        image_extensions = [".gif", ".png", ".jpeg", ".jpg"]

        image_paths = sorted([p for p in Path(data_dir).rglob("*") if p.suffix.lower() in image_extensions])
        if images_num is not None:
            image_paths = image_paths[:images_num]

        LOGGER.info(
            f"Creating PIL dataloader on data_dir={data_dir} #images={len(image_paths)} "
            f"image_size=({width}, {height}) batch_size={batch_size}"
        )

        onehot = np.eye(classes)

        batch = []
        for image_path in image_paths:
            img = Image.open(image_path.as_posix()).convert("RGB")
            img = img.resize((width, height))
            img = (np.array(img).astype(np.float32) / 255) - np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
            img = img / np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

            true_class = np.array([int(image_path.parent.name)])
            assert tuple(img.shape) == (height, width, 3)
            img = img[np.newaxis, ...]
            batch.append((img, image_path.as_posix(), true_class))
            if len(batch) >= batch_size:
                ids = [image_path for _, image_path, *_ in batch]
                x = {"INPUT__0": np.ascontiguousarray(
                                    np.transpose(np.concatenate([img for img, *_ in batch]), 
                                                 (0, 3, 1, 2)).astype(np.float32 if precision == "fp32" else np.float16))}
                y_real = {"OUTPUT__0": onehot[np.concatenate([class_ for *_, class_ in batch])].astype(
                    np.float32 if precision == "fp32" else np.float16                              
                )}
                batch = []
                yield ids, x, y_real
    return _dataloader
