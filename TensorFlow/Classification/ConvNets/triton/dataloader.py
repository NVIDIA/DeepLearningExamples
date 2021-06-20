import logging
from pathlib import Path

import numpy as np
from PIL import Image

from rn50_model import HEIGHT, WIDTH

LOGGER = logging.getLogger(__name__)


def get_dataloader_fn(
    *, data_dir: str, batch_size: int = 1, width: int = WIDTH, height: int = HEIGHT, images_num: int = None
):
    image_extensions = [".gif", ".png", ".jpeg", ".jpg"]

    image_paths = sorted([p for p in Path(data_dir).rglob("*") if p.suffix.lower() in image_extensions])
    if images_num is not None:
        image_paths = image_paths[:images_num]

    LOGGER.info(
        f"Creating PIL dataloader on data_dir={data_dir} #images={len(image_paths)} "
        f"image_size=({width}, {height}) batch_size={batch_size}"
    )

    def _dataloader_fn():
        batch = []
        for image_path in image_paths:
            img = Image.open(image_path.as_posix()).convert('RGB')
            img = img.resize((width, height))
            img = np.array(img).astype(np.float32)
            true_class = np.array([int(image_path.parent.name)])
            assert tuple(img.shape) == (height, width, 3)
            img = img[np.newaxis, ...]
            batch.append((img, image_path.as_posix(), true_class))
            if len(batch) >= batch_size:
                ids = [image_path for _, image_path, *_ in batch]
                x = {
                    "input": np.concatenate([img for img, *_ in batch]),
                }
                y_real = {"classes": np.concatenate([class_ for *_, class_ in batch])}
                batch = []
                yield ids, x, y_real

    return _dataloader_fn
