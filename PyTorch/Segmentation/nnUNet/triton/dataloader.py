import numpy as np
from data_loading.dali_loader import fetch_dali_loader
from sklearn.model_selection import KFold
from utils.utils import get_split, load_data


def get_dataloader_fn(*, data_dir: str, batch_size: int, precision: str):
    kwargs = {
        "dim": 3,
        "gpus": 1,
        "seed": 0,
        "num_workers": 8,
        "meta": None,
        "oversampling": 0,
        "benchmark": False,
        "patch_size": [128, 128, 128],
    }

    imgs, lbls = load_data(data_dir, "*_x.npy"), load_data(data_dir, "*_y.npy")
    kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
    _, val_idx = list(kfold.split(imgs))[2]
    imgs, lbls = get_split(imgs, val_idx), get_split(lbls, val_idx)
    dataloader = fetch_dali_loader(imgs, lbls, batch_size, "bermuda", **kwargs)

    def _dataloader_fn():
        for i, batch in enumerate(dataloader):
            fname = [f"{i}_{j}" for j in range(batch_size)]
            img = batch["image"].numpy()
            if "fp16" in precision:
                img = img.astype(np.half)
            img = {"INPUT__0": img}
            lbl = {"OUTPUT__0": batch["label"].squeeze(1).numpy().astype(int)}
            yield fname, img, lbl

    return _dataloader_fn
