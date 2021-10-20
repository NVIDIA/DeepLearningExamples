# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from apex.optimizers import FusedAdam, FusedSGD
from monai.inferers import sliding_window_inference
from scipy.special import expit, softmax
from skimage.transform import resize
from utils.scheduler import WarmupCosineSchedule
from utils.utils import (
    flip,
    get_dllogger,
    get_path,
    get_test_fnames,
    get_tta_flips,
    get_unet_params,
    is_main_process,
    layout_2d,
)

from models.loss import Loss, LossBraTS
from models.metrics import Dice
from models.unet import UNet


class NNUnet(pl.LightningModule):
    def __init__(self, args, bermuda=False, data_dir=None):
        super(NNUnet, self).__init__()
        self.save_hyperparameters()
        self.args = args
        self.bermuda = bermuda
        if data_dir is not None:
            self.args.data = data_dir
        self.build_nnunet()
        self.best_mean = 0
        self.best_mean_epoch = 0
        self.best_dice = self.n_class * [0]
        self.best_epoch = self.n_class * [0]
        self.best_mean_dice = self.n_class * [0]
        self.test_idx = 0
        self.test_imgs = []
        if not self.bermuda:
            self.learning_rate = args.learning_rate
            loss = LossBraTS if self.args.brats else Loss
            self.loss = loss(self.args.focal)
            self.tta_flips = get_tta_flips(args.dim)
            self.dice = Dice(self.n_class, self.args.brats)
            if self.args.exec_mode in ["train", "evaluate"]:
                self.dllogger = get_dllogger(args.results)

    def forward(self, img):
        return torch.argmax(self.model(img), 1)

    def _forward(self, img):
        if self.args.benchmark:
            if self.args.dim == 2 and self.args.data2d_dim == 3:
                img = layout_2d(img, None)
            return self.model(img)
        return self.tta_inference(img) if self.args.tta else self.do_inference(img)

    def compute_loss(self, preds, label):
        if self.args.deep_supervision:
            pred0, pred1, pred2 = preds
            loss = self.loss(pred0, label) + 0.5 * self.loss(pred1, label) + 0.25 * self.loss(pred2, label)
            return loss / 1.75
        return self.loss(preds, label)

    def training_step(self, batch, batch_idx):
        img, lbl = self.get_train_data(batch)
        pred = self.model(img)
        loss = self.compute_loss(pred, lbl)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.current_epoch < self.args.skip_first_n_eval:
            return None
        img, lbl = batch["image"], batch["label"]
        pred = self._forward(img)
        loss = self.loss(pred, lbl)
        self.dice.update(pred, lbl[:, 0], loss)

    def test_step(self, batch, batch_idx):
        if self.args.exec_mode == "evaluate":
            return self.validation_step(batch, batch_idx)
        img = batch["image"]
        pred = self._forward(img).squeeze(0).cpu().detach().numpy()
        if self.args.save_preds:
            meta = batch["meta"][0].cpu().detach().numpy()
            min_d, max_d = meta[0, 0], meta[1, 0]
            min_h, max_h = meta[0, 1], meta[1, 1]
            min_w, max_w = meta[0, 2], meta[1, 2]
            n_class, original_shape, cropped_shape = pred.shape[0], meta[2], meta[3]
            if not all(cropped_shape == pred.shape[1:]):
                resized_pred = np.zeros((n_class, *cropped_shape))
                for i in range(n_class):
                    resized_pred[i] = resize(
                        pred[i], cropped_shape, order=3, mode="edge", cval=0, clip=True, anti_aliasing=False
                    )
                pred = resized_pred
            final_pred = np.zeros((n_class, *original_shape))
            final_pred[:, min_d:max_d, min_h:max_h, min_w:max_w] = pred
            if self.args.brats:
                final_pred = expit(final_pred)
            else:
                final_pred = softmax(final_pred, axis=0)

            self.save_mask(final_pred)

    def build_nnunet(self):
        in_channels, n_class, kernels, strides, self.patch_size = get_unet_params(self.args)
        self.n_class = n_class - 1
        if self.args.brats:
            n_class = 3
        self.model = UNet(
            in_channels=in_channels,
            n_class=n_class,
            kernels=kernels,
            strides=strides,
            dimension=self.args.dim,
            normalization_layer=self.args.norm,
            negative_slope=self.args.negative_slope,
            deep_supervision=self.args.deep_supervision,
            more_chn=self.args.more_chn,
        )
        if is_main_process():
            print(f"Filters: {self.model.filters},\nKernels: {kernels}\nStrides: {strides}")

    def do_inference(self, image):
        if self.args.dim == 3:
            return self.sliding_window_inference(image)
        if self.args.data2d_dim == 2:
            return self.model(image)
        if self.args.exec_mode == "predict":
            return self.inference2d_test(image)
        return self.inference2d(image)

    def tta_inference(self, img):
        pred = self.do_inference(img)
        for flip_idx in self.tta_flips:
            pred += flip(self.do_inference(flip(img, flip_idx)), flip_idx)
        pred /= len(self.tta_flips) + 1
        return pred

    def inference2d(self, image):
        batch_modulo = image.shape[2] % self.args.val_batch_size
        if batch_modulo != 0:
            batch_pad = self.args.val_batch_size - batch_modulo
            image = nn.ConstantPad3d((0, 0, 0, 0, batch_pad, 0), 0)(image)
        image = torch.transpose(image.squeeze(0), 0, 1)
        preds_shape = (image.shape[0], self.n_class + 1, *image.shape[2:])
        preds = torch.zeros(preds_shape, dtype=image.dtype, device=image.device)
        for start in range(0, image.shape[0] - self.args.val_batch_size + 1, self.args.val_batch_size):
            end = start + self.args.val_batch_size
            pred = self.model(image[start:end])
            preds[start:end] = pred.data
        if batch_modulo != 0:
            preds = preds[batch_pad:]
        return torch.transpose(preds, 0, 1).unsqueeze(0)

    def inference2d_test(self, image):
        preds_shape = (image.shape[0], self.n_class + 1, *image.shape[2:])
        preds = torch.zeros(preds_shape, dtype=image.dtype, device=image.device)
        for depth in range(image.shape[2]):
            preds[:, :, depth] = self.sliding_window_inference(image[:, :, depth])
        return preds

    def sliding_window_inference(self, image):
        return sliding_window_inference(
            inputs=image,
            roi_size=self.patch_size,
            sw_batch_size=self.args.val_batch_size,
            predictor=self.model,
            overlap=self.args.overlap,
            mode=self.args.blend,
        )

    def validation_epoch_end(self, outputs):
        dice, loss = self.dice.compute()
        self.dice.reset()
        dice_mean = torch.mean(dice)
        if dice_mean >= self.best_mean:
            self.best_mean = dice_mean
            self.best_mean_dice = dice[:]
            self.best_mean_epoch = self.current_epoch
        for i, dice_i in enumerate(dice):
            if dice_i > self.best_dice[i]:
                self.best_dice[i], self.best_epoch[i] = dice_i, self.current_epoch

        if is_main_process():
            metrics = {}
            metrics.update({"Mean dice": round(torch.mean(dice).item(), 2)})
            metrics.update({"Highest": round(torch.mean(self.best_mean_dice).item(), 2)})
            if self.n_class > 1:
                metrics.update({f"L{i+1}": round(m.item(), 2) for i, m in enumerate(dice)})
            metrics.update({"val_loss": round(loss.item(), 4)})
            self.dllogger.log(step=self.current_epoch, data=metrics)
            self.dllogger.flush()

        self.log("val_loss", loss)
        self.log("dice_mean", dice_mean)

    def test_epoch_end(self, outputs):
        if self.args.exec_mode == "evaluate":
            self.eval_dice = self.dice.compute()

    def configure_optimizers(self):
        optimizer = {
            "sgd": FusedSGD(self.parameters(), lr=self.learning_rate, momentum=self.args.momentum),
            "adam": FusedAdam(self.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay),
        }[self.args.optimizer.lower()]

        if self.args.scheduler:
            scheduler = {
                "scheduler": WarmupCosineSchedule(
                    optimizer=optimizer,
                    warmup_steps=250,
                    t_total=self.args.epochs * len(self.trainer.datamodule.train_dataloader()),
                ),
                "interval": "step",
                "frequency": 1,
            }
            return {"optimizer": optimizer, "monitor": "val_loss", "lr_scheduler": scheduler}
        return {"optimizer": optimizer, "monitor": "val_loss"}

    def save_mask(self, pred):
        if self.test_idx == 0:
            data_path = get_path(self.args)
            self.test_imgs, _ = get_test_fnames(self.args, data_path)
        fname = os.path.basename(self.test_imgs[self.test_idx]).replace("_x", "")
        np.save(os.path.join(self.save_dir, fname), pred, allow_pickle=False)
        self.test_idx += 1

    def get_train_data(self, batch):
        img, lbl = batch["image"], batch["label"]
        if self.args.dim == 2 and self.args.data2d_dim == 3:
            img, lbl = layout_2d(img, lbl)
        return img, lbl
