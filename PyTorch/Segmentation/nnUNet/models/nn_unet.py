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
import torch_optimizer as optim
from dllogger import JSONStreamBackend, Logger, StdOutBackend, Verbosity
from monai.inferers import sliding_window_inference
from utils.utils import flip, get_config_file, is_main_process

from models.metrics import Dice, Loss
from models.unet import UNet


class NNUnet(pl.LightningModule):
    def __init__(self, args):
        super(NNUnet, self).__init__()
        self.args = args
        self.save_hyperparameters()
        self.build_nnunet()
        self.loss = Loss()
        self.dice = Dice(self.n_class)
        self.best_sum = 0
        self.eval_dice = 0
        self.best_sum_epoch = 0
        self.best_dice = self.n_class * [0]
        self.best_epoch = self.n_class * [0]
        self.best_sum_dice = self.n_class * [0]
        self.learning_rate = args.learning_rate
        if self.args.exec_mode in ["train", "evaluate"]:
            self.dllogger = Logger(
                backends=[
                    JSONStreamBackend(Verbosity.VERBOSE, os.path.join(args.results, "logs.json")),
                    StdOutBackend(Verbosity.VERBOSE, step_format=lambda step: f"Epoch: {step} "),
                ]
            )

        self.tta_flips = (
            [[2], [3], [2, 3]] if self.args.dim == 2 else [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
        )

    def forward(self, img):
        if self.args.benchmark:
            return self.model(img)
        return self.tta_inference(img) if self.args.tta else self.do_inference(img)

    def training_step(self, batch, batch_idx):
        img, lbl = batch["image"], batch["label"]
        if self.args.dim == 2 and len(lbl.shape) == 3:
            lbl = lbl.unsqueeze(1)
        pred = self.model(img)
        loss = self.compute_loss(pred, lbl)
        return loss

    def validation_step(self, batch, batch_idx):
        img, lbl = batch["image"], batch["label"]
        if self.args.dim == 2 and len(lbl.shape) == 3:
            lbl = lbl.unsqueeze(1)
        pred = self.forward(img)
        loss = self.loss(pred, lbl)
        dice = self.dice(pred, lbl[:, 0])
        return {"val_loss": loss, "val_dice": dice}

    def test_step(self, batch, batch_idx):
        if self.args.exec_mode == "evaluate":
            return self.validation_step(batch, batch_idx)
        img = batch["image"]
        pred = self.forward(img)
        if self.args.save_preds:
            self.save_mask(pred, batch["fname"])

    def build_unet(self, in_channels, n_class, kernels, strides):
        return UNet(
            in_channels=in_channels,
            n_class=n_class,
            kernels=kernels,
            strides=strides,
            normalization_layer=self.args.norm,
            negative_slope=self.args.negative_slope,
            deep_supervision=self.args.deep_supervision,
            dimension=self.args.dim,
        )

    def get_unet_params(self):
        config = get_config_file(self.args)
        in_channels = config["in_channels"]
        patch_size = config["patch_size"]
        spacings = config["spacings"]
        n_class = config["n_class"]

        strides, kernels, sizes = [], [], patch_size[:]
        while True:
            spacing_ratio = [spacing / min(spacings) for spacing in spacings]
            stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
            kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
            if all(s == 1 for s in stride):
                break
            sizes = [i / j for i, j in zip(sizes, stride)]
            spacings = [i * j for i, j in zip(spacings, stride)]
            kernels.append(kernel)
            strides.append(stride)
            if len(strides) == 5:
                break
        strides.insert(0, len(spacings) * [1])
        kernels.append(len(spacings) * [3])

        return in_channels, n_class, kernels, strides, patch_size

    def build_nnunet(self):
        in_channels, n_class, kernels, strides, self.patch_size = self.get_unet_params()
        self.model = self.build_unet(in_channels, n_class, kernels, strides)
        self.n_class = n_class - 1
        if is_main_process():
            print(f"Filters: {self.model.filters}")
            print(f"Kernels: {kernels}")
            print(f"Strides: {strides}")

    def compute_loss(self, preds, label):
        if self.args.deep_supervision:
            loss = self.loss(preds[0], label)
            for i, pred in enumerate(preds[1:]):
                downsampled_label = nn.functional.interpolate(label, pred.shape[2:])
                loss += 0.5 ** (i + 1) * self.loss(pred, downsampled_label)
            c_norm = 1 / (2 - 2 ** (-len(preds)))
            return c_norm * loss
        return self.loss(preds, label)

    def do_inference(self, image):
        if self.args.dim == 2:
            if self.args.data2d_dim == 2:
                return self.model(image)
            if self.args.exec_mode == "predict" and not self.args.benchmark:
                return self.inference2d_test(image)
            return self.inference2d(image)

        return self.sliding_window_inference(image)

    def tta_inference(self, img):
        pred = self.do_inference(img)
        for flip_idx in self.tta_flips:
            pred += flip(self.do_inference(flip(img, flip_idx)), flip_idx)
        pred /= len(self.tta_flips) + 1
        return pred

    def inference2d(self, image):
        batch_modulo = image.shape[2] % self.args.val_batch_size
        if self.args.benchmark:
            image = image[:, :, batch_modulo:]
        elif batch_modulo != 0:
            batch_pad = self.args.val_batch_size - batch_modulo
            image = nn.ConstantPad3d((0, 0, 0, 0, batch_pad, 0), 0)(image)

        image = torch.transpose(image.squeeze(0), 0, 1)
        preds_shape = (image.shape[0], self.n_class + 1, *image.shape[2:])
        preds = torch.zeros(preds_shape, dtype=image.dtype, device=image.device)
        for start in range(0, image.shape[0] - self.args.val_batch_size + 1, self.args.val_batch_size):
            end = start + self.args.val_batch_size
            pred = self.model(image[start:end])
            preds[start:end] = pred.data

        if batch_modulo != 0 and not self.args.benchmark:
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
            mode=self.args.val_mode,
        )

    @staticmethod
    def metric_mean(name, outputs):
        return torch.stack([out[name] for out in outputs]).mean(dim=0)

    def validation_epoch_end(self, outputs):
        loss = self.metric_mean("val_loss", outputs)
        dice = 100 * self.metric_mean("val_dice", outputs)
        dice_sum = torch.sum(dice)
        if dice_sum >= self.best_sum:
            self.best_sum = dice_sum
            self.best_sum_dice = dice[:]
            self.best_sum_epoch = self.current_epoch
        for i, dice_i in enumerate(dice):
            if dice_i > self.best_dice[i]:
                self.best_dice[i], self.best_epoch[i] = dice_i, self.current_epoch

        if is_main_process():
            metrics = {}
            metrics.update({"mean dice": round(torch.mean(dice).item(), 2)})
            metrics.update({"TOP_mean": round(torch.mean(self.best_sum_dice).item(), 2)})
            metrics.update({f"L{i+1}": round(m.item(), 2) for i, m in enumerate(dice)})
            metrics.update({f"TOP_L{i+1}": round(m.item(), 2) for i, m in enumerate(self.best_sum_dice)})
            metrics.update({"val_loss": round(loss.item(), 4)})
            self.dllogger.log(step=self.current_epoch, data=metrics)
            self.dllogger.flush()

        self.log("val_loss", loss)
        self.log("dice_sum", dice_sum)

    def test_epoch_end(self, outputs):
        if self.args.exec_mode == "evaluate":
            self.eval_dice = 100 * self.metric_mean("val_dice", outputs)

    def configure_optimizers(self):
        optimizer = {
            "sgd": torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.args.momentum),
            "adam": torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay),
            "adamw": torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay),
            "radam": optim.RAdam(self.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay),
        }[self.args.optimizer.lower()]

        scheduler = {
            "none": None,
            "multistep": torch.optim.lr_scheduler.MultiStepLR(optimizer, self.args.steps, gamma=self.args.factor),
            "cosine": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_epochs),
            "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=self.args.factor, patience=self.args.lr_patience
            ),
        }[self.args.scheduler.lower()]

        opt_dict = {"optimizer": optimizer, "monitor": "val_loss"}
        if scheduler is not None:
            opt_dict.update({"lr_scheduler": scheduler})
        return opt_dict

    def save_mask(self, pred, fname):
        fname = str(fname[0].cpu().detach().numpy(), "utf-8").replace("_x", "_pred")
        pred = nn.functional.softmax(torch.tensor(pred), dim=1)
        pred = pred.squeeze().cpu().detach().numpy()
        np.save(os.path.join(self.save_dir, fname), pred, allow_pickle=False)
