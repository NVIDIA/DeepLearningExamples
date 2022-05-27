# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

# Copyright 2019-2022 Ross Wightman
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
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from timm.data import create_dataset, create_loader
from timm.models.helpers import load_checkpoint
from timm.utils import AverageMeter, accuracy

from .gpunet_modules import (
    ConvBnAct,
    EdgeResidual,
    Epilogue,
    EpilogueD,
    Fused_IRB,
    Inverted_Residual_Block,
    InvertedResidual,
    Prologue,
    PrologueD,
    PrologueLargeD,
)


class GPUNet(nn.Module):
    def __init__(self, imgRes):
        super(GPUNet, self).__init__()
        self.imgRes = imgRes
        self.network = nn.Sequential()

    def add_layer(self, name, layer):
        self.network.add_module(name, layer)

    def forward(self, x):
        return self.network(x)


class GPUNet_Builder:
    def config_checker(self, layerConfig):
        assert "layer_type" in layerConfig.keys()
        layerType = layerConfig["layer_type"]
        if layerType == "head":
            assert "num_in_channels" in layerConfig.keys()
            assert "num_out_channels" in layerConfig.keys()
        elif layerType == "tail":
            assert "num_in_channels" in layerConfig.keys()
            assert "num_out_channels" in layerConfig.keys()
            assert "num_classes" in layerConfig.keys()
        elif layerType == "irb":
            assert "num_in_channels" in layerConfig.keys()
            assert "num_out_channels" in layerConfig.keys()
            assert "kernel_size" in layerConfig.keys()
            assert "expansion" in layerConfig.keys()
            assert "stride" in layerConfig.keys()

    def test_model(
        self,
        model: nn.Module = None,
        testBatch: int = 10,
        checkpoint: str = "./pth",
        imgRes: tuple = (3, 224, 224),
        dtype: str = "fp16",
        val_path: str = "/mnt/dldata/",
        crop_pct: float = 0.942,
        is_prunet: bool = False,
    ):

        assert model is not None

        if dtype == "fp16":
            dtype = torch.float16
        elif dtype == "fp32":
            dtype = torch.float32
        else:
            raise NotImplementedError

        errMsg = "checkpoint not found at {}, ".format(checkpoint)
        errMsg += "retrieve with get_config_and_checkpoint_files "
        assert os.path.isfile(checkpoint) is True, errMsg

        if is_prunet:
            model.load_state_dict(torch.load(checkpoint))
        else:
            load_checkpoint(model, checkpoint, use_ema=True)
        model = model.to("cuda", dtype)
        imagenet_val_path = val_path

        dataset = create_dataset(
            root=imagenet_val_path,
            name="",
            split="validation",
            load_bytes=False,
            class_map="",
        )

        criterion = nn.CrossEntropyLoss().cuda()
        data_config = {
            "input_size": (3, imgRes[1], imgRes[2]),
            "interpolation": "bicubic",
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "crop_pct": crop_pct,
        }
        print("data_config:", data_config)
        batch_size = testBatch
        loader = create_loader(
            dataset,
            input_size=data_config["input_size"],
            batch_size=batch_size,
            use_prefetcher=True,
            interpolation=data_config["interpolation"],
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=1,
            crop_pct=data_config["crop_pct"],
            pin_memory=False,
            tf_preprocessing=False,
        )

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.eval()
        input = torch.randn((batch_size,) + tuple(data_config["input_size"])).to(
            "cuda", dtype
        )

        with torch.no_grad():
            # warmup, reduce variability of first batch time
            # especially for comparing torchscript
            model(input)
            end = time.time()
            for batch_idx, (input, target) in enumerate(loader):
                target = target.to("cuda")
                input = input.to("cuda", dtype)
                output = model(input)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1.item(), input.size(0))
                top5.update(acc5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % 10 == 0:
                    print(
                        "Test: [{0:>4d}/{1}]  "
                        "Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                        "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                        "Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  "
                        "Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})".format(
                            batch_idx,
                            len(loader),
                            batch_time=batch_time,
                            rate_avg=input.size(0) / batch_time.avg,
                            loss=losses,
                            top1=top1,
                            top5=top5,
                        )
                    )

        top1a, top5a = top1.avg, top5.avg
        results = OrderedDict(
            top1=round(top1a, 4),
            top1_err=round(100 - top1a, 4),
            top5=round(top5a, 4),
            top5_err=round(100 - top5a, 4),
            img_size=data_config["input_size"][-1],
            interpolation=data_config["interpolation"],
        )
        print(
            " * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})".format(
                results["top1"],
                results["top1_err"],
                results["top5"],
                results["top5_err"],
            )
        )
        return results

    def export_onnx(self, model: GPUNet = None, name: str = "gpunet.onnx"):
        assert model is not None, "please input the model"
        x = torch.rand((1, 3, model.imgRes, model.imgRes))
        torch.onnx.export(model, x, name, export_params=True, opset_version=10)

    def get_model(self, config: list = None):
        msg = "the model json needs specify whether a distilled model or not."
        assert "distill" in config[0].keys(), msg

        if config[0]["distill"]:
            return self._get_distill_model(config)
        else:
            return self._get_model(config)

    def _get_model(self, config: list = None):
        assert len(config) > 0
        dataLayer = config[0]
        assert dataLayer["layer_type"] == "data"
        assert dataLayer["img_resolution"] > 0
        imgRes = dataLayer["img_resolution"]

        net = GPUNet(imgRes)
        dropPathRateBase = 0.2
        layerCount = len(config) - 1
        layerCounter = 0
        for layerConfig in config:
            dropPathRate = dropPathRateBase * layerCounter / layerCount
            layerCounter = layerCounter + 1
            assert "layer_type" in layerConfig.keys()
            self.config_checker(layerConfig)
            layerType = layerConfig["layer_type"]
            if layerType == "head":
                name = "head: " + str(layerCounter)
                layer = Prologue(
                    num_in_channels=layerConfig["num_in_channels"],
                    num_out_channels=layerConfig["num_out_channels"],
                    act_layer=layerConfig.get("act", "swish"),
                )
                net.add_layer(name, layer)
            elif layerType == "tail":
                name = " layer" + str(layerCounter)
                layer = Epilogue(
                    num_in_channels=layerConfig["num_in_channels"],
                    num_out_channels=layerConfig["num_out_channels"],
                    num_classes=layerConfig["num_classes"],
                )
                net.add_layer(name, layer)
            elif layerType == "conv":
                name = "stage: " + str(layerConfig["stage"]) + " layer"
                name += str(layerCounter)
                layer = ConvBnAct(
                    in_chs=layerConfig["num_in_channels"],
                    out_chs=layerConfig["num_out_channels"],
                    kernel_size=layerConfig["kernel_size"],
                    stride=layerConfig["stride"],
                    act_layer=layerConfig["act"],
                    drop_path_rate=dropPathRate,
                )
                net.add_layer(name, layer)
            elif layerType == "irb":
                name = "stage: " + str(layerConfig["stage"]) + " layer"
                name += str(layerCounter)
                layer = InvertedResidual(
                    in_chs=layerConfig["num_in_channels"],
                    out_chs=layerConfig["num_out_channels"],
                    dw_kernel_size=layerConfig["kernel_size"],
                    stride=layerConfig["stride"],
                    exp_ratio=layerConfig["expansion"],
                    use_se=layerConfig["use_se"],
                    act_layer=layerConfig["act"],
                    drop_path_rate=dropPathRate,
                )
                net.add_layer(name, layer)
            elif layerType == "fused_irb":
                name = "stage: " + str(layerConfig["stage"]) + " layer"
                name += str(layerCounter)
                layer = EdgeResidual(
                    in_chs=layerConfig["num_in_channels"],
                    out_chs=layerConfig["num_out_channels"],
                    exp_kernel_size=layerConfig["kernel_size"],
                    stride=layerConfig["stride"],
                    dilation=1,
                    pad_type="same",
                    exp_ratio=layerConfig["expansion"],
                    use_se=layerConfig["use_se"],
                    act_layer=layerConfig["act"],
                    drop_path_rate=dropPathRate,
                )
                net.add_layer(name, layer)
            elif layerType == "data":
                net.imgRes = layerConfig["img_resolution"]
            else:
                raise NotImplementedError
        net.eval()
        return net

    def _get_distill_model(self, config: list = None):
        assert config is not None
        # json -> model
        dataLayer = config[0]
        assert dataLayer["layer_type"] == "data"
        assert dataLayer["img_resolution"] > 0
        imgRes = dataLayer["img_resolution"]
        net = GPUNet(imgRes)

        irbCounter = 0
        for layerConfig in config:
            irbCounter = irbCounter + 1
            assert "layer_type" in layerConfig.keys()
            self.config_checker(layerConfig)
            layerType = layerConfig["layer_type"]
            if layerType == "head":
                name = "head:"
                layer = PrologueD(
                    num_in_channels=layerConfig["num_in_channels"],
                    num_out_channels=layerConfig["num_out_channels"],
                )
                net.add_layer(name, layer)
            elif layerType == "head_large":
                name = "head:"
                layer = PrologueLargeD(
                    num_in_channels=layerConfig["num_in_channels"],
                    num_out_channels=layerConfig["num_out_channels"],
                )
                net.add_layer(name, layer)
            elif layerType == "tail":
                name = "tail:"
                layer = EpilogueD(
                    num_in_channels=layerConfig["num_in_channels"],
                    num_out_channels=layerConfig["num_out_channels"],
                    num_classes=layerConfig["num_classes"],
                )
                net.add_layer(name, layer)
            elif layerType == "irb":
                name = "stage: " + str(layerConfig["stage"]) + " irb"
                name += str(irbCounter)
                layer = Inverted_Residual_Block(
                    num_in_channels=layerConfig["num_in_channels"],
                    num_out_channels=layerConfig["num_out_channels"],
                    kernel_size=layerConfig["kernel_size"],
                    stride=layerConfig["stride"],
                    expansion=layerConfig["expansion"],
                    groups=layerConfig["groups"],
                )
                net.add_layer(name, layer)
            elif layerType == "fused_irb":
                name = "stage: " + str(layerConfig["stage"]) + " fused_irb"
                name += str(irbCounter)
                layer = Fused_IRB(
                    num_in_channels=layerConfig["num_in_channels"],
                    num_out_channels=layerConfig["num_out_channels"],
                    kernel_size=layerConfig["kernel_size"],
                    stride=layerConfig["stride"],
                    expansion=layerConfig["expansion"],
                    groups=layerConfig["groups"],
                )
                net.add_layer(name, layer)
            elif layerType == "data":
                net.imgRes = layerConfig["img_resolution"]
            else:
                raise NotImplementedError
        return net
