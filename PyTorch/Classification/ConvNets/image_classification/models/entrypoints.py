# Copyright (c) 2018-2019, NVIDIA CORPORATION
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

def nvidia_efficientnet(type='efficient-b0', pretrained=True, **kwargs):
    """Constructs a EfficientNet model.
    For detailed information on model input and output, training recipies, inference and performance
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com
    Args:
        pretrained (bool, True): If True, returns a model pretrained on IMAGENET dataset.
    """

    from .efficientnet import _ce

    return _ce(type)(pretrained=pretrained, **kwargs)


def nvidia_convnets_processing_utils():
    import numpy as np
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    import numpy as np
    import json
    import requests
    import validators

    class Processing:

        @staticmethod
        def prepare_input_from_uri(uri, cuda=False):
            img_transforms = transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
            )

            if (validators.url(uri)):
                img = Image.open(requests.get(uri, stream=True).raw)
            else:
                img = Image.open(uri)

            img = img_transforms(img)
            with torch.no_grad():
                # mean and std are not multiplied by 255 as they are in training script
                # torch dataloader reads data into bytes whereas loading directly
                # through PIL creates a tensor with floats in [0,1] range
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                img = img.float()
                if cuda:
                    mean = mean.cuda()
                    std = std.cuda()
                    img = img.cuda()
                input = img.unsqueeze(0).sub_(mean).div_(std)

            return input

        @staticmethod
        def pick_n_best(predictions, n=5):
            predictions = predictions.float().cpu().numpy()
            topN = np.argsort(-1*predictions, axis=-1)[:,:n]
            imgnet_classes = Processing.get_imgnet_classes()
            
            results=[]
            for idx,case in enumerate(topN):
                r = []
                for c, v in zip(imgnet_classes[case], predictions[idx, case]):
                    r.append((f"{c}", f"{100*v:.1f}%"))
                print(f"sample {idx}: {r}")
                results.append(r)
            
            return results

        @staticmethod
        def get_imgnet_classes():
            import os
            import json
            imgnet_classes_json = "LOC_synset_mapping.json"

            if not os.path.exists(imgnet_classes_json):
                print("Downloading Imagenet Classes names.")
                import urllib
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/NVIDIA/DeepLearningExamples/master/PyTorch/Classification/ConvNets/LOC_synset_mapping.json", 
                    filename=imgnet_classes_json)
                print("Downloading finished.")
            imgnet_classes = np.array(json.load(open(imgnet_classes_json, "r")))

            return imgnet_classes

    return Processing()
