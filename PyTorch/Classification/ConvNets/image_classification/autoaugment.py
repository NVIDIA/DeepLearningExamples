from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random


class AutoaugmentImageNetPolicy(object):
    """
    Randomly choose one of the best 24 Sub-policies on ImageNet.
    Reference: https://arxiv.org/abs/1805.09501
    """
    def __init__(self):
        self.policies = [
            SubPolicy(0.8, "equalize", 1, 0.8, "shearY", 4),
            SubPolicy(0.4, "color", 9, 0.6, "equalize", 3),
            SubPolicy(0.4, "color", 1, 0.6, "rotate", 8),
            SubPolicy(0.8, "solarize", 3, 0.4, "equalize", 7),
            SubPolicy(0.4, "solarize", 2, 0.6, "solarize", 2),

            SubPolicy(0.2, "color", 0, 0.8, "equalize", 8),
            SubPolicy(0.4, "equalize", 8, 0.8, "solarizeadd", 3),
            SubPolicy(0.2, "shearX", 9, 0.6, "rotate", 8),
            SubPolicy(0.6, "color", 1, 1.0, "equalize", 2),
            SubPolicy(0.4, "invert", 9, 0.6, "rotate", 0),

            SubPolicy(1.0, "equalize", 9, 0.6, "shearY", 3),
            SubPolicy(0.4, "color", 7, 0.6, "equalize", 0),
            SubPolicy(0.4, "posterize", 6, 0.4, "autocontrast", 7),
            SubPolicy(0.6, "solarize", 8, 0.6, "color", 9),
            SubPolicy(0.2, "solarize", 4, 0.8, "rotate", 9),

            SubPolicy(1.0, "rotate", 7, 0.8, "translateY", 9),
            SubPolicy(0.0, "shearX", 0, 0.8, "solarize", 4),
            SubPolicy(0.8, "shearY", 0, 0.6, "color", 4),
            SubPolicy(1.0, "color", 0, 0.6, "rotate", 2),
            SubPolicy(0.8, "equalize", 4, 0.0, "equalize", 8),

            SubPolicy(1.0, "equalize", 4, 0.6, "autocontrast", 2),
            SubPolicy(0.4, "shearY", 7, 0.6, "solarizeadd", 7),
            SubPolicy(0.8, "posterize", 2, 0.6, "solarize", 10),
            SubPolicy(0.6, "solarize", 8, 0.6, "equalize", 1),
            SubPolicy(0.8, "color", 6, 0.4, "rotate", 5),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class SubPolicy(object):
    def __init__(self, p1, method1, magnitude_idx1, p2, method2, magnitude_idx2):
        operation_factory = OperationFactory()
        self.p1 = p1
        self.p2 = p2
        self.operation1 = operation_factory.get_operation(method1, magnitude_idx1)
        self.operation2 = operation_factory.get_operation(method2, magnitude_idx2)

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img)
        if random.random() < self.p2:
            img = self.operation2(img)
        return img


class OperationFactory:
    def __init__(self):
        fillcolor = (128, 128, 128)
        self.ranges = {
            "shearX": np.linspace(0, 0.3, 11),
            "shearY": np.linspace(0, 0.3, 11),
            "translateX": np.linspace(0, 250, 11),
            "translateY": np.linspace(0, 250, 11),
            "rotate": np.linspace(0, 30, 11),
            "color": np.linspace(0.1, 1.9, 11),
            "posterize": np.round(np.linspace(0, 4, 11), 0).astype(np.int),
            "solarize": np.linspace(0, 256, 11),
            "solarizeadd": np.linspace(0, 110, 11),
            "contrast": np.linspace(0.1, 1.9, 11),
            "sharpness": np.linspace(0.1, 1.9, 11),
            "brightness": np.linspace(0.1, 1.9, 11),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        def rotate_with_fill(img, magnitude):
            magnitude *= random.choice([-1, 1])
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        def solarize_add(image, addition=0, threshold=128):
            lut = []
            for i in range(256):
                if i < threshold:
                    res = i + addition if i + addition <= 255 else 255
                    res = res if res >= 0 else 0
                    lut.append(res)
                else:
                    lut.append(i)
            from PIL.ImageOps import _lut
            return _lut(image, lut)

        self.operations = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(magnitude),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "solarizeadd": lambda img, magnitude: solarize_add(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(magnitude),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(magnitude),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(magnitude),
            "autocontrast": lambda img, _: ImageOps.autocontrast(img),
            "equalize": lambda img, _: ImageOps.equalize(img),
            "invert": lambda img, _: ImageOps.invert(img)
        }

    def get_operation(self, method, magnitude_idx):
        magnitude = self.ranges[method][magnitude_idx]
        return lambda img: self.operations[method](img, magnitude)
