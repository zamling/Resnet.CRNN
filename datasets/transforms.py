
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string





class RandomBrightness(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            brightness_factor = random.uniform(0.5, 2)
            image = F.adjust_brightness(image, brightness_factor)
        return image, target

class RandomContrast(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            contrast_factor = random.uniform(0.5, 2)
            image = F.adjust_contrast(image, contrast_factor)
        return image, target

class RandomHue(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            hue_factor = random.uniform(-0.25, 0.25)
            image = F.adjust_hue(image, hue_factor)
        return image, target

class RandomSaturation(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            saturation_factor = random.uniform(0.5, 2)
            image = F.adjust_saturation(image, saturation_factor)
        return image, target

class RandomGamma(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            gamma_factor = random.uniform(0.5, 2)
            image = F.adjust_gamma(image, gamma_factor)
        return image, target
