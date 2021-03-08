from abc import abstractmethod
import random
from typing import Any
from imgaug.augmenters.meta import Sometimes
import torch

from torchvision.transforms import functional as F
import cv2
import imgaug as ia
import imgaug.augmenters as iaa


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Resize(object):
    def __init__(self, size, orig_size=(1920, 1080)):
        self.size = size  # w, h
        self.orig_size = orig_size

    def __call__(self, image, target):
        width, height = self.orig_size
        new_w, new_h = self.size
        rat_w, rat_h = new_w / width, new_h / height
        image = F.resize(image, self.size[::-1])
        bbox = target["boxes"]
        bbox[:, [0, 2]] *= rat_w
        bbox[:, [1, 3]] *= rat_h
        return image, target


from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np


class ExtraBBAug():
    def __init__(self,
                 rotate=(-30, 30),
                 translate_percent={
                     "x": (0, 0.2),
                     "y": (0, 0.2)
                 }) -> None:
        self.angles = rotate
        self.translate_percent = translate_percent

    def __call__(self, image, target) -> Any:
        image_np = np.asarray(image)
        bb = _convert_bb(image_np, target['boxes'])
        tsfm = iaa.Sequential([
            iaa.Sometimes(0.2, iaa.Multiply((1.2, 1.5), per_channel=0.2)),
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.Affine(translate_percent=self.translate_percent,
                       scale=(0.8, 1.2)),
            iaa.Sometimes(0.5, iaa.Affine(rotate=(-90, 90)))
        ],
                              random_order=True)
        im, bb_ts = tsfm(image=image_np, bounding_boxes=bb)
        bb_torch = _convert_bb_to_torch(bb_ts)
        target['boxes'] = bb_torch
        return im, target


def _convert_bb_to_torch(bb: BoundingBoxesOnImage) -> torch.Tensor:
    _list = []
    for i in range(len(bb)):
        el = [bb[i].x1, bb[i].y1, bb[i].x2, bb[i].y2]
        _list.append(el)
    return torch.tensor(_list)


def _convert_bb(image, boundingboxes):
    return BoundingBoxesOnImage([BoundingBox(*bb) for bb in boundingboxes],
                                shape=image.shape)
