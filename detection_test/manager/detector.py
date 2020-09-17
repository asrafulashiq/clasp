from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy as np

from rcnn import rcnn_utils

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)
_RED = (255, 10, 10)
_BLUE = (10, 10, 255)

class_to_color = {"passengers": _BLUE, "items": _GREEN, "other": _RED}


def get_clasp_dataset():
    """A dummy CLASP dataset that includes only the 'classes' field."""
    classes = ["__background__", "passengers", "items", "other"]
    ds = {i: name for i, name in enumerate(classes)}
    return ds


def get_class_string(class_index, dataset):
    class_text = (dataset[class_index]
                  if dataset is not None else "id{:d}".format(class_index))
    return class_text


def vis_class(img, pos, class_str, font_scale=0.5):
    """Visualizes the class."""
    img = img.astype(np.uint8)
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0

    color = class_to_color.get(class_str, _RED)
    cv2.rectangle(img, back_tl, back_br, color, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img,
                txt,
                txt_tl,
                font,
                font_scale,
                _GRAY,
                lineType=cv2.LINE_AA)
    return img


def vis_bbox(img, bbox, class_str=None, thick=2):
    """Visualizes a bounding box."""
    img = img.astype(np.uint8)
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    color = class_to_color.get(class_str, _RED)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=thick)
    return img


def vis_one_image_opencv(im,
                         boxes,
                         scores,
                         classes,
                         thresh=0.5,
                         dataset=None,
                         show_class=True,
                         vis=True):
    """Constructs a numpy array with the detections visualized."""

    if boxes is None or boxes.shape[0] == 0 or max(scores) < thresh:
        return None, None, None, None

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    _boxes = []
    _score = []
    _class = []

    for i in sorted_inds:
        bbox = boxes[i]
        score = scores[i]
        if score < thresh:
            continue

        # show class (off by default)
        class_str = get_class_string(classes[i], dataset)
        if vis and show_class:
            im = vis_class(im, (bbox[0], bbox[1] - 2), class_str)

        if vis:
            im = vis_bbox(
                im,
                (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]),
                class_str,
            )

        _boxes.append(bbox)
        _score.append(score)
        _class.append(class_str)

    return im, np.array(_boxes), np.array(_score), np.array(_class)


class DummyDetector:
    def __init__(self, ckpt=None, thres=0.5, labels_to_keep=(1, 2)):
        self.model = None
        self.ckpt = ckpt

        # threshold for score
        self.thres = thres

        self.labels_to_keep = labels_to_keep
        # 1 : pax, 2: items

        self.create_model()

    def create_model(self):
        self.model = rcnn_utils.RCNN_Detector(
            ckpt=self.ckpt,
            labels_to_keep=self.labels_to_keep,
            thres=self.thres)
        #* lables to keep is important, 1 means pax, 2 is bins
        self.dummy_coco_dataset = get_clasp_dataset()

    def predict_box(self, im, show=False):
        ret = self.model(im)
        if ret is not None:
            boxes, scores, classes = ret
            nim = im
            # if show:
            nim, boxes, scores, classes = vis_one_image_opencv(
                im,
                boxes,
                scores,
                classes,
                thresh=self.thres,
                dataset=self.dummy_coco_dataset,
                vis=show)
            return nim, boxes, scores, classes
        else:
            return im, None, None, None

    def predict_box_batch(self, imlist, show=False):
        rets = self.model.predict_batch(imlist)
        results = []
        for i, ret in enumerate(rets):
            im = imlist[i]
            if ret is not None:
                boxes, scores, classes = ret
                nim = imlist[i]
                # if show:
                nim, boxes, scores, classes = vis_one_image_opencv(
                    im,
                    boxes,
                    scores,
                    classes,
                    thresh=self.thres,
                    dataset=self.dummy_coco_dataset,
                    vis=show)
                results.append((nim, boxes, scores, classes))
            else:
                results.append((im, None, None, None))
        return results


class BinDetector:
    def __init__(self, ckpt=None, thres=0.5):
        self.model = None
        self.ckpt = ckpt
        self.thres = thres
        self.create_model()

    def create_model(self):
        self.model = rcnn_utils.RCNN_Detector(ckpt=self.ckpt,
                                              labels_to_keep=(2, 3),
                                              thres=self.thres)
        self.dummy_coco_dataset = get_clasp_dataset()

    def predict_box(self, im, show=False):
        ret = self.model(im)
        if ret is not None:
            boxes, scores, classes = ret
            nim = im
            if show:
                nim, boxes, scores, classes = vis_one_image_opencv(
                    im,
                    boxes,
                    scores,
                    classes,
                    thresh=self.thres,
                    dataset=self.dummy_coco_dataset,
                )
            return nim, boxes, scores, classes
        else:
            return im, None, None, None


class PAXDetector:
    def __init__(self, ckpt=None, thres=0.5, labels_to_keep=(1, 2)):
        self.model = None
        self.ckpt = ckpt

        # threshold for score
        self.thres = thres

        self.labels_to_keep = labels_to_keep
        # 1 : pax, 2: items

        self.create_model()

    def create_model(self):
        self.model = rcnn_utils.RCNN_Detector(ckpt=self.ckpt,
                                              labels_to_keep=(1, ),
                                              thres=self.thres)
        self.dummy_coco_dataset = get_clasp_dataset()

    def predict_box(self, im, show=False):
        ret = self.model(im)
        if ret is not None:
            boxes, scores, classes = ret
            nim = im
            if show:
                nim, boxes, scores, classes = vis_one_image_opencv(
                    im,
                    boxes,
                    scores,
                    classes,
                    thresh=self.thres,
                    dataset=self.dummy_coco_dataset,
                )
            return nim, boxes, scores, classes
        else:
            return im, None, None, None
