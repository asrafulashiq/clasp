#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
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
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.environ['HOME']+'clasp/detectron')


from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)
_RED = (10, 10, 255)
_BLUE = (255, 10, 10)


class_to_color = {
    'person':_BLUE,
    'bin_empty':_GREEN,
    'bin_full':_RED
}


def get_class_string(class_index, score, dataset):
    class_text = dataset.classes[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text #+ ' {:0.2f}'.format(score).lstrip('0')


def vis_class(img, pos, class_str, font_scale=1):
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
    cv2.putText(img, txt, txt_tl, font, font_scale,
                _GRAY, lineType=cv2.LINE_AA)
    return img


def vis_bbox(img, bbox, class_str=None, thick=3):
    """Visualizes a bounding box."""
    img = img.astype(np.uint8)
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    color = class_to_color.get(class_str, _RED)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=thick)
    return img


def vis_one_image_opencv(
        im, boxes, segms=None, keypoints=None, thresh=0.9, kp_thresh=2,
        show_box=False, dataset=None, show_class=False):
    """Constructs a numpy array with the detections visualized."""

    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return None, None, None, None

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    _boxes = []
    _score = []
    _class = []

    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue

        # show class (off by default)
        class_str = get_class_string(classes[i], score, dataset)
        if show_class:
            im = vis_class(im, (bbox[0], bbox[1] - 2), class_str)

        # show box (off by default)
        if show_box:
            im = vis_bbox(
                im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]), class_str)

        _boxes.append(bbox)
        _score.append(score)
        _class.append(class_str)

    return im, np.array(_boxes), np.array(_score), np.array(_class)


class Conf():
    pass


def parse_conf_bin(**kw):
    conf = Conf()
    conf.thresh = 0.7
    conf.kp_thresh = 2.0
    conf.im_or_folder = None
    for k, v in kw.items():
        setattr(conf, k, v)
    return conf


def parse_conf_pax(**kw):
    conf = Conf()
    conf.thresh = 0.7
    conf.kp_thresh = 2.0
    conf.im_or_folder = None
    for k, v in kw.items():
        setattr(conf, k, v)
    return conf



def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes


class BinDetector():
    def __init__(self, cfg=None, weights=None):
        self.args = parse_conf_bin(
            cfg=cfg, weights=weights
        )
        self.model = None
        self.create_model()

    def create_model(self):
        merge_cfg_from_file(self.args.cfg)
        # cfg.NUM_GPUS = 1
        cfg.immutable(False)
        self.args.weights = cache_url(self.args.weights, cfg.DOWNLOAD_CACHE)
        assert_and_infer_cfg(cache_urls=False)

        assert not cfg.MODEL.RPN_ONLY, \
            'RPN models are not supported'
        assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
            'Models that require precomputed proposals are not supported'

        model = infer_engine.initialize_model_from_cfg(self.args.weights)
        # dummy_coco_dataset = dummy_datasets.get_coco_dataset()
        dummy_coco_dataset = dummy_datasets.get_clasp_dataset()

        assert not cfg.MODEL.RPN_ONLY, \
            'RPN models are not supported'
        assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
            'Models that require precomputed proposals are not supported'

        self.model = infer_engine.initialize_model_from_cfg(self.args.weights)
        self.dummy_coco_dataset = dummy_datasets.get_clasp_dataset()

    def predict_box(self, im, show=False):
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, _, _ = infer_engine.im_detect_all(
                self.model, im, None, timers=timers
            )

        n_im, boxes, scores, _class = vis_one_image_opencv(
            im, cls_boxes, segms=None, keypoints=None, thresh=self.args.thresh,
            kp_thresh=self.args.kp_thresh,
            show_box=show, dataset=self.dummy_coco_dataset, show_class=show)
        return n_im, boxes, scores, _class


class PAXDetector():
    def __init__(self, cfg=None, weights=None):
        self.args = parse_conf_pax(
            cfg=cfg, weights=weights
        )
        self.model = None
        self.create_model()

    def create_model(self):
        cfg.immutable(False)
        merge_cfg_from_file(self.args.cfg)
        # cfg.NUM_GPUS = 1
        self.args.weights = cache_url(self.args.weights, cfg.DOWNLOAD_CACHE)
        assert_and_infer_cfg(cache_urls=False)

        assert not cfg.MODEL.RPN_ONLY, \
            'RPN models are not supported'
        assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
            'Models that require precomputed proposals are not supported'

        self.model = infer_engine.initialize_model_from_cfg(self.args.weights)
        self.dummy_coco_dataset = dummy_datasets.get_clasp_dataset()

    def predict_box(self, im, show=False):
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, _, _ = infer_engine.im_detect_all(
                self.model, im, None, timers=timers
            )


        n_im, boxes, scores, _class = vis_one_image_opencv(
            im, cls_boxes, segms=None, keypoints=None, thresh=self.args.thresh,
            kp_thresh=self.args.kp_thresh,
            show_box=show, dataset=self.dummy_coco_dataset, show_class=show)
        return n_im, boxes, scores, _class



if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    # setup_logging(__name__)
    # args = parse_args()
    dirname = os.path.dirname(os.path.abspath(__file__))
    cfg='/home/ash/Desktop/clasp/detectron/'+\
        'configs/my_cfg/e2e_faster_rcnn_R-50-C4_1x.yaml'
    weights='/home/ash/Desktop/clasp/weights/detection-weights' +\
        '/train/clasp_detect/generalized_rcnn/model_final.pkl',

    bin_detection = BinDetector(cfg=cfg, weights=weights)
    imfile = '/home/ash/Desktop/images_track_clasp/images/2332.jpg'
    im = cv2.imread(imfile)
    print("BEGIN PREDICTION")
    new_im, boxes = bin_detection.predict_box(im)
    from matplotlib import pyplot as plt
    plt.imshow(new_im[:,:,::-1])
    plt.show()
