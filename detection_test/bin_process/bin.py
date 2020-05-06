"""
Bin class description
"""

import math
from collections import deque
import tools.utils_box as utils
import cv2
import numpy as np
import torch

# import simatrack

from siammask.tools.test import siamese_init, siamese_track, load_config, load_pretrain
from siammask.siammask_sharp.custom import Custom
import argparse


class Bin:
    def __init__(self,
                 label=None,
                 state=None,
                 pos=None,
                 default_state="items",
                 maxlen=10):

        self.maxlen = maxlen
        self.init_conf()

        self._default_state = default_state
        self._label = label
        self.state = state
        self.pos = pos  # (x1, y1, x2, y2)

    def init_conf(self):
        self._state_conf_num = 20
        self._state_store_list = deque([], maxlen=self.maxlen)

        self._idle_count = 0
        self._num_detection_fail = 0
        self._num_tracker_fail = 0

        self._pos_count = 0

    # def init_tracker(self, box, frame):
    #     self.tracker = cv2.TrackerKCF_create()
    #     bb = tuple([box[0], box[1], box[2]-box[0]+1, box[3]-box[1]+1])
    #     self.tracker.init(frame, bb)

    @torch.no_grad()
    def init_tracker(self, box, frame):
        if hasattr(self, 'siammask'):
            del self.siammask
        self.load_siammask()
        self.device = torch.device("cuda:0")
        bb = tuple([box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1])
        x, y, w, h = bb
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        state = siamese_init(frame,
                             target_pos,
                             target_sz,
                             self.siammask,
                             self.cfg_siam["hp"],
                             device=self.device)
        self.track_state = state

    @torch.no_grad()
    def load_siammask(self):
        class Empty:
            pass

        args = Empty()
        args.resume = "siammask/siammask_sharp/SiamMask_DAVIS.pth"
        args.config = "siammask/siammask_sharp/config_davis.json"

        cfg = load_config(args)

        net = Custom(anchors=cfg["anchors"])
        net = load_pretrain(net, args.resume)
        net.eval().cuda()
        self.siammask = net
        self.cfg_siam = cfg

    @torch.no_grad()
    def update_tracker(self, frame):
        prev_pos = self.track_state["target_pos"]
        state = siamese_track(self.track_state,
                              frame,
                              mask_enable=False,
                              refine_enable=True,
                              device=self.device)  # track
        target_pos = state["target_pos"]
        target_sz = state["target_sz"]

        w, h = target_sz
        x, y = target_pos - target_sz / 2

        distance = np.linalg.norm(prev_pos - target_pos)

        if distance > 30:
            status = False
        else:
            status = True

        if status:
            x2, y2 = x + w, y + h
            bbox = [int(x), int(y), int(x2), int(y2)]
            self.track_state = state
        else:
            # tracker failed
            bbox = None

        return status, bbox

    def clear_track(self):
        if hasattr(self, 'siammask'): del self.siammask
        if hasattr(self, 'track_state'): del self.track_state

    def increment_idle(self):
        self._idle_count += 1

    def increment_det_fail(self):
        self._num_detection_fail += 1

    def increment_track_fail(self):
        self._num_tracker_fail += 1

    @property
    def num_det_fail(self):
        return self._num_detection_fail

    @property
    def num_track_fail(self):
        return self._num_tracker_fail

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, _l):
        self._label = _l

    @property
    def state(self):
        return "items"

    @property
    def cls(self):
        return "items"

    @property
    def pos(self):
        return self._pos

    @property
    def width(self):
        return self._pos[2] - self._pos[0] + 1

    @property
    def centroid(self):
        x1, y1, x2, y2 = self.pos
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def height(self):
        return self._pos[3] - self._pos[1] + 1

    @property
    def idle_count(self):
        return self._idle_count

    @pos.setter
    def pos(self, new_pos):
        self._pos = new_pos
        self._pos_count += 1

    def reset_det_fail(self):
        self._num_detection_fail = 0

    def reset_track_fail(self):
        self._num_tracker_fail = 0

    @state.setter
    def state(self, new_state):
        self._state_store_list.append(new_state)

    @property
    def area(self):
        return self.width * self.height

    def set_bound(self, width, height):
        """ Bound of the whole scene """
        self._bound = (width, height)

    @staticmethod
    def dist(x1, y1, x2, y2):
        """ distance between two points  """
        return utils.dist(x1, y1, x2, y2)

    @staticmethod
    def calc_centroid(x1, y1, x2, y2):
        return (x1 + x2) / 2, (y1 + y2) / 2

    def distance(self, bin):
        """ distance between two bin  """
        x1, y1 = self.centroid
        x2, y2 = bin.centroid
        return Bin.dist(x1, y1, x2, y2)

    def distance_bbox(self, bbox):
        """ distance from new x1,y1,x2,y2 """
        x1, y1 = self.centroid
        x2, y2 = Bin.calc_centroid(*bbox)
        return Bin.dist(x1, y1, x2, y2)

    def iou_bbox(self, bbox, ratio_type="min"):
        return utils.iou_bbox(self.pos, bbox, ratio_type)

    def __eq__(self, bin2):
        if self.pos == bin2.pos and self.label == bin2.label and self.state == bin2.state:
            return True
        return False