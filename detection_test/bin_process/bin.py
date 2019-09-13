"""Bin class description
"""

import math
from collections import deque
import tools.utils_box as utils
import cv2

class Bin:
    def __init__(self, label=None, state=None, pos=None,
                 default_state="items", maxlen=10):

        self.maxlen = maxlen
        self.init_conf()

        self._default_state = default_state
        self._label = label
        self.state = state
        self.pos = pos  # (x1, y1, x2, y2)

        # bin tracker
        self.tracker = cv2.TrackerKCF_create()

    def init_conf(self):
        self._state_conf_num = 20
        self._state_store_list = deque([], maxlen=self.maxlen)
        self._idle_count = 0

    def increment_idle(self):
        self._idle_count += 1

    @property
    def label(self):
        return self._label

    @property
    def state(self):
        return self._state

    @property
    def cls(self):
        return self._state

    @property
    def pos(self):
        return self._pos

    @property
    def width(self):
        return self._pos[2] - self._pos[0] + 1

    @property
    def centroid(self):
        x1, y1, x2, y2 = self.pos
        return (x1+x2)/2, (y1+y2)/2

    @property
    def height(self):
        return self._pos[3] - self._pos[1] + 1

    @property
    def idle_count(self):
        return self._idle_count

    @pos.setter
    def pos(self, new_pos):
        self._pos = new_pos
        self._idle_count = 0

    @state.setter
    def state(self, new_state):
        self._state_store_list.append(new_state)
        if len(self._state_store_list) < self._state_store_list.maxlen:
            self._state = self._default_state
        else:
            self._state = max(self._state_store_list,
                              key=self._state_store_list.count)

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
        return (x1+x2)/2, (y1+y2)/2

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

    def iou_bbox(self, bbox, ratio_type='min'):
        return utils.iou_bbox(self.pos, bbox, ratio_type)

    def __eq__(self, bin2):
        if self.pos == bin2.pos and self.label == bin2.label and \
            self.state == bin2.state:
            return True
        return False

    def init_tracker(self, box, frame):
        bb = tuple([box[0], box[1], box[2]-box[0]+1, box[3]-box[1]+1])
        self.tracker.init(frame, bb)

    def update_tracker(self, frame):
        status, bb = self.tracker.update(frame)

        if status:
            x, y, w, h = bb
            x2, y2 = x + w, y + h
            bbox = [x, y, x2, y2]
        else:
            # tracker failed
            bbox = None
        return status, bbox

