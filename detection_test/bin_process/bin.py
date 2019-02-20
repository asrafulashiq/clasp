"""Bin class description
"""

import math
from collections import deque

class Bin:

    def __init__(self, label=None, state=None, pos=None):
        self.init_conf()

        self._label = label
        self.state = state
        self.pos = pos  # (x1, y1, x2, y2)


    def init_conf(self):
        self._state_conf_num = 20
        self._state_store_list = deque([], maxlen=30)

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
        return self._pos[2] - self._pos[0]

    @property
    def centroid(self):
        x1, y1, x2, y2 = self.pos
        return (x1+x2)/2, (y1+y2)/2

    @property
    def height(self):
        return self._pos[3] - self._pos[1]

    @pos.setter
    def pos(self, new_pos):
        self._pos = new_pos

    @state.setter
    def state(self, new_state):
        self._state_store_list.append(new_state)
        if len(self._state_store_list) < \
            self._state_store_list.maxlen:
            self._state = "bin_empty"
        else:
            self._state = max(self._state_store_list,
                              key=self._state_store_list.count)

    @property
    def area(self):
        return self.width * self.height

    def set_bound(self, width, height):
        """ Bound of the whole scene """
        self._bound = (w, h)

    @staticmethod
    def dist(x1, y1, x2, y2):
        """ distance between two points  """
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

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
        a = self.pos
        b = bbox

        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        # AREA OF OVERLAP - Area where the boxes intersect
        width = (x2 - x1)
        height = (y2 - y1)
        # handle case where there is NO overlap
        if (width < 0) or (height < 0):
            return 0.0
        area_overlap = width * height

        # COMBINED AREA
        if ratio_type == 'min':
            b_area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            area_combined = min(b_area, self.area)
        else:
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            area_combined = area_a + area_b - area_overlap

        # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
        iou = area_overlap / (area_combined+1e-5)
        return iou

    def __eq__(self, bin2):
        if self.pos == bin2.pos and self.label == bin2.label and \
            self.state == bin2.state:
            return True
        return False
