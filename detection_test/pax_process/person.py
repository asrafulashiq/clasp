""" Perosn class
"""

import tools.utils_box as utils

class Person:
    def __init__(self, label=None, pos=None):
        # self.init_conf()

        self.label = label  # set label
        self.pos = pos  # set pos : (x1, y1, x2, y2)

    @property
    def label(self):
        return self._label

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, new_pos):
        self._pos = new_pos

    @label.setter
    def label(self, new_label):
        self._label = new_label

    @property
    def centroid(self):
        x1, y1, x2, y2 = self.pos
        return (x1+x2)/2, (y1+y2)/2

    @property
    def width(self):
        return self._pos[2] - self._pos[0]

    @property
    def height(self):
        return self._pos[3] - self._pos[1]

    @property
    def area(self):
        return self.width * self.height

    def iou_bbox(self, bbox, ratio_type='min'):
        return utils.iou_bbox(self.pos, bbox, ratio_type)
