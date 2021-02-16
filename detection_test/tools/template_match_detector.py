import cv2
import os, sys
import numpy as np
from typing import List, Tuple, Any


class TemplateMatchDetector(object):
    def __init__(self):

        template_path = [
            os.path.expanduser("~/Desktop/template.png"),
            os.path.expanduser("~/Desktop/template2.png"),
            os.path.expanduser("~/Desktop/template3.png")
        ]

        self.templates = []
        for path in template_path:
            assert os.path.exists(path)
            template = cv2.imread(path)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            template = cv2.resize(template,
                                  dsize=(0, 0),
                                  fx=1. / 3,
                                  fy=1. / 3,
                                  interpolation=cv2.INTER_CUBIC)
            self.templates.append(template)

        self.method = cv2.TM_CCOEFF_NORMED
        self.threshold = 0.6

    def detect(self, image: np.ndarray, *args,
               **kwargs) -> Tuple[Any, Any, Any]:
        # -> Boxes, Scores, Classes

        im_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        boxes, scores, classes = [], [], []
        for template in self.templates:
            res = cv2.matchTemplate(im_gray, self.template, self.method)
            loc = np.where(res >= self.threshold)

            w, h = self.template.shape[:2]
            _boxes = np.array([(*pt, pt[0] + w, pt[1] + h)
                               for pt in zip(*loc[::-1])])
            _boxes = non_max_suppression_fast(_boxes)

            _scores = np.array([0.95] * len(_boxes))
            _classes = np.array(['items'] * len(_boxes))

            boxes.append(_boxes)
            scores.append(_scores)
            classes.append(_classes)

        if len(boxes) > 0:
            boxes = np.concatenate(boxes, axis=0)
            classes = np.concatenate(classes, axis=0)
            scores = np.concatenate(scores, axis=0)

        return boxes, scores, classes


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh=0.5) -> np.ndarray:
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("float")