"""This script is responsible for managing passengers in a camera"""

import logging
import os
import numpy as np

from pax_process.person import Person
from visutils.vis import vis_pax


class PAXManager:
    """ passenger manager """

    def __init__(self, pax=None, log=None, detector=None, camera='9'):
        self.log = log
        if log is None:
            self.log = logging.getLogger('pax manager')
            self.log.setLevel(logging.DEBUG)
            self.log.clasp_log = self.log.info

        if pax is None:
            self._current_pax = []
        else:
            self._current_pax = pax
        self._camera = camera

        # initialize configuration
        if camera == '9':
            self.init_cam_9()

        self.detector = detector

    @property
    def detector(self):
        return self._detector

    @detector.setter
    def detector(self, _det):
        self._detector = _det

    def init_cam_9(self):
        self._left_pax = []
        self._min_iou = 0.4
        self._pax_count = 0
        self._thres_incoming_pax_exit = 460  # x
        self._thres_out_pax_exit = 350
        self._thres_incoming_pax_init_x = 1420

    def __len__(self):
        return len(self._current_pax)

    def __getitem__(self, index):
        return self._current_pax[index]

    def __iter__(self):
        return iter(self._current_pax)

    def add_pax(self, box):
        #! Just for now
        self._pax_count = 0

        self._pax_count += 1
        new_pax = Person(
            label=self._pax_count,
            pos=box
        )
        self._current_pax.append(new_pax)
        # self.log.clasp_log(f"Person {self._pax_count} enters")

    def update_state(self, im, boxes, scores, classes):

        if im is None:
            return None

        if classes is None:
            return

        ind = [i for i in range(len(classes)) if classes[i] == 'person']

        if len(ind) > 0:
            boxes, scores, classes = boxes[ind], scores[ind], classes[ind]
        else:
            boxes, scores, classes = None, None, None

        explored_indices = []

        tmp_iou = {}

        # if len(ind) > 0:
        #     for pax in self._current_pax:
        #         iou_to_boxes = []
        #         for _counter in range(boxes.shape[0]):
        #             _iou = pax.iou_bbox(boxes[_counter])
        #             iou_to_boxes.append(_iou)
        #             tmp_iou[_counter] = max(tmp_iou.get(_counter, 0), _iou)

        #         closest_index = np.argmax(iou_to_boxes)
        #         if closest_index in explored_indices:
        #             continue

        #         if iou_to_boxes[closest_index] > self._min_iou:
        #             pax.pos = boxes[closest_index]
        #             pax.state = classes[closest_index]
        #             explored_indices.append(closest_index)

        # # refine detected pax and detect new pax
        # if len(ind) > 0:
        #     for i in range(boxes.shape[0]):
        #         if i in explored_indices:
        #             continue
        #         # if new pax is closest to a current pax
        #         # set the current pax as new pax
        #         box, _, cls = boxes[i], scores[i], classes[i]

        #         self.add_pax(box)  # add new pax


        #! Just for now
        if len(ind) > 0:
            self._current_pax = []
            for i in range(boxes.shape[0]):
                if i in explored_indices:
                    continue
                # if new pax is closest to a current pax
                # set the current pax as new pax
                box, _, cls = boxes[i], scores[i], classes[i]

                self.add_pax(box)  # add new pax


        # detect pax exit
        # self.process_exit()
        return 0

    def visualize(self, im):
        return vis_pax(im, self._current_pax)

    def process_exit(self):
        _ind = []
        for i in range(len(self)):
            pax = self._current_pax[i]
            if pax.centroid[0] < self._thres_out_pax_exit:
                # pax exit
                self.log.clasp_log(f"Person {pax.label} exits")
                self._left_pax.append(pax)
            else:
                _ind.append(i)
        self._current_pax = [self._current_pax[i] for i in _ind]
