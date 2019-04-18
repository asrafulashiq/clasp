"""This script is responsible for managing bins in a camera"""

import logging
import os
import numpy as np

from bin_process.bin import Bin
from visutils.vis import vis_bins


class BinManager:

    def __init__(self, bins=None, log=None, detector=None, camera='9'):
        self.log = log
        if log is None:
            self.log = logging.getLogger('bin manager')
            self.log.setLevel(logging.DEBUG)
            self.log.clasp_log = self.log.info

        if bins is None:
            self._current_bins = []
        else:
            self._current_bins = bins

        self._camera = camera

        # initialize configuration
        if camera == '9':
            self.init_cam_9()
        elif camera == '11':
            self.init_cam_11()

        self.detector = detector

    @property
    def detector(self):
        return self._detector

    @detector.setter
    def detector(self, _det):
        self._detector = _det

    def init_cam_9(self):
        self._left_bins = []
        self._min_iou = 0.4
        self._bin_count = 0
        self._thres_incoming_bin_exit = 460  # x
        self._thres_out_bin_exit = 350
        self._thres_incoming_bin_init_x = 1420
        self._thres_max_idle_count = 5

        self._default_bin_state = "bin_empty"
        self.maxlen = 30

    def init_cam_11(self):
        self._left_bins = []
        self._min_iou = 0.4
        self._bin_count = 0
        self._thres_incoming_bin_exit = 1530  # x
        self._thres_out_bin_exit = 110
        self._thres_incoming_bin_init_x = 1700
        self._thres_max_idle_count = 5

        self._default_bin_state = "bin_full"
        self.maxlen = 5


    #* ONLY FOR CAMERA 11
    def set_cam9_manager(self, _manager):
        if self._camera == '11':
            self._manager_cam_9 = _manager
        else:
            raise Exception("This is only allowed in camera 11")


    def __len__(self):
        return len(self._current_bins)

    def __getitem__(self, index):
        return self._current_bins[index]

    def __iter__(self):
        return iter(self._current_bins)

    def add_bin(self, box, cls):
        self._bin_count += 1
        label = self._bin_count
        state = cls

        if self._camera == '11':
            # set label based on camera 9
            try:
                mbin = self._manager_cam_9._left_bins.pop(0)
                label = mbin.label
                state = mbin.state
            except IndexError:
                state = "bin_full"

        new_bin = Bin(
            label=label,
            state=state,
            pos=box,
            default_state=self._default_bin_state,
            maxlen=self.maxlen
        )
        self._current_bins.append(new_bin)
        self.log.clasp_log(f"Bin {self._bin_count} enters")

    def update_state(self, im, boxes, scores, classes):

        if im is None:
            return

        if classes is None:
            for bin in self._current_bins:
                bin.increment_idle()
            return

        ind = [i for i in range(len(classes)) if classes[i]
               in ('bin_empty', 'bin_full')]

        if len(ind) > 0:
            boxes, scores, classes = boxes[ind], scores[ind], classes[ind]
        else:
            boxes, scores, classes = None, None, None

        explored_indices = []

        tmp_iou = {}

        if len(ind) > 0:
            for bin in self._current_bins:
                iou_to_boxes = []
                for _counter in range(boxes.shape[0]):
                    _iou = bin.iou_bbox(boxes[_counter])
                    iou_to_boxes.append(_iou)
                    tmp_iou[_counter] = max(tmp_iou.get(_counter, 0), _iou)

                closest_index = np.argmax(iou_to_boxes)
                if closest_index in explored_indices:
                    bin.increment_idle()
                    continue

                if iou_to_boxes[closest_index] > self._min_iou:
                    bin.pos = boxes[closest_index]
                    bin.state = classes[closest_index]
                    explored_indices.append(closest_index)
                else:
                    bin.increment_idle()

        # refine detected bins and detect new bin
        if len(ind) > 0:
            for i in range(boxes.shape[0]):
                if i in explored_indices:
                    continue
                # if new bin is closest to a current bin
                # set the current bin as new bin
                box, _, cls = boxes[i], scores[i], classes[i]

                if Bin.calc_centroid(*box)[0] < self._thres_incoming_bin_exit:
                    continue

                if tmp_iou.get(i, 0) > self._min_iou:
                    continue

                if Bin.calc_centroid(*box)[0] > self._thres_incoming_bin_init_x:
                    continue

                if self._camera == '11' and cls == 'bin_empty':
                    continue

                self.add_bin(box, cls)  # add new bin

        # detect bin exit
        self.process_exit()
        return 0

    def visualize(self, im):
        return vis_bins(im, self._current_bins)

    def process_exit(self):
        _ind = []
        for i in range(len(self)):
            bin = self._current_bins[i]
            # if self._camera == '9':
            if bin.centroid[0] < self._thres_out_bin_exit:
                # bin exit
                self.log.clasp_log(f"Bin {bin.label} exits")
                self._left_bins.append(bin)
            elif self._camera == '11' and bin.state == "bin_empty":
                # if bin is emptied in camera 11, then don't process
                self.log.clasp_log(f"Bin {bin.label} divested")
                # self._left_bins.append(bin)
            else:
                if bin.idle_count < self._thres_max_idle_count:
                    _ind.append(i)

        self._current_bins = [self._current_bins[i] for i in _ind]

