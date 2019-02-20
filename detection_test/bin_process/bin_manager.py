"""This script is responsible for managing bins in a camera"""

import logging
import os
import numpy as np

from bin_process.bin import Bin
from bin_process.bin_detector import BinDetector
from visutils.vis import vis_bins


class BinManager:

    def __init__(self, bins=None, detector=None, log=None, camera='9',
                cfg=None, weights=None):
        self.log = log
        if log is None:
            self.log = logging.getLogger('bin manager')
            self.log.setLevel(logging.DEBUG)

        self._detector = detector
        if bins is None:
            self._current_bins = []
        else:
            self._current_bins = bins
        self._camera = camera

        # initialize configuration
        self.init_conf()
        if weights is not None and cfg is not None:
            assert os.path.exists(weights), f"{weights} path not found"
            assert os.path.exists(cfg), f"{cfg} path not found"
            self.init_detector(cfg=cfg, weights=weights)


    def init_conf(self):
        """ config for bin manager """

        self._left_bins = []

        self._min_iou = 0.4
        self._bin_count = 0
        self._thres_incoming_bin_exit = 460  # x
        self._thres_out_bin_exit = 350

    def init_detector(self, cfg=None, weights=None):
        self.log.info("Bin Detector initializing")
        self._detector = BinDetector(cfg=cfg, weights=weights)

    def __len__(self):
        return len(self._current_bins)

    def __getitem__(self, index):
        return self._current_bins[index]

    def __iter__(self):
        return iter(self._current_bins)

    def add_bin(self, box, cls):
        self._bin_count += 1
        new_bin = Bin(
            label=self._bin_count,
            state=cls,
            pos=box
        )
        self._current_bins.append(new_bin)
        self.log.info(f"Bin {self._bin_count} enters")


    def run_detector_on_image(self, im=None):
        if im is None:
            self.log.warning("No image detected")
            return im

        if self._detector is None:
            self.log.error("Detector not initialized")
            return im

        new_im, boxes, scores, classes = self._detector.predict_box(im, show=True)

        if boxes is None:
            self.log.info("No bin detected")
            return im

        explored_indices = []

        tmp_iou = {}
        for bin in self._current_bins:

            iou_to_boxes = []
            for _counter in range(boxes.shape[0]):
                _iou = bin.iou_bbox(boxes[_counter])
                iou_to_boxes.append(_iou)
                tmp_iou[_counter] = max(tmp_iou.get(_counter, 0), _iou)

            closest_index = np.argmax(iou_to_boxes)
            if closest_index in explored_indices:
                continue

            if iou_to_boxes[closest_index] > self._min_iou:
                bin.pos = boxes[closest_index]
                bin.state = classes[closest_index]
                explored_indices.append(closest_index)


        # refine detected bins and detect new bin
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

            self.add_bin(box, cls)  # add new bin

        # detect bin exit
        self.process_exit()
        return vis_bins(im, self._current_bins)


    def process_exit(self):
        _ind = []
        for i in range(len(self)):
            bin = self._current_bins[i]
            if bin.centroid[0] < self._thres_out_bin_exit:
                # bin exit
                self.log.info(f"Bin {bin.label} exits")
            else:
                _ind.append(i)
        self._current_bins = [self._current_bins[i] for i in _ind]

