"""This script is responsible for managing bins in a camera"""

import logging
import os
import numpy as np

from bin_process.bin import Bin
from visutils.vis import vis_bins
import tools.utils_geo as geo
import tools.utils_box as utils_box
from tools import nms


class BinManager:

    def __init__(self, bins=None, log=None, detector=None, camera='cam09'):
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
        if camera == 'cam09':
            self.init_cam_9()
        elif camera == 'cam11':
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
        self._thres_incoming_bin_bound = [
            (143, 230), (170, 120), (496, 180), (467, 302)
        ]  # bound for detecting incoming
        # self._thres_out_bin_exit = 350 / 3

        self._thres_out_bin_bound = [
            (122, 231), (149, 111), (73, 91), (48, 213)
        ]
        # self._thres_incoming_bin_init_x = 1420 / 3
        self._thres_max_idle_count = 5
        self._box_conveyor_belt = [
            (26, 210), (61, 82), (496, 180), (467, 302)
        ]  # conveyor belt co-ords (x,y) from bottom left

        self._max_det_fail = 5
        self._max_track_fail = 10

        self._default_bin_state = "items"
        self.maxlen = 5

    def init_cam_11(self):
        self._left_bins = []
        self._min_iou = 0.4
        self._bin_count = 0
        self._thres_incoming_bin_exit = 1530  # x
        self._thres_out_bin_exit = 110
        self._thres_incoming_bin_init_x = 1700
        self._thres_max_idle_count = 5

        self._default_bin_state = "items"
        self.maxlen = 5

    # * ONLY FOR CAMERA 11

    def set_cam9_manager(self, _manager):
        if self._camera == 'cam11':
            self._manager_cam_9 = _manager
        else:
            raise Exception("This is only allowed in camera 11")

    def __len__(self):
        return len(self._current_bins)

    def __getitem__(self, index):
        return self._current_bins[index]

    def __iter__(self):
        return iter(self._current_bins)

    def add_bin(self, box, cls, im):
        self._bin_count += 1
        label = self._bin_count
        state = cls

        if self._camera == 'cam11':
            # set label based on camera 9
            try:
                mbin = self._manager_cam_9._left_bins.pop(0)
                label = mbin.label
                state = mbin.state
            except IndexError:
                state = "items"

        new_bin = Bin(
            label=label,
            state=state,
            pos=box,
            default_state=self._default_bin_state,
            maxlen=self.maxlen
        )
        new_bin.init_tracker(box, im)
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
               in ('items', )]

        if len(ind) > 0:
            _boxes, _scores, _classes = boxes[ind], scores[ind], classes[ind]
            boxes, scores, classes = nms.multi_nms(_boxes, _scores, _classes,
                                                   thresh=0.4, low_score=0.3)
        else:
            boxes, scores, classes = None, None, None

        explored_indices = []

        tmp_iou = {}

        if len(ind) > 0:
            for bin in self._current_bins:
                status, _bb_track = bin.update_tracker(im)
                if not status:
                    bin.increment_track_fail()
                else:
                    bin.reset_track_fail()

                iou_to_boxes = []
                for _counter in range(boxes.shape[0]):
                    _iou = bin.iou_bbox(boxes[_counter])
                    iou_to_boxes.append(_iou)
                    tmp_iou[_counter] = max(tmp_iou.get(_counter, 0),
                                            utils_box.iou_bbox(boxes[_counter], bin.pos))

                closest_index = np.argmax(iou_to_boxes)
                if closest_index in explored_indices:
                    bin.increment_det_fail()
                    continue

                if iou_to_boxes[closest_index] > self._min_iou:
                    bin.reset_det_fail()
                    new_box = boxes[closest_index]
                    if status:
                        _track_iou = bin.iou_bbox(_bb_track)
                        # _track_iou = utils_box.iou_bbox(_bb_track, new_box)
                        if _track_iou > iou_to_boxes[closest_index] * 0.8:
                            new_box = _bb_track
                            bin.reset_track_fail()
                        else:
                            bin.init_tracker(new_box, im)

                    bin.pos = new_box
                    explored_indices.append(closest_index)
                else:
                    bin.increment_det_fail()
                    # bin.increment_idle()
        else:
            for bin in self._current_bins:
                bin.increment_det_fail()

        # refine detected bins and detect new bin
        if len(ind) > 0:
            for i in range(boxes.shape[0]):
                if i in explored_indices:
                    continue
                # if new bin is closest to a current bin
                # set the current bin as new bin
                box, _, cls = boxes[i], scores[i], classes[i]
                c_box = Bin.calc_centroid(*box)

                if not geo.point_in_box(c_box, self._thres_incoming_bin_bound):
                    continue

                if tmp_iou.get(i, 0) > self._min_iou:
                    continue

                if not geo.point_in_box(c_box,
                                        self._box_conveyor_belt):
                    continue

                # if self._camera == 'cam11' and cls == 'items':
                #     continue

                self.add_bin(box, cls, im)  # add new bin

        # detect bin exit
        self.process_exit(im)
        return 0

    def visualize(self, im):
        return vis_bins(im, self._current_bins)

    def process_exit(self, im):
        _ind = []
        for i in range(len(self)):
            bin = self._current_bins[i]
            # if self._camera == '9':
            if geo.point_in_box(bin.centroid, self._thres_out_bin_bound):
                # bin exit
                self.log.clasp_log(f"Bin {bin.label} exits")
                self._left_bins.append(bin)
            # elif self._camera == 'cam11':
            #     # if bin is emptied in camera 11, then don't process
            #     self.log.clasp_log(f"Bin {bin.label} divested")
            #     # self._left_bins.append(bin)
            else:
                # if bin.idle_count < self._thres_max_idle_count:
                #     _ind.append(i)

                pass_det = bin.num_det_fail < self._max_det_fail
                pass_track =  bin.num_track_fail < self._max_track_fail
                if pass_det and pass_track:
                    _ind.append(i)
                else:
                    # something is wrong
                    if (not pass_det):
                        pass
                    else:
                        # divestment or revestment
                        self.log.info(f"Bin {bin.label} - New divestment/revestment")
                        _ind.append(i)
                        bin.init_tracker(bin.pos, im)
                        bin.reset_track_fail()

        self._current_bins = [self._current_bins[i] for i in _ind]
