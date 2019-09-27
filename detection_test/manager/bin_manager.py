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
    def __init__(self, bins=None, log=None, detector=None, camera="cam09", write=True):
        self.log = log
        if log is None:
            self.log = logging.getLogger("bin manager")
            self.log.setLevel(logging.DEBUG)
            self.log.clasp_log = self.log.info

        if bins is None:
            self._current_bins = []
        else:
            self._current_bins = bins
        self._current_events = []

        self._camera = camera

        # initialize configuration
        self._mul = 3
        if camera == "cam09":
            self.init_cam_9()
        elif camera == "cam11":
            self.init_cam_11()

        self.detector = detector
        self.current_frame = -1


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
            (123, 227),
            (146, 111),
            (496, 180),
            (467, 302),
        ]  # bound for detecting incoming
        # self._thres_out_bin_exit = 350 / 3

        self._thres_out_bin_bound = [(111, 225), (131, 113), (73, 91), (48, 213)]
        # self._thres_incoming_bin_init_x = 1420 / 3
        self._thres_max_idle_count = 5
        self._box_conveyor_belt = [
            (26, 210),
            (61, 82),
            (496, 180),
            (467, 302),
        ]  # conveyor belt co-ords (x,y) from bottom left

        self._max_det_fail = 5 * self._mul
        self._max_track_fail = 10 * self._mul

        self._default_bin_state = "items"
        self.maxlen = 5 * self._mul
        self._rat_track_det = 0.8

        self._min_area = 40 * 40
        self._min_dim = 40
        self._max_area = 120 * 120

    def init_cam_11(self):
        self._left_bins = []
        self._min_iou = 0.4
        self._bin_count = 0
        self._thres_incoming_bin_exit = 1530  # x
        self._thres_out_bin_exit = 110
        self._thres_incoming_bin_init_x = 1700
        self._thres_max_idle_count = 5

        self._thres_incoming_bin_bound = [(617, 189), (617, 90), (430, 70), (430, 176)]

        self._box_conveyor_belt = [(636, 189), (636, 60), (8, 60), (8, 189)]  #

        self._thres_out_bin_bound = [(0, 77), (0, 187), (15, 184), (15, 72)]

        self._max_det_fail = 3 * self._mul
        self._max_track_fail = 10 * self._mul

        self._default_bin_state = "items"
        self.maxlen = 5 * self._mul
        self._rat_track_det = 1.6

    # * ONLY FOR CAMERA 11

    def set_cam9_manager(self, _manager):
        if self._camera == "cam11":
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

        if self._camera == "cam11":
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
            maxlen=self.maxlen,
        )
        new_bin.init_tracker(box, im)
        self._current_bins.append(new_bin)
        self.log.clasp_log(f"Bin {self._bin_count} enters")
        self._current_events.append(
            [self.current_frame, new_bin.label, new_bin.cls, *new_bin.pos, "enter",
            f"Item {new_bin.label} enters in camera {self._camera[-2:]}"]
        )

    def update_state(self, im, boxes, scores, classes, frame_num):

        self.current_frame = frame_num
        if im is None:
            return

        if classes is None:
            for bin in self._current_bins:
                bin.increment_idle()
            return

        ind = [i for i in range(len(classes)) if classes[i] in ("items",)]

        if len(ind) > 0:
            _boxes, _scores, _classes = boxes[ind], scores[ind], classes[ind]
            if self._camera == 'cam09':
                ind = []
                for i in range(len(_boxes)):
                    w, h = _boxes[i][2]-_boxes[i][0], _boxes[i][3] - _boxes[i][1]
                    if w > self._min_dim and h > self._min_dim and w * h < self._max_area:
                        ind.append(i)
                if len(ind) == 0:
                    boxes, scores, classes = None, None, None
                else:
                    _boxes, _scores, _classes = _boxes[ind], _scores[ind], _classes[ind]
                    boxes, scores, classes = nms.multi_nms(
                        _boxes, _scores, _classes, thresh=0.4, low_score=0.3
                    )
            else:
                boxes, scores, classes = nms.multi_nms(
                    _boxes, _scores, _classes, thresh=0.4, low_score=0.3
                )
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
                    tmp_iou[_counter] = max(
                        tmp_iou.get(_counter, 0),
                        utils_box.iou_bbox(boxes[_counter], bin.pos),
                    )

                closest_index = np.argmax(iou_to_boxes)
                if closest_index in explored_indices:
                    bin.increment_det_fail()
                    continue

                if iou_to_boxes[closest_index] > self._min_iou:
                    bin.reset_det_fail()
                    new_box = boxes[closest_index]
                    if status:
                        _track_iou = bin.iou_bbox(_bb_track)
                        if (
                            self._camera == "cam09"
                            and _track_iou
                            > iou_to_boxes[closest_index] * self._rat_track_det
                        ) or (
                            self._camera == "cam11"
                            and utils_box.iou_bbox(_bb_track, new_box, 'min') > 0.85
                        ):
                            new_box = _bb_track
                            bin.reset_track_fail()
                        else:
                            bin.init_tracker(new_box, im)

                    bin.pos = new_box
                    explored_indices.append(closest_index)

                    if bin._pos_count < 40 * self._mul and not status:
                        bin.init_tracker(new_box, im)

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

                if not geo.point_in_box(c_box, self._box_conveyor_belt):
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

                self._current_events.append(
                    [self.current_frame, bin.label, bin.cls, *bin.pos, "exit",
                    f"Item {bin.label} exits from camera {self._camera[-2:]}"]
                )

            else:
                pass_det = bin.num_det_fail < self._max_det_fail
                pass_track = bin.num_track_fail < self._max_track_fail
                if pass_det and pass_track:
                    _ind.append(i)
                else:
                    # something is wrong
                    if not pass_det:
                        self.log.clasp_log(f"Bin {bin.label} divested")

                        # check if bin has overlap with other bins
                        for other_bin in self._current_bins:
                            if other_bin.label != bin.label:
                                _iou = utils_box.iou_bbox(bin.pos, other_bin.pos, "combined")

                                if _iou > 0.5:
                                    self._current_events.append(
                                        [self.current_frame, other_bin.label, other_bin.cls, *other_bin.pos, "chng",
                                        f"Item {other_bin.label} divested/revested"]
                                    )
                                    break

                    else:
                        # divestment or revestment
                        self.log.info(f"Bin {bin.label} - New divestment/revestment")
                        _ind.append(i)
                        bin.init_tracker(bin.pos, im)
                        bin.reset_track_fail()

                        self._current_events.append(
                            [self.current_frame, bin.label, bin.cls, *bin.pos, "chng",
                            f"Item {bin.label} divested/revested"]
                        )

        self._current_bins = [self._current_bins[i] for i in _ind]


    def add_info(self, list_info, im):
        for each_i in list_info:
            _id, cls, x1, y1, x2, y2 = each_i
            if cls == "items":
                box = [x1, y1, x2, y2]
                new_bin = Bin(
                    label=_id,
                    state=cls,
                    pos=box,
                    default_state=self._default_bin_state,
                    maxlen=self.maxlen,
                )
                new_bin.init_tracker(box, im)
                self._current_bins.append(new_bin)