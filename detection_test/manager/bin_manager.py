"""This script is responsible for managing bins in a camera"""

import logging
import os
import numpy as np
import collections
from bin_process.bin import Bin
from visutils.vis import vis_bins
import tools.utils_geo as geo
import tools.utils_box as utils_box
from tools import nms
from colorama import Fore, Back


class BinManager:
    def __init__(self,
                 bins=None,
                 log=None,
                 detector=None,
                 camera="cam09",
                 write=True):
        self.log = log
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
        elif camera == "cam13":
            self.init_cam_13()

        # number of iteration to wait for new bin detection
        self._wait_new_bin = 10

        self.detector = detector
        self.current_frame = -1

        # exit bin labels of previous cameras
        self._prev_cam_exit = None

        self._empty_bins = []
        self._dummy_bin_count = collections.defaultdict(int)

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
        # self._thres_incoming_bin_bound = [
        #     (165, 117),
        #     (164, 235),
        #     (481, 222),
        #     (475, 110),
        # ]  # bound for detecting incoming

        self._thres_incoming_bin_bound = [
            (110, 117),
            (110, 340),
            (590, 292),
            (570, 114),
        ]

        self._thres_out_bin_bound = [
            (60, 122),  # (111, 225),
            (65, 243),  # (131, 113),
            (100, 236),  # (73, 91),
            (100, 119),  # (48, 213),
        ]
        # self._thres_incoming_bin_init_x = 1420 / 3
        self._thres_max_idle_count = 5
        self._box_conveyor_belt = [
            (51, 120),  # (26, 210),
            (60, 233),  # (61, 82),
            (480, 226),  # (496, 180),
            (474, 110),  # (467, 302),
        ]  # conveyor belt co-ords (x,y) from bottom left

        self._max_det_fail = 8 * self._mul
        self._max_track_fail = 10 * self._mul

        self._default_bin_state = "items"
        self.maxlen = 5 * self._mul
        self._rat_track_det = 0.8  # FIXME : should it be less

        self._min_area = 40 * 70
        self._min_dim = 40
        self._max_area = 120 * 120

    def init_cam_11(self):
        self._left_bins = []
        self._min_iou = 0.4
        self._bin_count = 0
        self._thres_max_idle_count = 5

        self._thres_incoming_bin_bound = [(617, 189), (617, 90), (430, 70),
                                          (430, 176)]

        self._box_conveyor_belt = [(636, 189), (636, 60), (8, 60), (8, 189)]  #

        # self._thres_out_bin_bound = [(0, 77), (0, 187), (15, 184), (15, 72)]
        self._thres_out_bin_bound = [(0, 77), (0, 187), (90, 184), (90, 62)]
        # NOTE: changed out bin bound to work on camera 13

        self._max_det_fail = 4 * self._mul
        self._max_track_fail = 10 * self._mul

        self._default_bin_state = "items"
        self.maxlen = 3 * self._mul
        self._rat_track_det = 1.2

        self._min_dim = 40
        self._max_area = 140 * 140

    def init_cam_13(self):
        self._left_bins = []
        self._min_iou = 0.4
        self._bin_count = 0
        self._thres_max_idle_count = 5

        self._thres_incoming_bin_bound = [(450, 51), (450, 180), (590, 167),
                                          (590, 48)]

        self._box_conveyor_belt = [(177, 65), (180, 194), (590, 167),
                                   (590, 48)]  #

        self._thres_out_bin_bound = [(177, 65), (180, 194), (74, 200),
                                     (74, 76)]

        self._max_det_fail = 5 * self._mul
        self._max_track_fail = 10 * self._mul

        self._default_bin_state = "items"
        self.maxlen = 3 * self._mul
        self._rat_track_det = 1.2

        self._min_dim = 40
        self._max_area = 150 * 150

    # NOTE: ONLY FOR CAMERA 11 or 13
    def set_prev_cam_manager(self, _manager):
        if self._camera == "cam11":
            self._manager_prev_cam = _manager
        elif self._camera == "cam13":
            self._manager_prev_cam = _manager
        else:
            raise Exception("This is only allowed in camera 11 or 13")

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

        if self._camera in ("cam11", "cam13"):
            # set label based on camera 9
            try:
                for i_man in range(len(self._manager_prev_cam._left_bins)):
                    # while len(self._manager_prev_cam._left_bins) > 0:
                    mbin = self._manager_prev_cam._left_bins[i_man]["bin"]
                    mbin_frame = self._manager_prev_cam._left_bins[i_man][
                        "frame_num"]

                    # NOTE: offline mode
                    time_offset = 50 if self._camera == "cam13" else 0

                    # left bin must precede current frame
                    if mbin_frame - time_offset > self.current_frame:
                        continue

                    # for camera 13, left frame and current frame should be close
                    if self._camera == "cam13" and np.abs(
                            mbin_frame - time_offset -
                            self.current_frame) > 100:
                        continue

                    label = mbin.label
                    state = mbin.state

                    # NOTE: don't use label that is already is being used
                    tmp_labs = []
                    for bin in self._current_bins:
                        tmp_labs.append(bin.label)
                    for bin_with_frame in self._left_bins:
                        bin = bin_with_frame["bin"]
                        if hasattr(bin, 'siammask'): bin.clear_track()
                        tmp_labs.append(bin.label)
                    if label in tmp_labs:
                        self._manager_prev_cam._left_bins.pop(i_man)
                        continue
                    break
                else:
                    return
            except IndexError:
                state = "items"
        # elif self._camera == "cam13":
        #     # set label based on camera 9
        #     try:
        #         while len(self._manager_prev_cam._left_bins) > 0:
        #             mbin = self._manager_prev_cam._left_bins.pop(0)["bin"]
        #             mbin_frame = self._manager_prev_cam._left_bins.pop(
        #                 0)["frame_num"]

        #             if mbin_frame - 50 <= self.current_frame:
        #                 self._manager_prev_cam._left_bins.pop(0)
        #                 continue

        #             label = mbin.label
        #             state = mbin.state
        #             tmp_labs = []
        #             for bin in self._current_bins:
        #                 tmp_labs.append(bin.label)
        #             for bin_with_frame in self._left_bins:
        #                 bin = bin_with_frame["bin"]
        #                 tmp_labs.append(bin.label)
        #             if label in tmp_labs:
        #                 continue
        #             break
        #         else:  # else is only executed when there is not break inside while
        #             return
        #     except IndexError:
        #         state = "items"
        #         return

        # NOTE: wait for new bin, wait at least 5 iteration to assign
        self._dummy_bin_count[label] += 1
        if self._dummy_bin_count[label] < self._wait_new_bin:
            self._bin_count -= 1
            return

        new_bin = Bin(
            label=label,
            state=state,
            pos=box,
            default_state=self._default_bin_state,
            maxlen=self.maxlen,
        )

        # # DEBUG
        # if label > 49:
        #     return

        if self._camera != "cam09":
            self._manager_prev_cam._left_bins.pop(i_man)

        new_bin.init_tracker(box, im)
        self._current_bins.append(new_bin)
        self.log.clasp_log(f"{self._camera} : Bin {self._bin_count} enters")

        if self._camera == "cam09":
            self._current_events.append([
                self.current_frame,
                new_bin.label,
                new_bin.cls,
                *new_bin.pos,
                "enter",
                f"B{new_bin.label} enters conveyor belt",
            ])
        elif self._camera == "cam11":
            self._current_events.append([
                self.current_frame,
                new_bin.label,
                new_bin.cls,
                *new_bin.pos,
                "enter",
                f"B{new_bin.label} exits X-Ray",
            ])

    def update_state(self, im, boxes, scores, classes, frame_num):

        self.current_frame = frame_num
        if im is None:
            return

        if classes is None:  # FIXME: ??
            for bin in self._current_bins:
                bin.increment_idle()

        # --------------------------------- Refine bb -------------------------------- #
        if classes is not None:
            ind = [i for i in range(len(classes)) if classes[i] in ("items", )]
        else:
            ind = []

        if len(ind) > 0:
            _boxes, _scores, _classes = boxes[ind], scores[ind], classes[ind]
            if self._camera == "cam09":
                ind = []
                for i in range(len(_boxes)):
                    w, h = (_boxes[i][2] - _boxes[i][0],
                            _boxes[i][3] - _boxes[i][1])
                    if (w > self._min_dim and h > self._min_dim
                            and w * h < self._max_area
                            and w * h > self._min_area):
                        ind.append(i)
                if len(ind) == 0:
                    boxes, scores, classes = None, None, None
                else:
                    _boxes, _scores, _classes = (
                        _boxes[ind],
                        _scores[ind],
                        _classes[ind],
                    )
                    boxes, scores, classes = nms.multi_nms(_boxes,
                                                           _scores,
                                                           _classes,
                                                           thresh=0.4,
                                                           low_score=0.3)
            else:
                ind = []
                for i in range(len(_boxes)):
                    w, h = (_boxes[i][2] - _boxes[i][0],
                            _boxes[i][3] - _boxes[i][1])
                    if (w > self._min_dim and h > self._min_dim
                            and w * h < self._max_area):
                        ind.append(i)
                if len(ind) == 0:
                    boxes, scores, classes = None, None, None
                else:
                    _boxes, _scores, _classes = (
                        _boxes[ind],
                        _scores[ind],
                        _classes[ind],
                    )
                    boxes, scores, classes = nms.multi_nms(_boxes,
                                                           _scores,
                                                           _classes,
                                                           thresh=0.4,
                                                           low_score=0.3)
                # boxes, scores, classes = nms.multi_nms(
                #     _boxes, _scores, _classes, thresh=0.4, low_score=0.3
                # )
        else:
            boxes, scores, classes = None, None, None

        if len(ind) > 0:
            # Sort boxes by x axis
            box_x_pos = [box[0] for box in boxes]
            ind_sort = np.argsort(box_x_pos)
            boxes, scores, classes = (
                boxes[ind_sort],
                scores[ind_sort],
                classes[ind_sort],
            )

        explored_indices = []
        tmp_iou = {}

        # ---------------------------- Track previous bin ---------------------------- #
        if len(ind) > 0 and len(self._current_bins) > 0:
            M_iou = []
            for bin in self._current_bins:
                iou_to_boxes = []
                for _counter in range(boxes.shape[0]):
                    _iou = bin.iou_bbox(boxes[_counter])
                    iou_to_boxes.append(_iou)
                M_iou.append(iou_to_boxes)
            M_iou = np.array(M_iou)
            M_iou[M_iou < self._min_iou] = 0

            box_ind_min = utils_box.get_min_ind_row(
                M_iou, thres=0.5)  # NOTE: Is this too small thres
            counter_box_ind_min = collections.Counter(box_ind_min)
            for bcount, bin in enumerate(self._current_bins):
                status, _bb_track = bin.update_tracker(im)
                if not status:
                    bin.increment_track_fail()
                else:
                    bin.reset_track_fail()

                for _counter in range(boxes.shape[0]):
                    tmp_iou[_counter] = max(
                        tmp_iou.get(_counter, 0),
                        utils_box.iou_bbox(boxes[_counter],
                                           bin.pos,
                                           ratio_type='min'))

                # closest_index = np.argmax(iou_to_boxes)
                closest_index = int(box_ind_min[bcount])

                # NOTE: If there are more than two bins having similar detection bb,
                # rely on tracker then
                rely_tracker_only = False
                if counter_box_ind_min[
                        closest_index] > 1 and bin._pos_count > 30 * self._mul:
                    if status:
                        bin.pos = _bb_track
                    bin.increment_det_fail()
                    rely_tracker_only = True

                if not rely_tracker_only:
                    # both detection and tracker have to agree with each other
                    if closest_index < 0 or closest_index in explored_indices:
                        bin.increment_det_fail()
                        #!
                        if status:
                            bin.pos = _bb_track
                        continue

                    iou_to_boxes = M_iou[bcount, :]
                    # '_min_iou' threshold to regain tracking
                    # we are prioritizing tracker here
                    if iou_to_boxes[closest_index] > self._min_iou:
                        bin.reset_det_fail()
                        new_box = boxes[closest_index]
                        if status:
                            _track_iou = bin.iou_bbox(_bb_track)
                            if (self._camera == "cam09" and
                                    _track_iou > iou_to_boxes[closest_index] *
                                    self._rat_track_det) or (
                                        (self._camera == "cam11"
                                         or self._camera == "cam13")
                                        and utils_box.iou_bbox(
                                            _bb_track, new_box, "min") > 0.85):
                                new_box = _bb_track
                                bin.reset_track_fail()
                            else:
                                bin.init_tracker(new_box, im)

                        #! DEBUG
                        bin.pos = new_box
                        explored_indices.append(closest_index)

                        if bin._pos_count < 30 * self._mul and not status:
                            bin.init_tracker(new_box, im)
                    else:
                        bin.increment_det_fail()
                        # bin.pos = _bb_track
                        # bin.increment_idle()
        else:
            for bin in self._current_bins:
                bin.increment_det_fail()

        # ------------------------------ Detect new bin ------------------------------ #
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

                # if not geo.point_in_box(c_box, self._box_conveyor_belt):
                #     continue

                # NOTE: check if high overlap with existing bins
                # do not add new bin if high overlap
                to_add = True
                for bin in self._current_bins:
                    if bin.iou_bbox(box, ratio_type="min") > 0.35:
                        to_add = False
                        break
                    if self._camera == "cam13":
                        if bin.pos[0] > box[0]:
                            to_add = False
                            break
                if not to_add:
                    continue

                self.add_bin(box, cls, im)  # add new bin

        # detect bin exit
        self.process_exit(im, frame_num)
        return 0

    def visualize(self, im):
        return vis_bins(im, self._current_bins)

    def process_exit(self, im, frame_num):
        _ind = []
        for i in range(len(self)):
            bin = self._current_bins[i]
            # if self._camera == '9':
            if geo.point_in_box(bin.centroid, self._thres_out_bin_bound):
                # bin exit
                self.log.clasp_log(f"{self._camera} : Bin {bin.label} exits")

                # delete tracker for resource minimization
                if hasattr(bin, 'siammask'): bin.clear_track()
                self._left_bins.append({"bin": bin, "frame_num": frame_num})

                if self._camera == "cam09":
                    msg = f"B{bin.label} enters X-Ray"
                else:
                    msg = f"B{bin.label} exits"

                self._current_events.append([
                    self.current_frame, bin.label, bin.cls, *bin.pos, "exit",
                    msg
                ])

            else:
                pass_det = bin.num_det_fail < self._max_det_fail
                pass_track = bin.num_track_fail < self._max_track_fail
                if pass_det and pass_track:
                    _ind.append(i)
                else:
                    # something is wrong
                    if not pass_det:
                        self.log.clasp_log(
                            f"{self._camera} : Bin {bin.label} divested")

                        if self._camera == "cam11":
                            msg = f"B{bin.label} empty"
                            self._current_events.append([
                                self.current_frame,
                                bin.label,
                                bin.cls,
                                *bin.pos,
                                "empty",
                                msg,
                            ])

                        # FIXME: what does this do? Fix It
                        # check if bin has overlap with other bins
                        flag = True
                        for other_bin in self._current_bins:
                            if other_bin.label != bin.label:
                                _iou = utils_box.iou_bbox(
                                    bin.pos, other_bin.pos, "combined")

                                if _iou > 0.5:
                                    # NOTE: In this case, we say that other bin has been
                                    # divested to current bin,
                                    # we left the id of other bin, and retain current bin

                                    if hasattr(bin, 'siammask'):
                                        bin.clear_track()
                                    self._empty_bins.append(bin)
                                    other_bin.label = bin.label  # label swapped
                                    flag = False

                                    # log
                                    self._current_events.append([
                                        self.current_frame,
                                        other_bin.label,
                                        other_bin.cls,
                                        *other_bin.pos,
                                        "chng",
                                        f"Item {other_bin.label} divested/revested",
                                    ])

                                    self.log.info(
                                        f"Item {other_bin.label} divested/revested"
                                    )

                                    break
                        if flag:
                            if hasattr(bin, 'siammask'): bin.clear_track()
                            self._empty_bins.append(bin)

                    else:
                        # NOTE: bin content changed
                        # divestment or revestment
                        self.log.info(
                            f"{self._camera} : Bin {bin.label} - New divestment/revestment"
                        )
                        _ind.append(i)
                        bin.init_tracker(bin.pos, im)
                        bin.reset_track_fail()

                        self._current_events.append([
                            self.current_frame,
                            bin.label,
                            bin.cls,
                            *bin.pos,
                            "chng",
                            f"Item {bin.label} divested/revested",
                        ])

        self._current_bins = [self._current_bins[i] for i in _ind]

    def add_info(self, list_info, im):
        for each_i in list_info:
            _id, cls, x1, y1, x2, y2 = each_i
            if cls == "items":
                box = [x1, y1, x2, y2]
                self._bin_count = max(self._bin_count, _id)
                new_bin = Bin(
                    label=_id,
                    state=cls,
                    pos=box,
                    default_state=self._default_bin_state,
                    maxlen=self.maxlen,
                )
                new_bin.init_tracker(box, im)
                self._current_bins.append(new_bin)

    def add_exit_info(self, list_info):
        for each_i in list_info:
            _id, cls, x1, y1, x2, y2, _type, frame_num = each_i
            if cls == "items":
                box = [x1, y1, x2, y2]
                new_bin = Bin(
                    label=_id,
                    state=cls,
                    pos=box,
                    default_state=self._default_bin_state,
                    maxlen=self.maxlen,
                )
                if hasattr(new_bin, 'siammask'): new_bin.clear_track()
                if _type == "exit":
                    self._left_bins.append({
                        "bin": new_bin,
                        "frame_num": frame_num
                    })
                elif _type == "empty":
                    self._empty_bins.append(new_bin)
                self._bin_count = max(self._bin_count, _id)