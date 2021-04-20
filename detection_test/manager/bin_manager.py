"""This script is responsible for managing bins in a camera"""

import os
from tools.time_calculator import ComplexityAnalysis
from typing import Any, Dict, List, NewType, Optional, Sequence, TypeVar
from nptyping import NDArray
import numpy as np
import collections
import itertools
import copy
from omegaconf import OmegaConf

from tools.utils import BinType
from bin_process.bin import Bin
from visutils.vis import vis_bins
import tools.utils_geo as geo
import tools.utils_box as utils_box
from tools import nms
import pandas as pd

ImageType = Optional[NDArray[(Any, Any, 3), np.float]]
BoxesType = NDArray[(Any, 4), np.float]
ScoresType = NDArray[(Any, ), np.float]

ClassesType = NDArray[(Any, ), Any]  # last Any should be BinType


class BinManager:
    def __init__(self,
                 config,
                 bins=None,
                 log=None,
                 camera="cam09",
                 analyzer: Optional[ComplexityAnalysis] = None):
        self.config = config
        self.log = log
        if bins is None:
            self._current_bins: List[Bin] = []
        else:
            self._current_bins: List[Bin] = bins
        self._current_events = []
        self.analyzer = analyzer

        self._camera = camera

        self.init_camera_params()

        self.current_frame = -1

        # exit bin labels of previous cameras
        self._prev_cam_exit = None

        self._empty_bins = []
        self._left_bins = []
        self._dummy_bin_count = collections.defaultdict(int)

    def init_camera_params(self):
        # initialize configuration
        self._mul = 3
        # CWD = os.path.dirname(os.path.abspath(__file__))
        # conf_file = os.path.join(CWD, "bin_params.yml")
        # bin_params_for_camera = OmegaConf.load(conf_file)
        bin_params_for_camera = OmegaConf.to_container(self.config.bin_params)

        spatial_params = [
            "_thres_incoming_bin_bound", "_thres_out_bin_bound",
            "_box_conveyor_belt", "_min_dim"
        ]
        spatial_params_area = ["_min_area", "_max_area"]

        temporal_params = [
            "_thres_max_idle_count",
            "_max_det_fail",
            "_max_track_fail",
            "maxlen",
        ]

        for k, v in itertools.chain(
                bin_params_for_camera[self._camera].items(),
                bin_params_for_camera['all'].items()):
            if isinstance(v, str) and "*" in v:
                v = eval(v)
            if k in spatial_params:
                v = recursive_mul(v, self.config.spatial_scale_mul)
            elif k in spatial_params_area:
                v = recursive_mul(v, self.config.spatial_scale_mul**2)
            elif k in temporal_params:
                v = recursive_mul(v, self.config.temporal_scale_mul)

            setattr(self, k, v)

    # NOTE: ONLY FOR CAMERA 11 or 13
    def set_prev_cam_manager(self, _manager):
        if self._camera == "cam11":
            self._manager_prev_cam = _manager
        elif self._camera == "cam13":
            self._manager_prev_cam = _manager
        else:
            raise Exception("This is only allowed in camera 11 or 13")

    def add_bin(self,
                box: NDArray[(Any, 4), np.float],
                cls: BinType,
                im: ImageType,
                safe=True):

        self._bin_count += 1
        label = self._bin_count

        if self._camera != "cam09" and cls == BinType.BIN_EMPTY:
            # NOTE empty bin is detected only in camera 9
            return

        if self._camera in ("cam11", "cam13"):
            # set label based on camera 9
            label = None
            try:
                i_man = 0
                do_return = False
                while True:
                    if i_man >= len(self._manager_prev_cam._left_bins):
                        do_return = True
                        break
                    # while len(self._manager_prev_cam._left_bins) > 0:
                    mbin = self._manager_prev_cam._left_bins[i_man]["bin"]
                    mbin_frame = self._manager_prev_cam._left_bins[i_man][
                        "frame_num"]

                    # FIXME
                    time_offset = int(
                        2 / 10 *
                        self.config.fps) if self._camera == "cam13" else 0

                    # left bin must precede current frame
                    if mbin_frame - time_offset > self.current_frame:
                        i_man += 1
                        continue

                    # for camera 13, left frame and current frame should be close
                    if self._camera == "cam13" and np.abs(
                            mbin_frame - time_offset -
                            self.current_frame) > 100:
                        i_man += 1
                        continue

                    label = mbin.label

                    # NOTE: don't use label that is already is being used
                    tmp_labs = []
                    for bin in self._current_bins:
                        tmp_labs.append(bin.label)
                    # for bin_with_frame in self._left_bins:
                    #     bin = bin_with_frame["bin"]
                    #     bin.clear_track()
                    #     tmp_labs.append(bin.label)
                    if label in tmp_labs:
                        self._manager_prev_cam._left_bins.pop(i_man)
                        # i_man += 1
                        continue

                    # self._manager_prev_cam._left_bins.pop(i_man)
                    break

                if do_return:
                    self._bin_count -= 1
                    return
            except IndexError:
                return
        # elif self._camera == "cam13":
        #     # set label based on camera 11
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
        #         return

        if label is None:
            self._bin_count -= 1
            return

        # NOTE: wait for new bin, wait at least 5 iteration to assign
        if safe and self._camera != "cam13":
            self._dummy_bin_count[label] += 1
            if self._dummy_bin_count[label] < self._wait_new_bin:
                self._bin_count -= 1
                return

        new_bin = Bin(label=label,
                      bin_type=cls,
                      pos=box,
                      maxlen=self.maxlen,
                      conf=self.config)

        # FIXME why this?
        if self._camera != "cam09":
            self._manager_prev_cam._left_bins.pop(i_man)

        new_bin.init_tracker(box, im)
        self._current_bins.append(new_bin)
        self.log.clasp_log(f"{self._camera} : Bin {label} enters")

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

    def _filter_boxes(self, im: ImageType, boxes: BoxesType,
                      scores: ScoresType, classes: ClassesType,
                      frame_num: int):
        # --------------------------------- Refine bb -------------------------------- #
        if classes is not None:
            ind = [
                i for i in range(len(classes))
                # if classes[i] in ("items", "item", "bin_empty")
            ]
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
        return boxes, scores, classes, ind

    def _track_current_bins(self, im: ImageType, boxes: BoxesType,
                            scores: ScoresType, classes: ClassesType,
                            frame_num: int, ind: int):
        # ---------------------------- Track previous bin ---------------------------- #
        explored_indices = []
        tmp_iou = {}
        if len(ind) > 0 and len(self._current_bins) > 0:
            M_iou = []
            for bin in self._current_bins:
                iou_to_boxes = []
                for _counter in range(boxes.shape[0]):
                    _iou = bin.iou_bbox(boxes[_counter], ratio_type='comb')
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

                # for later use: new bin
                for _counter in range(boxes.shape[0]):
                    tmp_iou[_counter] = max(
                        tmp_iou.get(_counter, 0),
                        utils_box.iou_bbox(boxes[_counter],
                                           bin.pos,
                                           ratio_type='comb'))

                # closest_index = np.argmax(iou_to_boxes)
                closest_index = int(box_ind_min[bcount])

                # NOTE: If there are more than two bins having similar detection bb,
                # rely on tracker then
                rely_tracker_only = False
                if (counter_box_ind_min[closest_index] > 1
                        and bin._pos_count > 30 * self._mul):
                    if status:
                        _w, _h = (_bb_track[2] - _bb_track[0],
                                  _bb_track[3] - _bb_track[1])
                        if _w * _h < self._max_area and _w * _h > self._min_area:
                            bin.pos = _bb_track
                            rely_tracker_only = True
                    bin.increment_det_fail()

                if not rely_tracker_only:
                    # both detection and tracker have to agree with each other
                    if closest_index < 0 or closest_index in explored_indices:
                        # REVIEW It
                        bin.increment_det_fail()
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

                        bin.bin_type = classes[closest_index]
                        bin.pos = new_box
                        explored_indices.append(closest_index)

                        if bin._pos_count < 30 * self._mul and not status:
                            bin.init_tracker(new_box, im)
                    else:
                        bin.increment_det_fail()

        else:
            for bin in self._current_bins:
                bin.increment_det_fail()

        return explored_indices, tmp_iou

    def _detect_new_bin(self, im: ImageType, boxes: BoxesType,
                        scores: ScoresType, classes: ClassesType,
                        frame_num: int, ind: List, explored_indices: List,
                        tmp_iou: Dict):
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

    def update_state(self, im: ImageType, boxes: BoxesType, scores: ScoresType,
                     classes: ClassesType, frame_num: int):

        self.current_frame = frame_num
        if im is None:
            return

        if classes is None:  # FIXME: ??
            for bin in self._current_bins:
                bin.increment_idle()

        with self.analyzer(f"TRACK_FILTERBOX_{self._camera}", False, True):
            boxes, scores, classes, ind = self._filter_boxes(
                im, boxes, scores, classes, frame_num)

        with self.analyzer(f"TRACK_CURRENT_{self._camera}", False, False):
            explored_indices, tmp_iou = self._track_current_bins(
                im, boxes, scores, classes, frame_num, ind)

        with self.analyzer(f"TRACK_NEW_{self._camera}", False, True):
            self._detect_new_bin(im, boxes, scores, classes, frame_num, ind,
                                 explored_indices, tmp_iou)

        with self.analyzer(f"TRACK_EXIT_{self._camera}", False, True):
            self._process_exit(im, frame_num)
        return 0  # successful return

    def visualize(self, im: ImageType):
        return vis_bins(im, self._current_bins)

    def _process_exit(self, im, frame_num):
        _ind = []

        list_overlap = []
        for i in range(len(self)):
            bin = self._current_bins[i]
            do_exit = False
            _l = []
            for j in range(len(self)):
                if j == i: _l.append(-1)
                _iou = bin.iou_bbox(self._current_bins[j].pos,
                                    ratio_type='min1')
                _l.append(_iou)
            list_overlap.append(_l)

        for i in range(len(self)):
            bin = self._current_bins[i]

            # overlam min with all other bins
            do_exit = False
            for j in range(len(self)):
                _iou = list_overlap[i][j]
                if (_iou > 0.8 and self._current_bins[j].bin_type ==
                        bin.bin_type == BinType.BIN_EMPTY):
                    # exit next bin
                    do_exit = True
                    list_overlap[j][i] = -1
                    break

                if do_exit:
                    self.log.clasp_log(
                        f"{self._camera} : Bin {bin.label} deleted")

                    continue

            if (self._camera != "cam09" and bin.bin_type == BinType.BIN_EMPTY
                    and bin._count_persistent_type > 5):
                # empty bin
                self.log.clasp_log(f"{self._camera} : Bin {bin.label} empty")
                continue

            if geo.point_in_box(bin.centroid, self._thres_out_bin_bound):
                # FIXME: Only for camera 11
                # bin is not completely out of bound from camera 11,
                # it's still need to be tracked
                if self._camera == "cam11":
                    if not geo.point_in_box(bin.centroid,
                                            self._thres_out_bin_bound):
                        _ind.append(i)
                    else:
                        bin.clear_track()
                        self.log.clasp_log(
                            f"{self._camera} : Bin {bin.label} out from camera view"
                        )
                    # check if bin has already entered to cam 13
                    found = False
                    for _bininfo in self._left_bins:
                        if bin.label == _bininfo["bin"].label:
                            found = True
                            break
                    if found:
                        continue

                # bin exit
                self.log.clasp_log(f"{self._camera} : Bin {bin.label} exits")

                # delete tracker for resource minimization
                if self._camera != "cam11":
                    bin.clear_track()
                    self._left_bins.append({
                        "bin": bin,
                        "frame_num": frame_num
                    })
                else:
                    _tmp = copy.copy(bin)
                    _tmp.clear_track()
                    self._left_bins.append({
                        "bin": _tmp,
                        "frame_num": frame_num
                    })

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
                        # divested only if there's a bigger bin
                        for j in range(len(self._current_bins)):
                            if list_overlap[i][j] > 0.9:

                                list_overlap[j][i] = -1
                                bin.clear_track()

                                # log
                                other_bin = self._current_bins[j]
                                self.log.clasp_log(
                                    f"{self._camera} : Bin {bin.label} divested"
                                )
                                self._current_events.append([
                                    self.current_frame,
                                    other_bin.label,
                                    other_bin.cls,
                                    *other_bin.pos,
                                    "chng",
                                    f"Item {other_bin.label} divested/revested",
                                ])
                                self._empty_bins.append(bin)

                                break
                        else:  # if not break
                            self.log.clasp_log(
                                f"{self._camera} : Bin {bin.label} deleted")

                        if self._camera != "cam09":
                            msg = f"B{bin.label} empty"
                            self._current_events.append([
                                self.current_frame,
                                bin.label,
                                bin.cls,
                                *bin.pos,
                                "empty",
                                msg,
                            ])

                        # else:
                        #     # REVIEW: check if bin has overlap with other bins, cam09 only
                        #     flag = True
                        #     for other_bin in self._current_bins:
                        #         if other_bin.label != bin.label:
                        #             _iou = utils_box.iou_bbox(
                        #                 bin.pos, other_bin.pos, "combined")

                        #             if _iou > 0.5:
                        #                 # NOTE: In this case, we say that other bin has been
                        #                 # divested to current bin,
                        #                 # we left the id of other bin, and retain current bin

                        #                 bin.clear_track()
                        #                 self._empty_bins.append(bin)
                        #                 other_bin.label = bin.label  # label swapped
                        #                 flag = False

                        #                 # log
                        #                 self._current_events.append([
                        #                     self.current_frame,
                        #                     other_bin.label,
                        #                     other_bin.cls,
                        #                     *other_bin.pos,
                        #                     "chng",
                        #                     f"Item {other_bin.label} divested/revested",
                        #                 ])

                        #                 self.log.info(
                        #                     f"Item {other_bin.label} divested/revested"
                        #                 )
                        #                 break
                        #     if flag:
                        #         bin.clear_track()
                        #         self._empty_bins.append(bin)

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

    def add_info(self, list_info: List[Any], im):
        for each_i in list_info:
            _id, cls, x1, y1, x2, y2 = each_i
            if isinstance(cls, BinType):
                box = [x1, y1, x2, y2]
                self._bin_count = max(self._bin_count, _id)
                new_bin = Bin(label=_id,
                              bin_type=cls,
                              pos=box,
                              maxlen=self.maxlen,
                              conf=self.config)
                new_bin.init_tracker(box, im)
                self._current_bins.append(new_bin)
            else:
                raise ValueError(f"{cls} is not of type BinType")

    def add_exit_info(self, list_info: List[Any]):
        for each_i in list_info:
            _id, cls, x1, y1, x2, y2, _type, frame_num = each_i
            if isinstance(cls, BinType):
                box = [x1, y1, x2, y2]
                new_bin = Bin(label=_id,
                              bin_type=cls,
                              pos=box,
                              maxlen=self.maxlen,
                              conf=self.config)
                new_bin.clear_track()
                if _type == "exit":
                    self._left_bins.append({
                        "bin": new_bin,
                        "frame_num": frame_num
                    })
                elif _type == "empty":
                    self._empty_bins.append(new_bin)
                self._bin_count = max(self._bin_count, _id)
            else:
                raise ValueError(f"{cls} is not of type BinType")

    def __len__(self):
        return len(self._current_bins)

    def __getitem__(self, index):
        return self._current_bins[index]

    def __iter__(self):
        return iter(self._current_bins)


# ------------------------------ Helper function ----------------------------- #
def recursive_mul(item, scale=1):
    """ recursively multiply all elements in item """
    if isinstance(item, collections.abc.Iterable):
        item = type(item)([recursive_mul(x, scale) for x in item])
        return item
    else:
        return type(item)(item * scale)