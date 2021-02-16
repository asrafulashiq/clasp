"""This script is responsible for managing bins in a camera"""

import os
import numpy as np
import collections
import itertools
import copy
from omegaconf import OmegaConf

from bin_process.bin import Bin
from visutils.vis import vis_bins
import tools.utils_geo as geo
import tools.utils_box as utils_box
from tools import nms
import pandas as pd


class BinManager:
    def __init__(self, config, bins=None, log=None, camera="cam09"):
        self.config = config
        self.log = log
        if bins is None:
            self._current_bins = []
        else:
            self._current_bins = bins
        self._current_events = []

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
        CWD = os.path.dirname(os.path.abspath(__file__))
        conf_file = os.path.join(CWD, "bin_params.yml")
        bin_params_for_camera = OmegaConf.load(conf_file)

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

    def __len__(self):
        return len(self._current_bins)

    def __getitem__(self, index):
        return self._current_bins[index]

    def __iter__(self):
        return iter(self._current_bins)

    def add_bin(self, box, cls, im, safe=True):

        self._bin_count += 1
        label = self._bin_count
        state = cls

        if self._camera in ("cam11", "cam13"):
            # set label based on camera 9
            try:
                # for i_man in range(len(self._manager_prev_cam._left_bins)):

                i_man = 0
                while True:
                    # while len(self._manager_prev_cam._left_bins) > 0:
                    mbin = self._manager_prev_cam._left_bins[i_man]["bin"]
                    mbin_frame = self._manager_prev_cam._left_bins[i_man][
                        "frame_num"]

                    # NOTE: offline mode
                    time_offset = 50 if self._camera == "cam13" else 0

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
                    state = mbin.state

                    # NOTE: don't use label that is already is being used
                    tmp_labs = []
                    for bin in self._current_bins:
                        tmp_labs.append(bin.label)
                    for bin_with_frame in self._left_bins:
                        bin = bin_with_frame["bin"]
                        bin.clear_track()
                        tmp_labs.append(bin.label)
                    if label in tmp_labs:
                        self._manager_prev_cam._left_bins.pop(i_man)
                        # i_man += 1
                        continue
                    break
                else:
                    return
            except IndexError:
                state = "items"
        elif self._camera == "cam13":
            # set label based on camera 11
            try:
                while len(self._manager_prev_cam._left_bins) > 0:
                    mbin = self._manager_prev_cam._left_bins.pop(0)["bin"]
                    mbin_frame = self._manager_prev_cam._left_bins.pop(
                        0)["frame_num"]

                    if mbin_frame - 50 <= self.current_frame:
                        self._manager_prev_cam._left_bins.pop(0)
                        continue

                    label = mbin.label
                    state = mbin.state
                    tmp_labs = []
                    for bin in self._current_bins:
                        tmp_labs.append(bin.label)
                    for bin_with_frame in self._left_bins:
                        bin = bin_with_frame["bin"]
                        tmp_labs.append(bin.label)
                    if label in tmp_labs:
                        continue
                    break
                else:  # else is only executed when there is not break inside while
                    return
            except IndexError:
                state = "items"
                return

        # NOTE: wait for new bin, wait at least 5 iteration to assign
        if safe:
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

        # FIXME why this?
        # if self._camera != "cam09":
        #     self._manager_prev_cam._left_bins.pop(i_man)

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

    def _filter_boxes(self, im, boxes, scores, classes, frame_num):
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

    def _track_current_bins(self, im, boxes, scores, classes, frame_num, ind):
        # ---------------------------- Track previous bin ---------------------------- #
        explored_indices = []
        tmp_iou = {}
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

                # for later use: new bin
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

        return explored_indices, tmp_iou

    def _detect_new_bin(self, im, boxes, scores, classes, frame_num, ind,
                        explored_indices, tmp_iou):
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

    def update_state(self, im, boxes, scores, classes, frame_num):

        self.current_frame = frame_num
        if im is None:
            return

        if classes is None:  # FIXME: ??
            for bin in self._current_bins:
                bin.increment_idle()

        boxes, scores, classes, ind = self._filter_boxes(
            im, boxes, scores, classes, frame_num)

        explored_indices, tmp_iou = self._track_current_bins(
            im, boxes, scores, classes, frame_num, ind)

        self._detect_new_bin(im, boxes, scores, classes, frame_num, ind,
                             explored_indices, tmp_iou)

        # FIXME Add bin results from NU here
        # self._add_nu_multibb(im)

        self._process_exit(im, frame_num)
        return 0  # successful return

    def visualize(self, im):
        return vis_bins(im, self._current_bins)

    def _process_exit(self, im, frame_num):
        _ind = []
        for i in range(len(self)):
            bin = self._current_bins[i]
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
                        self.log.clasp_log(
                            f"{self._camera} : Bin {bin.label} divested")

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

                        else:
                            # REVIEW: check if bin has overlap with other bins, cam09 only
                            flag = True
                            for other_bin in self._current_bins:
                                if other_bin.label != bin.label:
                                    _iou = utils_box.iou_bbox(
                                        bin.pos, other_bin.pos, "combined")

                                    if _iou > 0.5:
                                        # NOTE: In this case, we say that other bin has been
                                        # divested to current bin,
                                        # we left the id of other bin, and retain current bin

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
                                bin.clear_track()
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
                new_bin.clear_track()
                if _type == "exit":
                    self._left_bins.append({
                        "bin": new_bin,
                        "frame_num": frame_num
                    })
                elif _type == "empty":
                    self._empty_bins.append(new_bin)
                self._bin_count = max(self._bin_count, _id)

    # def _add_nu_multibb(self, im):
    #     """
    #     check if there is any nu action

    #     check if nu action bb matches with any bb by iou
    #     if does not match, add nu bb as potential detection
    #     start tracking nu bb
    #     """

    #     if not hasattr(self, "nu_bb"):
    #         self.nu_bb = pd.read_csv("info/results-multibb-nu.txt",
    #                                  header=None)
    #         self.nu_bb = self.nu_bb.iloc[:, [0, 1, 10, 11, 12, 13]].copy()
    #         self.nu_bb = self.nu_bb.drop_duplicates()
    #         self.nu_bb.columns = ['cam', 'frame', 'x1', 'y1', 'x2', 'y2']
    #         # self.nu_bb['frame'] = (self.nu_bb['frame'] *
    #         #                        int(self.config.temporal_scale_mul))
    #         self.nu_bb[['x1', 'x2']] *= self.config.size[0]
    #         self.nu_bb[['y1', 'y2']] *= self.config.size[1]
    #         self.nu_bb = self.nu_bb.sort_values(by='frame')
    #         self.nu_bb.loc[:, 'cam'] = self.nu_bb['cam'].apply(
    #             lambda x: x.replace('cam9', 'cam09'))

    #     # get closest index from current frame
    #     df = self.nu_bb[self.nu_bb['cam'] == self._camera]

    #     idx = np.searchsorted(df['frame'], self.current_frame)
    #     if (idx > 0 and (df.loc[idx, 'frame'] - self.current_frame) <=
    #             self.config.temporal_scale_mul):
    #         row = df[df['frame'] == df.loc[idx, 'frame']]
    #         thresh_iou = 0.3

    #         # import pdb
    #         # pdb.set_trace()

    #         for i, info in row.iterrows():
    #             bbox = info[['x1', 'y1', 'x2', 'y2']].to_numpy()
    #             # check if there's any bin bb close with info bb

    #             found = False
    #             for i in range(len(self)):
    #                 bin = self._current_bins[i]
    #                 iou_value = bin.iou_bbox(bbox, ratio_type="min")

    #                 if iou_value > thresh_iou:
    #                     found = True
    #                     break

    #             if not found:
    #                 # if all iou's are less than thresh
    #                 #   then this bb should be considered for tracking
    #                 # import pdb
    #                 # pdb.set_trace()
    #                 self.add_bin(bbox, "item", im, safe=False)


# ------------------------------ Helper function ----------------------------- #
def recursive_mul(item, scale=1):
    """ recursively multiply all elements in item """
    if isinstance(item, collections.abc.Iterable):
        item = type(item)([recursive_mul(x, scale) for x in item])
        return item
    else:
        return type(item)(item * scale)