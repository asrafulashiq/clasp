from visutils.vis import vis_bbox
import torch
torch.backends.cudnn.benchmark = True

from manager.bin_manager import BinManager
# from manager.pax_manager import PAXManager
from manager.detectron2_utils import DetectorObj

import pickle
import os
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Tuple, List, Any, Optional
from loguru import logger
from tools.time_calculator import ComplexityAnalysis
from tools.utils import Dummy


class Manager:
    def __init__(
            self,
            config,
            log=None,
            bin_only=True,
            write=True,  # whether to save intermediate results
            template_match=False,
            analyzer: ComplexityAnalysis = None):
        self.file_num = config.file_num
        self.bin_only = bin_only
        self.cameras = config.cameras
        self.config = config
        self.log = log
        self.analyzer = analyzer
        if self.analyzer is None:
            self.analyzer = Dummy()
        if log is None:
            self.log = logger
            self.log.clasp_log = self.log.info

        self.init_cameras()

        self._det_bin = {}

        if not self.config.run_detector:
            # get detection results from pkl
            for cam in self.cameras:
                self.get_dummy_detection_pkl(self.file_num, cam)
        else:
            # init detector
            self.init_detectors()

        self.current_frame = None
        self.write = write
        if write:
            self.write_list = []

        if template_match:
            from tools.template_match_detector import TemplateMatchDetector
            self.empty_bin_template_matches = TemplateMatchDetector()
        else:
            self.empty_bin_template_matches = None

    def get_info_from_frame(self, df, frame, cam="cam09"):
        if frame not in df["frame"].values:
            frame += 1
        info = df[(df["frame"] == frame) & (df["camera"] == cam)]
        list_info = []
        for _, row in info.iterrows():
            list_info.append([
                row["id"], row["class"], row["x1"], row["y1"], row["x2"],
                row["y2"]
            ])
        return list_info

    def get_dummy_detection_pkl(self, file_num="9A", camera="cam09"):
        """ load item detection results from pkl file """
        root = self.config.out_dir + "/out_pkl/"
        _bin = root + (file_num + "_" + camera + ".pkl")
        with open(_bin, "rb") as fp:
            self._det_bin[camera] = pickle.load(fp)

    def init_detectors(self):
        detector = DetectorObj(self.config)
        self._detector = detector

    def init_cameras(self):
        self._bin_managers = {}

        for camera in self.cameras:
            self._bin_managers[camera] = BinManager(self.config,
                                                    camera=camera,
                                                    log=self.log)

            # set previous camera manager
            if camera == "cam11":
                self._bin_managers[camera].set_prev_cam_manager(
                    self._bin_managers.get(
                        "cam09",
                        BinManager(self.config, camera="cam09", log=self.log)))
            elif camera == "cam13":
                self._bin_managers[camera].set_prev_cam_manager(
                    self._bin_managers.get(
                        "cam11",
                        BinManager(self.config, camera="cam11", log=self.log)))
            elif camera == "cam14":
                self._bin_managers[camera].set_prev_cam_manager(
                    self._bin_managers.get(
                        "cam13",
                        BinManager(self.config, camera="cam13", log=self.log)))

    def get_item_bb(self, camera, frame_num,
                    image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ get results from pkl file """
        # boxes: x1, y1, x2, y2
        boxes, scores, classes = None, None, None
        if self.config.run_detector:
            _, boxes, scores, classes = self._detector.predict_box(image,
                                                                   show=False)
        else:
            if frame_num in self._det_bin[camera]:
                boxes, scores, classes, _ = self._det_bin[camera][frame_num]

        return boxes, scores, classes

    def filter_det(self, ret, class_to_keep="items"):
        """ filter detection result """
        boxes, scores, classes, _ = ret
        ind = np.where(classes == class_to_keep)
        if len(ind[0]) == 0:
            return None, None, None
        boxes = boxes[ind]
        scores = scores[ind]
        classes = classes[ind]
        return boxes, scores, classes

    def pre_calculate_detector(self, imlist, return_im=True, max_batch=10):
        if self.config.run_detector:
            info_detector_list = self._detector.predict_box_batch(
                imlist, max_batch=max_batch)
            return info_detector_list

    def run_tracking_per_frame(self,
                               im=None,
                               det=None,
                               cam="cam09",
                               frame_num=None,
                               return_im=True):
        self.current_frame = frame_num
        self.log.addinfo(self.file_num, cam, frame_num)
        if im is None:
            self.log.warning("No image detected")
            return im

        if cam in self._bin_managers:
            _, boxes, scores, classes = det
            self._bin_managers[cam].update_state(im, boxes, scores, classes,
                                                 frame_num)

        if return_im:
            with self.analyzer("DRAW_BIN", False):
                ret = self.draw(im, cam=cam)
            return ret

    def run_detector_image(self,
                           im=None,
                           cam=None,
                           frame_num=None,
                           return_im=True):
        """ main loop """

        self.current_frame = frame_num
        self.log.addinfo(self.file_num, cam, frame_num)
        if im is None:
            self.log.warning("No image detected")
            return im

        # get dummy results
        if cam in self._bin_managers:
            with self.analyzer("DET"):
                boxes, scores, classes = self.get_item_bb(cam, frame_num, im)

                if self.empty_bin_template_matches:
                    _b, _s, _c = self.empty_bin_template_matches.detect(im)
                    if len(_b) > 0:
                        # boxes, scores, classes = _b, _s, _c
                        boxes = np.concatenate((boxes, _b), axis=0)
                        scores = np.concatenate((scores, _s), axis=0)
                        classes = np.concatenate((classes, _c), axis=0)

            # # :: Something wrong with frame 2757 to 2761 of exp1 cam 09
            # if (boxes is not None and self.file_num == "exp1"
            #         and cam == "cam09" and frame_num >= 2757
            #         and frame_num <= 2762):
            #     boxes, scores, classes, _ = self._det_bin[cam][2756]
            #     boxes[[0, 2]] -= 7
            #     for bin in self._bin_managers[cam]._current_bins:
            #         pos = bin.pos
            #         pos[0] -= 7
            #         pos[2] -= 7
            #         bin.pos = pos
            #         if frame_num > 2761:
            #             bin.init_tracker(pos, im)
            # else:

            with self.analyzer("TRACK"):
                self._bin_managers[cam].update_state(im, boxes, scores,
                                                     classes, frame_num)

            if self.config.debug:
                from visutils import vis
                with self.analyzer("DEBUG"):
                    if boxes is not None:
                        for _i in range(len(boxes)):
                            vis_bbox(im,
                                     boxes[_i],
                                     thick=2,
                                     alpha=-1,
                                     color=vis._ORANGE,
                                     filled=False)

        if return_im:
            with self.analyzer("DRAW_BIN", False):
                ret = self.draw(im, cam=cam)
                return ret

    def draw(self, im, cam="cam09"):
        if cam in self._bin_managers:
            im = self._bin_managers[cam].visualize(im)
        return im

    def write_info_upto_frame(self, df, frame, cam="cam09"):
        info = df[(df["frame"] <= frame) & (df["camera"] == cam)]
        for _, row in info.iterrows():
            line = ",".join([str(_s) for _s in row.values])
            self.write_list.append(line)

    def write_exit_info_upto_frame(self, df, frame, cam="cam09"):
        info = df[(df["frame"] <= frame)
                  & ((df["type"] == "exit") | (df["type"] == "empty"))
                  & (df["camera"] == cam)]
        if not info.empty:
            list_info = []
            for _, row in info.iterrows():
                list_info.append([
                    row["id"], row["class"], row["x1"], row["y1"], row["x2"],
                    row["y2"], row["type"], row["frame"]
                ])
            self._bin_managers[cam].add_exit_info(list_info)

    # ----------------------------------- Other ---------------------------------- #
    # --------------------------- Not updated properly --------------------------- #

    def load_prev_exit_info(self,
                            info_file,
                            current_cam="cam11",
                            prev_cam="cam09"):
        names = [
            "file", "camera", "frame", "id", "class", "x1", "y1", "x2", "y2",
            "type", "msg"
        ]
        df = pd.read_csv(
            str(info_file),
            sep=",",
            header=None,
            names=names,
            index_col=None,
        )
        info = df[(df["type"] == "exit") & (df["camera"] == prev_cam)]
        if not info.empty:
            list_info = []
            for _, row in info.iterrows():
                list_info.append([
                    row["id"], row["class"], row["x1"], row["y1"], row["x2"],
                    row["y2"], row["type"], row["frame"]
                ])
            self._bin_managers[current_cam]._manager_prev_cam.add_exit_info(
                list_info)
        self.log.info("Loaded previous exit info")

    def write_info(self):
        """ write a line to the info file """

        for cam, bin_manager in self._bin_managers.items():
            for each_bin in bin_manager._current_bins:
                bbox = ",".join(str(int(i)) for i in each_bin.pos)
                line = (
                    f"{self.file_num},{cam},{bin_manager.current_frame},{each_bin.label},{each_bin.cls},{bbox},"
                    + f"loc, -1")
                self.write_list.append(line)

    def final_write(self):
        for cam, bin_manager in self._bin_managers.items():
            # write all event to the bottom
            for event in bin_manager._current_events:
                frame, _id, cls, x1, y1, x2, y2, _type, msg = event
                line = "{},{},{},{},{},{},{},{},{},{},{}".format(
                    self.file_num,
                    cam,
                    frame,
                    _id,
                    cls,
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                    _type,
                    msg,
                )
                self.write_list.append(line)

        # _name = f"log_info_{self.config.file_num}.csv"
        # write_file = Path(self.config.out_dir) / "run" / "info" / _name
        write_file = Path(self.config.rpi_all_results_csv)
        write_file.parent.mkdir(exist_ok=True, parents=True)

        with write_file.open("w") as fp:
            fp.write("\n".join(self.write_list))

        self.log.info(f"Output info to : {write_file}")

    # Live API log for NU
    def init_data_dict(self):
        self.data_dict_keys = [
            "file", "cam", "frame", "id", "class", "x1", "y1", "x2", "y2",
            "type", "msg"
        ]
        self.data_dict = {k: [] for k in self.data_dict_keys}

    def load_info(self):
        for cam, bin_manager in self._bin_managers.items():
            for each_bin in bin_manager._current_bins:
                bbox = ",".join(str(int(i)) for i in each_bin.pos)
                line = (
                    f"{self.file_num},{cam},{bin_manager.current_frame},{each_bin.label},{each_bin.cls},{bbox},"
                    + f"loc, -1")
                self.write_list.append(line)
                vals = [
                    self.file_num, cam, bin_manager.current_frame,
                    each_bin.label, each_bin.cls,
                    *[3 * int(_k) for _k in bbox.split(',')], "loc", "-1"
                ]
                kvpairs = {k: v
                           for k, v in zip(self.data_dict_keys, vals)}.items()
                self.data_dict = self.append_to_dict(self.data_dict, kvpairs)

    def get_batch_info(self):
        df = pd.DataFrame(self.data_dict)
        return df

    @staticmethod
    def append_to_dict(dictionary, kvpairs):
        for k, v in kvpairs:
            if k in dictionary:
                dictionary[k].append(v)
        return dictionary