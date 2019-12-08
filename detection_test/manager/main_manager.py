from manager.bin_manager import BinManager
from manager.pax_manager import PAXManager

import logging
import os
import numpy as np
from pathlib import Path
import pandas as pd


HOME = os.environ["HOME"]


class Manager:
    def __init__(self, log=None, file_num="exp2", config=None, bin_only=False, write=True):
        self._bin_detector = None
        self._pax_detector = None
        self.file_num = file_num
        self.bin_only = bin_only

        self.config = config
        self.log = log
        if log is None:
            self.log = logging.getLogger("Manager")
            self.log.setLevel(logging.DEBUG)
            self.log.clasp_log = self.log.info

        self.init_cameras()

        # ! DUMMY
        self._det_bin = {}
        self._det_pax = {}
        for cam in ["cam09", "cam11"]:
            self.get_dummy_detection_pkl(self.file_num, cam)

        self.current_frame = None
        self.write = write
        if write:
            self.write_list = []

    def get_info_fram_frame(self, df, frame, cam="cam09"):
        if frame not in df["frame"].values:
            frame += 1
        info = df[(df["frame"] == frame) & (df["camera"] == cam)]
        list_info = []
        for _, row in info.iterrows():
            list_info.append([row["id"], row["class"], row["x1"], row["y1"], row["x2"], row["y2"]])
        return list_info

    def load_info(self, info_file, frame_num, image, camera="cam09"):
        df = pd.read_csv(
            str(info_file),
            sep=",",
            header=None,
            names=["file", "camera", "frame", "id", "class", "x1", "y1", "x2", "y2", "type", "msg"],
            index_col=None,
        )

        list_info = self.get_info_fram_frame(df, frame_num, camera)
        self._bin_managers[camera].add_info(list_info, image)

    def get_dummy_detection_pkl(self, file_num="9A", camera="cam09"):
        import pickle
        import os

        root = self.config.out_dir + "/out_pkl/"
        _bin = root + (file_num + "_" + camera + ".pkl")
        # _pax = root + ("pax_" + file_num + "_" + camera + ".pkl")
        with open(_bin, "rb") as fp:
            self._det_bin[camera] = pickle.load(fp)
        # with open(_pax, 'rb') as fp:
        #     self._det_pax[camera] = pickle.load(fp)

        # pax box from MU data

    def init_cameras(self):
        # bin_manager in camera 9 and 11
        self._bin_managers = {}
        self._pax_managers = {}

        for camera in ["cam09", "cam11"]:
            self._bin_managers[camera] = BinManager(camera=camera, log=self.log)
            # if self._bin_detector is not None:
            #     self._bin_managers[camera].detector = self._bin_detector
            if camera == "cam11":
                self._bin_managers[camera].set_cam9_manager(self._bin_managers["cam09"])

        # FIXME
        # for camera in ["9", "11"]:
        # if not self.bin_only:
        #     for camera in ["cam11"]:
        #         self._pax_managers[camera] = PAXManager(camera=camera, log=self.log)
        #         if self._pax_detector is not None:
        #             self._pax_managers[camera].detector = self._pax_detector

    # def init_pax_detector(self, cfg=None, weights=None):
    #     self.log.info("Pax Detector initializing")
    #     from manager.detector_pax import PAXDetector

    #     self._pax_detector = PAXDetector(ckpt=weights)

    # def init_bin_detector(self, cfg=None, weights=None):
    #     self.log.info("Bin Detector initializing")
    #     from manager.detector import BinDetector

    #     self._bin_detector = BinDetector(ckpt=weights)

    def filter_det(self, ret, class_to_keep="items"):
        boxes, scores, classes, _ = ret
        ind = np.where(classes == class_to_keep)
        if len(ind[0]) == 0:
            return None, None, None
        boxes = boxes[ind]
        scores = scores[ind]
        classes = classes[ind]
        return boxes, scores, classes

    def run_detector_image(self, im=None, cam="cam09", frame_num=None, return_im=True):

        self.current_frame = frame_num
        self.log.addinfo(self.file_num, cam, frame_num)
        if im is None:
            self.log.warning("No image detected")
            return im

        # get dummy results
        if cam in self._bin_managers:
            if frame_num in self._det_bin[cam]:
                boxes, scores, classes, _ = self._det_bin[cam][frame_num]
                self._bin_managers[cam].update_state(im, boxes, scores, classes, frame_num)

        #### FIXME This is temporary
        # if cam in self._pax_managers:
        #     boxes, scores, classes = self._det_pax[cam][frame_num]
        #     self._pax_managers[cam].update_state(im, boxes, scores, classes)
        if not self.bin_only:
            if cam in self._pax_managers:
                self._pax_managers[cam].update_dummy(im, frame_num=frame_num)

        if return_im:
            return self.draw(im, cam=cam)

    def draw(self, im, cam="9"):
        if cam in self._bin_managers:
            im = self._bin_managers[cam].visualize(im)

        if not self.bin_only:
            if cam in self._pax_managers:
                im = self._pax_managers[cam].visualize(im)
        return im

    def write_info(self):
        for cam, bin_manager in self._bin_managers.items():
            for each_bin in bin_manager._current_bins:
                bbox = ",".join(str(int(i)) for i in each_bin.pos)
                line = (
                    f"{self.file_num},{cam},{self.current_frame},{each_bin.label},{each_bin.cls},{bbox},"
                    + f"loc, -1"
                )
                self.write_list.append(line)

    def final_write(self):
        if self.write:
            for cam, bin_manager in self._bin_managers.items():
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

            write_file = Path(self.config.out_dir) / "run" / "info.csv"
            write_file.parent.mkdir(exist_ok=True, parents=True)

            with write_file.open("w") as fp:
                fp.write("\n".join(self.write_list))
