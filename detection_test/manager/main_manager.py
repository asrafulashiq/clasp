from manager.bin_manager import BinManager
from manager.pax_manager import PAXManager

import logging
import os
import numpy as np

class Manager:
    def __init__(self, log=None, bin_cfg=None, bin_weights=None, pax_cfg=None,
                 pax_weights=None, file_num='9A'):
        self._bin_detector = None
        self._pax_detector = None
        self.file_num = file_num

        self.log = log
        if log is None:
            self.log = logging.getLogger('Manager')
            self.log.setLevel(logging.DEBUG)
            self.log.clasp_log = self.log.info

        # if pax_weights is not None and pax_cfg is not None:
        #     assert os.path.exists(pax_weights), f"{pax_weights} path not found"
        #     assert os.path.exists(pax_cfg), f"{pax_cfg} path not found"
        #     self.init_pax_detector(cfg=pax_cfg, weights=pax_weights)

        # if bin_weights is not None and bin_cfg is not None:
        #     assert os.path.exists(bin_weights), f"{bin_weights} path not found"
        #     assert os.path.exists(bin_cfg), f"{bin_cfg} path not found"
        #     self.init_bin_detector(cfg=bin_cfg, weights=bin_weights)

        self.init_cameras()

        # ! DUMMY
        self._det_bin = {}
        self._det_pax = {}
        for cam in ['9', '11']:
            self.get_dummy_detection_pkl(self.file_num, cam)

    def get_dummy_detection_pkl(self, file_num='9A', camera='9'):
        import pickle
        root = '/home/ash/dataset/clasp_videos/out_pkl/'
        _bin = root + ('bin_' + file_num+'_' + camera + '.pkl')
        _pax = root + ('pax_' + file_num+'_' + camera + '.pkl')
        with open(_bin, 'rb') as fp:
            self._det_bin[camera] = pickle.load(fp)
        with open(_pax, 'rb') as fp:
            self._det_pax[camera] = pickle.load(fp)


    def init_cameras(self):
        # bin_manager in camera 9 and 11
        self._bin_managers = {}
        self._pax_managers = {}

        for camera in ['9', '11']:
            self._bin_managers[camera] = BinManager(camera=camera, log=self.log)
            if self._bin_detector is not None:
                self._bin_managers[camera].detector = self._bin_detector
            if camera == '11':
                self._bin_managers[camera].set_cam9_manager(
                    self._bin_managers['9'])

        for camera in ['9', '11']:
            self._pax_managers[camera] = PAXManager(
                camera=camera, log=self.log)
            if self._pax_detector is not None:
                self._pax_managers[camera].detector = self._pax_detector

    def init_pax_detector(self, cfg=None, weights=None):
        self.log.info("Pax Detector initializing")
        from manager.detector_pax import PAXDetector
        self._pax_detector = PAXDetector(cfg=cfg, weights=weights)

    def init_bin_detector(self, cfg=None, weights=None):
        self.log.info("Bin Detector initializing")
        from manager.detector import BinDetector
        self._bin_detector = BinDetector(cfg=cfg, weights=weights)

    def run_detector_image(self, im=None, cam='9', frame_num=None, return_im=True):

        self.log.addinfo(self.file_num, cam, frame_num)
        if im is None:
            self.log.warning("No image detected")
            return im

        if frame_num >= 2646:
            a = 3

        # get dummy results
        if cam in self._bin_managers:
            if frame_num in self._det_bin[cam]:
                boxes, scores, classes = self._det_bin[cam][frame_num]
                self._bin_managers[cam].update_state(im, boxes, scores, classes)

        if cam in self._pax_managers:
            boxes, scores, classes = self._det_pax[cam][frame_num]
            self._pax_managers[cam].update_state(im, boxes, scores, classes)

        if return_im:
            return self.draw(im, cam=cam)

    def draw(self, im, cam='9'):
        im = self._bin_managers[cam].visualize(im)
        im = self._pax_managers[cam].visualize(im)
        return im
