from manager.bin_manager import BinManager
from manager.pax_manager import PAXManager
from manager.detector import Detector

import logging
import os
import numpy as np

class Manager:
    def __init__(self, log=None, cfg=None, weights=None):
        self.log = log
        if log is None:
            self.log = logging.getLogger('Manager')
            self.log.setLevel(logging.DEBUG)

        if weights is not None and cfg is not None:
            assert os.path.exists(weights), f"{weights} path not found"
            assert os.path.exists(cfg), f"{cfg} path not found"
            self.init_detector(cfg=cfg, weights=weights)

    def init_detector(self, cfg=None, weights=None):
        self.log.info("Detector initializing")
        self._detector = Detector(cfg=cfg, weights=weights)

    def run_detector_image(self, im=None, cam='9'):
        if im is None:
            self.log.warning("No image detected")
            return im

        if self._detector is None:
            self.log.error("Detector not initialized")
            return im

        new_im, boxes, scores, classes = self._detector.predict_box(
            im, show=True)

        if boxes is None:
            self.log.info("No bin detected")
            return im

        
