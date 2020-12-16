"""This script is responsible for managing passengers in a camera"""

import logging
import os
import numpy as np
import cv2

from pax_process.person import Person
from visutils.vis import vis_pax
import tools.MU_utils as mutils


HOME = os.environ["HOME"]

class PAXManager:
    """ passenger manager """

    def __init__(self, pax=None, log=None, detector=None, camera="9"):
        self.log = log
        if log is None:
            self.log = logging.getLogger("pax manager")
            self.log.setLevel(logging.DEBUG)
            self.log.clasp_log = self.log.info

        if pax is None:
            self._current_pax = []
        else:
            self._current_pax = pax
        self._camera = camera

        # initialize configuration
        if camera == "9":
            self.init_cam_9()

        # FIXME : temporary
        if camera == "11":
            self.dummy_pax_from_MU()

        self.detector = detector

    @property
    def detector(self):
        return self._detector

    @detector.setter
    def detector(self, _det):
        self._detector = _det

    def init_cam_9(self):
        self._left_pax = []
        self._min_iou = 0.4
        self._pax_count = 0
        self._thres_incoming_pax_exit = 460  # x
        self._thres_out_pax_exit = 350
        self._thres_incoming_pax_init_x = 1420

    def __len__(self):
        return len(self._current_pax)

    def __getitem__(self, index):
        return self._current_pax[index]

    def __iter__(self):
        return iter(self._current_pax)

    def add_pax(self, box):
        #! Just for now
        self._pax_count = 0

        self._pax_count += 1
        new_pax = Person(label=self._pax_count, pos=box)
        self._current_pax.append(new_pax)
        # self.log.clasp_log(f"Person {self._pax_count} enters")

    def dummy_pax_from_MU(self):
        boxs = np.loadtxt(
            HOME + "/dataset/det10ACam11PersonMASK_30FPS_cluster_score", delimiter=","
        )

        boxs = mutils.temporal_format(boxs)
        boxs = boxs[np.where(boxs[:, 6] >= 0.4), :][0]

        self.pax_boxs = boxs[np.where(boxs[:, 7] == 1), :][0]
        # format: [frame,ins_ind,x,y,w,h,score,class_id,angle]
        self.det_cluster_id = []
        self.pax_boxs_window = []
        self.ID_ind = 1

    def update_dummy(self, im, frame_num):

        # parameters for tracker
        time_lag = 5
        score_th = 0.15
        min_cluster_size = 3
        iou_thr = 0.1
        ID_ind = self.ID_ind


        fr = frame_num + 64
        pax_boxs = self.pax_boxs
        det_cluster_id = self.det_cluster_id
        pax_boxs_window = self.pax_boxs_window

        fr_pax_box = pax_boxs[np.where(pax_boxs[:, 0] == fr), :][0]

        if len(fr_pax_box) > 0:
            im_h, im_w, _ = im.shape
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if len(det_cluster_id) > 0:
                pax_boxs_prev = np.vstack([det_cluster_id, fr_pax_box])
            else:
                pax_boxs_window.append(fr_pax_box)
                pax_boxs_prev, _ = mutils.expand_from_temporal_list(
                    pax_boxs_window, None
                )
            # wait for loop back frames
            if fr >= pax_boxs_prev[:, 0][0] + time_lag - 1:
                det_cluster_id, bbox, scores, labels, self.ID_ind = \
                    mutils.tracking_temporal_clustering(
                            fr,
                            pax_boxs_prev,
                            time_lag,
                            min_cluster_size,
                            iou_thr,
                            det_cluster_id,
                            ID_ind,
                            score_th,
                            None, # ax
                            im_h,
                            im_w,
                    )
                if len(bbox) > 0:
                    self._current_pax = []
                    for pi in range(len(bbox)):
                        new_pax = Person(label=labels[pi], pos=bbox[pi])
                        self._current_pax.append(new_pax)

        self.pax_boxs_window = pax_boxs_window
        self.det_cluster_id = det_cluster_id
        return

    def update_state(self, im, boxes, scores, classes):

        if im is None:
            return None

        if classes is None:
            return

        ind = [i for i in range(len(classes)) if classes[i] == "person"]

        if len(ind) > 0:
            boxes, scores, classes = boxes[ind], scores[ind], classes[ind]
        else:
            boxes, scores, classes = None, None, None

        explored_indices = []

        tmp_iou = {}

        # if len(ind) > 0:
        #     for pax in self._current_pax:
        #         iou_to_boxes = []
        #         for _counter in range(boxes.shape[0]):
        #             _iou = pax.iou_bbox(boxes[_counter])
        #             iou_to_boxes.append(_iou)
        #             tmp_iou[_counter] = max(tmp_iou.get(_counter, 0), _iou)

        #         closest_index = np.argmax(iou_to_boxes)
        #         if closest_index in explored_indices:
        #             continue

        #         if iou_to_boxes[closest_index] > self._min_iou:
        #             pax.pos = boxes[closest_index]
        #             pax.state = classes[closest_index]
        #             explored_indices.append(closest_index)

        # # refine detected pax and detect new pax
        # if len(ind) > 0:
        #     for i in range(boxes.shape[0]):
        #         if i in explored_indices:
        #             continue
        #         # if new pax is closest to a current pax
        #         # set the current pax as new pax
        #         box, _, cls = boxes[i], scores[i], classes[i]

        #         self.add_pax(box)  # add new pax

        #! Just for now
        if len(ind) > 0:
            self._current_pax = []
            for i in range(boxes.shape[0]):
                if i in explored_indices:
                    continue
                # if new pax is closest to a current pax
                # set the current pax as new pax
                box, _, cls = boxes[i], scores[i], classes[i]

                self.add_pax(box)  # add new pax

        # detect pax exit
        # self.process_exit()
        return 0

    def visualize(self, im):
        return vis_pax(im, self._current_pax)

    def process_exit(self):
        _ind = []
        for i in range(len(self)):
            pax = self._current_pax[i]
            if pax.centroid[0] < self._thres_out_pax_exit:
                # pax exit
                self.log.clasp_log(f"Person {pax.label} exits")
                self._left_pax.append(pax)
            else:
                _ind.append(i)
        self._current_pax = [self._current_pax[i] for i in _ind]
