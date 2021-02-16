""" Test background subtraction algorithm """

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import tools.utils as utils
from config import get_conf, get_parser, add_server_specific_arg
from tools.clasp_logger import ClaspLogger
from manager.main_manager import Manager
import skimage
import os
import torch
from colorama import init, Fore
import cv2 as cv
import matplotlib.pyplot as plt

init(autoreset=True)

parser = get_parser()
if os.uname()[1] == 'lambda-server':  # code is in clasp server
    parser = add_server_specific_arg(parser)
conf = get_conf(parser)


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh=0.5):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


if __name__ == "__main__":
    log = ClaspLogger()

    file_num = conf.file_num
    cameras = conf.cameras

    dict_prev_cam = {
        "cam09": None,
        "cam11": "cam09",
        "cam13": "cam11",
        "cam14": "cam13"
    }

    imlist = []
    src_folder = {}
    out_folder = {}

    # store image names
    for cam in cameras:
        src_folder[cam] = Path(
            conf.fmt_filename_src.format(file_num=file_num, cam=cam))
        assert src_folder[cam].exists()

        out_folder[cam] = Path(
            conf.fmt_filename_out.format(file_num=file_num, cam=cam))
        out_folder[cam].mkdir(parents=True, exist_ok=True)

        if cam == "cam13":
            # NOTE: camera 13 is 50 frames behind (approx)
            start_frame = conf.start_frame - 50
        else:
            start_frame = conf.start_frame

        imfiles = utils.get_fp_from_dir(src_folder[cam],
                                        out_folder=out_folder[cam],
                                        start_frame=start_frame,
                                        skip_end=conf.skip_end,
                                        delta=1,
                                        fmt=conf.fmt,
                                        end_frame=conf.end_frame)

        # remove future frames
        for fp in imfiles[1:]:
            if os.path.exists(fp):
                os.remove(str(fp))

        imlist.append(
            utils.get_images_from_dir(src_folder[cam],
                                      start_frame=start_frame,
                                      skip_end=conf.skip_end,
                                      delta=conf.delta,
                                      end_frame=conf.end_frame,
                                      fmt=conf.fmt))

    # Process loop
    pbar = tqdm(zip(*imlist))

    fgbg = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=40,
                                                    useHistory=True,
                                                    maxPixelStability=500)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=40)

    sift = cv2.ORB_create()
    bf = cv2.BFMatcher()

    template = cv2.imread(os.path.expanduser("~/Desktop/template.png"))
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.resize(template,
                          dsize=(0, 0),
                          fx=1. / 3,
                          fy=1. / 3,
                          interpolation=cv2.INTER_CUBIC)
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    # methods = [
    #     'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
    #     'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'
    # ]
    methods = ['cv.TM_CCOEFF_NORMED']

    for counter, ret in enumerate(pbar):

        for i_cnt in range(len(ret)):
            im_rgb, imfile, frame_num = ret[i_cnt]
            im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
            im = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)

            for meth in methods:
                method = eval(meth)
                # Apply template Matching
                res = cv2.matchTemplate(im, template, method)

                threshold = 0.5
                loc = np.where(res >= threshold)

                boxes = np.array([(*pt, pt[0] + w, pt[1] + h)
                                  for pt in zip(*loc[::-1])])
                boxes = non_max_suppression_fast(boxes)

                for box in boxes:
                    im_rgb = cv.rectangle(im_rgb, (box[0], box[1]),
                                          (box[2], box[3]), (0, 0, 255), 3)

                # cv2.imshow("result", im_rgb)
                # cv2.waitKey(0)
                cv2.imwrite(
                    str((Path("~/Desktop/tmp_folder") /
                         imfile.name).expanduser()), im_rgb)

            # new_im = cv2.addWeighted(im, 0.3, fgmask, 0.7, 0)
            # skimage.io.imsave(
            #     str((Path("~/Desktop/tmp_folder") / imfile.name).expanduser()),
            #     new_im)

        pbar.set_description(f"Processing: {Fore.CYAN}{frame_num}")
