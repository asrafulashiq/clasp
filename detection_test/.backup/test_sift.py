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

init(autoreset=True)

parser = get_parser()
if os.uname()[1] == 'lambda-server':  # code is in clasp server
    parser = add_server_specific_arg(parser)
conf = get_conf(parser)

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
    kp1, des1 = sift.detectAndCompute(template, None)

    for counter, ret in enumerate(pbar):

        for i_cnt in range(len(ret)):
            im, imfile, frame_num = ret[i_cnt]
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            image = im

            saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            (success, saliencyMap) = saliency.computeSaliency(image)
            # if we would like a *binary* map that we could process for contours,
            # compute convex hull's, extract bounding boxes, etc., we can
            # additionally threshold the saliency map
            threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            img3 = threshMap
            # kp2, des2 = sift.detectAndCompute(im, None)

            # matches = bf.knnMatch(des1, des2, k=2)
            # Sort them in the order of their distance.
            # matches = sorted(matches, key=lambda x: x.distance)
            # good_matches = matches[:10]

            # src_pts = np.float32([kp1[m.queryIdx].pt
            #                       for m in good_matches]).reshape(-1, 1, 2)
            # dst_pts = np.float32([kp2[m.trainIdx].pt
            #                       for m in good_matches]).reshape(-1, 1, 2)
            # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # matchesMask = mask.ravel().tolist()
            # h, w = template.shape[:2]
            # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
            #                   [w - 1, 0]]).reshape(-1, 1, 2)

            # dst = cv2.perspectiveTransform(pts, M)
            # dst += (w, 0)  # adding offset

            # draw_params = dict(
            #     matchColor=(0, 255, 0),  # draw matches in green color
            #     singlePointColor=None,
            #     matchesMask=matchesMask,  # draw only inliers
            #     flags=2)

            # img3 = cv2.drawMatches(template, kp1, im, kp2, good_matches, None,
            #                        **draw_params)

            # # Draw bounding box in Red
            # img3 = cv2.polylines(img3, [np.int32(dst)], True, (0, 0, 255), 3,
            #                      cv2.LINE_AA)

            # Apply ratio test
            # good = []
            # for m, n in matches:
            #     # if m.distance < 0.75 * n.distance:
            #     good.append([m])
            # img3 = cv2.drawMatchesKnn(
            #     template,
            #     kp1,
            #     im,
            #     kp2,
            #     good,
            #     None,
            #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            cv2.imshow("result", img3)
            cv2.waitKey()

            # new_im = cv2.addWeighted(im, 0.3, fgmask, 0.7, 0)
            # skimage.io.imsave(
            #     str((Path("~/Desktop/tmp_folder") / imfile.name).expanduser()),
            #     new_im)

        pbar.set_description(f"Processing: {Fore.CYAN}{frame_num}")
