# user-defined imports
# other imports
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import tools.utils as utils
from config import conf
import skimage
import os
import torch
import matplotlib.pyplot as plt

from siammask.tools.test import siamese_init, siamese_track, load_config, load_pretrain
from siammask.siammask_sharp.custom import Custom


def load_siammask():
    class Empty:
        pass

    args = Empty()
    args.resume = "siammask/siammask_sharp/SiamMask_DAVIS.pth"
    args.config = "siammask/siammask_sharp/config_davis.json"

    cfg = load_config(args)

    net = Custom(anchors=cfg["anchors"])
    net = load_pretrain(net, args.resume)
    net.eval().cuda()
    siammask = net
    cfg_siam = cfg
    return siammask, cfg_siam


file_num = "exp2_train"
cameras = ["cam09"]

imlist = []
src_folder = {}
out_folder = {}

#! NOTE: camera 13 is 50 frames behind

# Store image names
for cam in cameras:
    src_folder[cam] = Path(conf.root) / file_num / cam
    assert src_folder[cam].exists()

    if cam == "cam13":
        start_frame = conf.start_frame - 50
    else:
        start_frame = conf.start_frame

    imlist.append(
        utils.get_images_from_dir(src_folder[cam],
                                  start_frame=start_frame,
                                  skip_end=conf.skip_end,
                                  delta=conf.delta,
                                  end_frame=conf.end_frame))

# Process
device = torch.device("cuda:0")

siammask, cfg = load_siammask()
siammask.to(device).eval()

for f, ret in enumerate(tqdm(zip(*imlist))):
    out1, = ret

    # Cam 09
    im, imfile, frame_num = out1
    im = im[..., ::-1]

    if f == 0:  # init
        init_rect = cv2.selectROI('SiamMask', im, False, False)
        x, y, w, h = init_rect
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        state = siamese_init(im,
                             target_pos,
                             target_sz,
                             siammask,
                             cfg['hp'],
                             device=device)  # init tracker
    elif f > 0:  # tracking
        state = siamese_track(state,
                              im,
                              mask_enable=True,
                              refine_enable=True,
                              device=device)  # track
        location = state['ploygon'].flatten()
        mask = state['mask'] > state['p'].seg_thr

        target_pos = state["target_pos"]
        target_sz = state["target_sz"]

        w, h = target_sz
        x, y = target_pos - target_sz / 2

        im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
        im = cv2.polylines(cv2.UMat(im),
                           [np.int0(location).reshape(
                               (-1, 1, 2))], True, (0, 255, 0), 3)
        cv2.imshow('SiamMask', im)
        key = cv2.waitKey(10)
        if key > 0:
            break
