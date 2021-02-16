""" Test background subtraction algorithm """

from pathlib import Path
import cv2
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

    for counter, ret in enumerate(pbar):

        for i_cnt in range(len(ret)):
            im, imfile, frame_num = ret[i_cnt]
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            im = cv2.GaussianBlur(im, (5, 5), 0)

            fgmask = fgbg.apply(im)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            # fgmask = fgbg.apply(im)
            # new_im = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            new_im = cv2.addWeighted(im, 0.3, fgmask, 0.7, 0)
            skimage.io.imsave(
                str((Path("~/Desktop/tmp_folder") / imfile.name).expanduser()),
                new_im)

        pbar.set_description(f"Processing: {Fore.CYAN}{frame_num}")
