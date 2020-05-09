""" Main file to run the program """

from pathlib import Path
import cv2
from tqdm import tqdm
import tools.utils as utils
from config import conf
from tools.clasp_logger import ClaspLogger
from manager.main_manager import Manager
import skimage
import os
import torch
from colorama import init, Fore

torch.backends.cudnn.benchmark = True

init(autoreset=True)

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

    manager = Manager(config=conf, log=log, bin_only=True)

    imlist = []
    src_folder = {}
    out_folder = {}

    # format filename
    fmt_filename_src = conf.root + "/{file_num}/{cam}"
    fmt_filename_out = conf.out_dir + "/run/{file_num}/{cam}"

    # store image names
    for cam in cameras:
        src_folder[cam] = Path(
            fmt_filename_src.format(file_num=file_num, cam=cam))
        assert src_folder[cam].exists()

        out_folder[cam] = Path(
            fmt_filename_out.format(file_num=file_num, cam=cam))
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
                                      end_frame=conf.end_frame))

    # Process loop
    pbar = tqdm(zip(*imlist))
    for counter, ret in enumerate(pbar):

        # load already computed info
        if conf.info is not None and os.path.exists(conf.info):
            for i_cnt in range(len(ret)):
                im, imfile, frame_num = ret[i_cnt]
                manager.load_info(conf.info,
                                  frame_num,
                                  im,
                                  camera=cameras[i_cnt])
            conf.info = None
            if conf.write:
                manager.write_info()
            continue

        if conf.load_prev_exit_info:
            # load exit info of previous camera
            for i_cnt in range(len(ret)):
                manager.load_prev_exit_info(
                    conf.info_prev,
                    current_cam=cameras[i_cnt],
                    prev_cam=dict_prev_cam[cameras[i_cnt]])
            conf.load_prev_exit_info = False

        for i_cnt in range(len(ret)):
            im, imfile, frame_num = ret[i_cnt]
            new_im = manager.run_detector_image(im,
                                                cam=cameras[i_cnt],
                                                frame_num=frame_num)
            skimage.io.imsave(str(out_folder[cameras[i_cnt]] / imfile.name),
                              new_im)

        pbar.set_description(f"Processing: {Fore.CYAN}{frame_num}")

        if conf.write:
            manager.write_info()

    if conf.write:
        manager.final_write()
