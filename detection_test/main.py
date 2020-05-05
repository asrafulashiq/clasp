# user-defined imports
# other imports
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

log = ClaspLogger()

file_num = "exp2_test"
cameras = ["cam09"]

manager = Manager(log=log,
                  file_num=file_num,
                  config=conf,
                  bin_only=True,
                  cameras=cameras)

imlist = []
src_folder = {}
out_folder = {}

#! NOTE: camera 13 is 50 frames behind

# Store image names
for cam in cameras:
    src_folder[cam] = Path(conf.root) / file_num / cam
    assert src_folder[cam].exists()

    out_folder[cam] = Path(conf.out_dir) / "run" / file_num / cam
    out_folder[cam].mkdir(parents=True, exist_ok=True)

    if cam == "cam13":
        start_frame = conf.start_frame - 50
    else:
        start_frame = conf.start_frame

    imfiles = utils.get_fp_from_dir(src_folder[cam],
                                    out_folder=out_folder[cam],
                                    start_frame=start_frame,
                                    skip_end=conf.skip_end,
                                    delta=1,
                                    end_frame=conf.end_frame)
    for fp in imfiles[1:]:
        if os.path.exists(fp):
            os.remove(str(fp))

    imlist.append(
        utils.get_images_from_dir(src_folder[cam],
                                  start_frame=start_frame,
                                  skip_end=conf.skip_end,
                                  delta=conf.delta,
                                  end_frame=conf.end_frame))

# Process

for counter, ret in enumerate(tqdm(zip(*imlist))):
    out1, = ret
    # out1, out2, out3 = ret
    if conf.info is not None and os.path.exists(conf.info):
        im, imfile, frame_num = out1
        manager.load_info(conf.info, frame_num, im, camera=cameras[0])

        # im, imfile, frame_num = out2
        # manager.load_info(conf.info, frame_num, im, camera=cameras[1])

        # im, imfile, frame_num = out3
        # manager.load_info(conf.info, frame_num, im, camera=cameras[2])

        conf.info = None
        if conf.write:
            manager.write_info()
        continue

    # Cam 09
    im, imfile, frame_num = out1
    log.info(f"processing : {Fore.CYAN}{frame_num}")
    new_im = manager.run_detector_image(im,
                                        cam=cameras[0],
                                        frame_num=frame_num)
    skimage.io.imsave(str(out_folder[cameras[0]] / imfile.name), new_im)

    # # Cam 11
    # im, imfile, frame_num = out2
    # new_im = manager.run_detector_image(im,
    #                                     cam=cameras[1],
    #                                     frame_num=frame_num)
    # skimage.io.imsave(str(out_folder[cameras[1]] / imfile.name), new_im)

    # # # Cam 13
    # im, imfile, frame_num = out3
    # new_im = manager.run_detector_image(im,
    #                                     cam=cameras[2],
    #                                     frame_num=frame_num)
    # skimage.io.imsave(str(out_folder[cameras[2]] / imfile.name), new_im)

    if conf.write:
        manager.write_info()

if conf.write:
    manager.final_write()
