import os
import sys
from pathlib import Path
import skimage
from tqdm import tqdm
import tools.utils as utils
from config import conf
from loguru import logger
from manager.detector import DummyDetector
import pickle
import matplotlib.pyplot as plt

file_num = conf.file_num
cameras = conf.cameras

detector = DummyDetector(ckpt=conf.bin_ckpt, thres=0.3, labels_to_keep=(2, ))

for camera in tqdm(cameras, desc="Camera", position=0):

    src_folder = Path(
        conf.fmt_filename_src.format(file_num=file_num, cam=camera))
    assert src_folder.exists()

    out_folder = Path(
        conf.fmt_filename_out_detection.format(file_num=file_num, cam=camera))
    out_folder.mkdir(parents=True, exist_ok=True)

    pickle_file = Path(
        conf.fmt_filename_out_pkl.format(file_num=file_num, cam=camera))
    pickle_file.parent.mkdir(exist_ok=True)

    _dict = {}  # data

    pbar = tqdm(utils.get_images_from_dir(src_folder,
                                          size=conf.size,
                                          start_frame=1,
                                          skip_end=0,
                                          delta=1,
                                          fmt=conf.fmt),
                position=2)

    for im, imfile, frame_num in pbar:
        pbar.set_description(f"processing : {imfile}")
        new_im, boxes, scores, _class = detector.predict_box(im, show=True)
        _dict[frame_num] = [boxes, scores, _class, imfile]

        if new_im is None:
            new_im = im
        if new_im is not None:
            # logger.info(f"save {imfile.name}")
            skimage.io.imsave(str(out_folder / imfile.name), new_im)

    with open(str(pickle_file), "wb") as fp:
        pickle.dump(_dict, fp)
    logger.info(f"SAVE to: {pickle_file}")
