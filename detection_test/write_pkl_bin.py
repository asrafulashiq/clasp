
import os
import sys
from pathlib import Path
import skimage
from tqdm import tqdm
import tools.utils as utils
from config import conf
import logging
from manager.detector import DummyDetector
import pickle
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

file_num = 'exp2'
cameras = ['cam09']

detector = DummyDetector(ckpt=conf.bin_ckpt, thres=0.2)

for camera in cameras:

    src_folder = Path(conf.root) / file_num / camera
    assert src_folder.exists()

    out_folder = Path(conf.out_dir) / 'out_detection' / file_num / camera / 'bin'
    out_folder.mkdir(parents=True, exist_ok=True)

    pickle_file = Path(conf.out_dir) / 'out_pkl' / \
        ( file_num+'_' + camera + '.pkl')
    pickle_file.parent.mkdir(exist_ok=True)

    _dict = {}

    for im, imfile, frame_num in tqdm(utils.get_images_from_dir(src_folder,
                                                                size=conf.size,
                                                                start_frame=400,
                                                                skip_end=5000, delta=5)):
        logging.info(f'processing : {imfile}')

        new_im, boxes, scores, _class = detector.predict_box(im, show=True)
        _dict[frame_num] = [boxes, scores, _class, imfile]
        if new_im is not None:
            skimage.io.imsave(str(out_folder/imfile.name), new_im)

    with open(str(pickle_file), 'wb') as fp:
        pickle.dump(_dict, fp)