
import os, sys
from pathlib import Path
import cv2
from tqdm import tqdm
import tools.utils as utils
from config import conf
import logging
from manager.detector import BinDetector
import pickle
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

file_num = '9A'
cameras = ['9']

detector = BinDetector(ckpt=conf.bin_detection_wts, thres=0.1)

for camera in cameras:

    src_folder = Path(conf.root) / file_num / camera
    assert src_folder.exists()

    out_folder = Path(conf.root) / 'out_detection' / file_num / camera / 'bin'
    out_folder.mkdir(parents=True, exist_ok=True)

    pickle_file = Path(conf.root) / 'out_pkl' / \
        ('bin_' + file_num+'_'+camera+'.pkl')

    _dict = {}

    for im, imfile, frame_num in tqdm(utils.get_images_from_dir(src_folder, skip_init=800,
                                                                skip_end=3000, delta=10)):
        logging.info(f'processing : {imfile}')

        new_im, boxes, scores, _class = detector.predict_box(im, show=True)
        _dict[frame_num] = [boxes, scores, _class]
        cv2.imwrite(str(out_folder/imfile.name), new_im)

    with open(str(pickle_file), 'wb') as fp:
        pickle.dump(_dict, fp)

    cv2.destroyAllWindows()
