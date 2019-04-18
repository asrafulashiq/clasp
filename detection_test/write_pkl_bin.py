# user-defined imports
# other imports
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

file_num = '10A'
cameras = ['11']

detector = BinDetector(cfg=conf.bin_detection_cfg,
                       weights=conf.bin_detection_wts)

for camera in cameras:

    src_folder = Path(conf.root) / file_num / camera
    assert src_folder.exists()

    out_folder = Path(conf.root) / 'out_pkl' / file_num / camera / 'bin'
    out_folder.mkdir(parents=True, exist_ok=True)

    pickle_file = Path(conf.root) / 'out_pkl' / \
        ('bin_' + file_num+'_'+camera+'.pkl')

    _dict = {}

    for im, imfile, frame_num in tqdm(utils.get_images_from_dir(src_folder, skip_init=1200,
                                                                skip_end=3500, delta=1)):
        logging.info(f'processing : {imfile}')

        new_im, boxes, scores, _class = detector.predict_box(im, show=True)
        _dict[frame_num] = [boxes, scores, _class]
        if boxes is not None:
            import pdb; pdb.set_trace()
        # utils.plot_cv(new_im)
        cv2.imwrite(str(out_folder/imfile.name), new_im)

    with open(str(pickle_file), 'wb') as fp:
        pickle.dump(_dict, fp)

    cv2.destroyAllWindows()
