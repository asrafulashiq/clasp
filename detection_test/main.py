# user-defined imports
# other imports
from pathlib import Path
import cv2

from tqdm import tqdm
from bin_process.bin_manager import BinManager
import tools.utils as utils
from config import conf
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

file_num = '9A'
camera = '9'

src_folder = Path(conf.root) / file_num / camera
assert src_folder.exists()

out_folder = Path(conf.root) / 'output' / file_num / camera
out_folder.mkdir(parents=True, exist_ok=True)

bin_manager = BinManager(camera=camera, weights=conf.detection_wts,
                         cfg=conf.detection_cfg)


for im, imfile in tqdm(utils.get_images_from_dir(src_folder, skip_init=931,
                                                skip_end=3000, delta=5)):
    logging.info(f'processing : {imfile}')
    new_im = bin_manager.run_detector_on_image(im)
    # utils.plot_cv(new_im)
    cv2.imwrite(str(out_folder/imfile.name), new_im )

cv2.destroyAllWindows()

