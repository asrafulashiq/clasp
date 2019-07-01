# user-defined imports
# other imports
from pathlib import Path
import cv2
from tqdm import tqdm
import tools.utils as utils
from config import conf
from tools.clasp_logger import ClaspLogger
from manager.main_manager import Manager


log = ClaspLogger()

file_num = '10A'
cameras = ['9', '11']


manager = Manager(log=log, file_num=file_num, bin_cfg=conf.bin_detection_cfg,
                  bin_weights=conf.bin_detection_wts,
                  pax_cfg=conf.pax_detection_cfg,
                  pax_weights=conf.pax_detection_wts,
                  config=conf, bin_only=True)

imlist = []
src_folder = {}
out_folder = {}

for cam in cameras:
    src_folder[cam] = Path(conf.root) / file_num / cam
    assert src_folder[cam].exists()

    out_folder[cam] = Path(conf.root) / 'output' / file_num / cam
    out_folder[cam].mkdir(parents=True, exist_ok=True)

    imlist.append(utils.get_images_from_dir(src_folder[cam], skip_init=350,
                              skip_end=3500, delta=1))


for out1, out2 in tqdm(zip(*imlist)):
    im, imfile, frame_num = out1

    # log.info(f'processing : {imfile}')
    new_im = manager.run_detector_image(im, cam=cameras[0], frame_num=frame_num)
    # utils.plot_cv(new_im)
    cv2.imwrite(str(out_folder[cameras[0]]/imfile.name), new_im)

    im, imfile, frame_num = out2

    # log.info(f'processing : {imfile}')
    new_im = manager.run_detector_image(
        im, cam=cameras[1], frame_num=frame_num)
    # utils.plot_cv(new_im)
    cv2.imwrite(str(out_folder[cameras[1]]/imfile.name), new_im)

# for im, imfile, frame_num in tqdm():
#     logging.info(f'processing : {imfile}')
#     new_im = manager.run_detector_image(im, cam=camera, frame_num=frame_num)
#     # utils.plot_cv(new_im)
#     cv2.imwrite(str(out_folder/imfile.name), new_im)

cv2.destroyAllWindows()

