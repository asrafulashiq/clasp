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

log = ClaspLogger()

file_num = "exp2"
cameras = ["cam09"]


manager = Manager(log=log, file_num=file_num, config=conf, bin_only=True)

imlist = []
src_folder = {}
out_folder = {}

for cam in cameras:
    src_folder[cam] = Path(conf.root) / file_num / cam
    assert src_folder[cam].exists()

    out_folder[cam] = Path(conf.out_dir) / "run" / file_num / cam
    out_folder[cam].mkdir(parents=True, exist_ok=True)

    imlist.append(utils.get_images_from_dir(src_folder[cam], skip_init=800, skip_end=3000, delta=5))


for out1 in imlist[0]:
    im, imfile, frame_num = out1

    log.info(f"processing : {frame_num}")
    new_im = manager.run_detector_image(im, cam=cameras[0], frame_num=frame_num)
    # utils.plot_cv(new_im)
    skimage.io.imsave(str(out_folder[cameras[0]] / imfile.name), new_im)

    # im, imfile, frame_num = out2

    # log.info(f'processing : {imfile}')
    # new_im = manager.run_detector_image(
    #     im, cam=cameras[1], frame_num=frame_num)
    # # utils.plot_cv(new_im)
    # skimage.io.imsave(str(out_folder[cameras[1]]/imfile.name), new_im)

# for im, imfile, frame_num in tqdm():
#     logging.info(f'processing : {imfile}')
#     new_im = manager.run_detector_image(im, cam=camera, frame_num=frame_num)
#     # utils.plot_cv(new_im)
#     cv2.imwrite(str(out_folder/imfile.name), new_im)

