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
cameras = ["cam09", "cam11"]


manager = Manager(log=log, file_num=file_num, config=conf, bin_only=True)


imlist = []
src_folder = {}
out_folder = {}

for cam in cameras:
    src_folder[cam] = Path(conf.root) / file_num / cam
    assert src_folder[cam].exists()

    out_folder[cam] = Path(conf.out_dir) / "run" / file_num / cam
    out_folder[cam].mkdir(parents=True, exist_ok=True)

    imlist.append(
        utils.get_images_from_dir(
            src_folder[cam], skip_init=conf.skip_init,
            skip_end=conf.skip_end, delta=conf.delta,
            end_file=conf.end_file
        )
    )


for out1, out2 in zip(*imlist):

    if conf.info is not None:
        im, imfile, frame_num = out1
        manager.load_info(conf.info, frame_num, im, camera='cam09')

        im, imfile, frame_num = out2
        manager.load_info(conf.info, frame_num, im, camera='cam11')

        conf.info = None
        continue

    im, imfile, frame_num = out1
    log.info(f"processing : {frame_num}")
    new_im = manager.run_detector_image(
        im, cam=cameras[0], frame_num=frame_num
    )
    # utils.plot_cv(new_im)
    skimage.io.imsave(str(out_folder[cameras[0]] / imfile.name), new_im)

    im, imfile, frame_num = out2
    new_im = manager.run_detector_image(
        im, cam=cameras[1], frame_num=frame_num
    )
    skimage.io.imsave(str(out_folder[cameras[1]] / imfile.name), new_im)

    if conf.write:
        manager.write_info()

if conf.write:
    manager.final_write()
