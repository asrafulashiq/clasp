# user-defined imports
import utils
from config import conf
from detector import BinDetector

# other imports
from pathlib import Path
import cv2
from tqdm import tqdm


file_num = '9A'
camera = '9'

src_folder = Path(conf.root) / file_num / camera
assert src_folder.exists()

out_folder = Path(conf.root) / 'output' / file_num / camera
out_folder.mkdir(parents=True, exist_ok=True)


video_out = None

bin_detector = BinDetector(cfg=conf.detection_cfg, weights=conf.detection_wts)

for im, imfile in tqdm(utils.get_images_from_dir(src_folder, skip_init=350,
                                                skip_end=3000, delta=5)):
    new_im, boxes, _class = bin_detector.predict_box(im)
    if video_out is None:
        video_out = cv2.VideoWriter(
            './video.avi', cv2.VideoWriter_fourcc(*"MJPG"), 10, (new_im.shape[1], new_im.shape[0]))
    if not video_out is None:
        video_out.write(new_im)
    # utils.plot_cv(new_im)
    # cv2.imwrite(str(out_folder/imfil e.name), new_im )

cv2.destroyAllWindows()
if not video_out is None:
    video_out.release()
