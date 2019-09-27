# user-defined imports
# other imports
from pathlib import Path
import cv2
from tqdm import tqdm
import tools.utils as utils
from config import conf
from visutils.vis_feed import VisFeed
import glob, os
import pandas as pd
import skimage
import shutil


file_num = "exp2"
cameras = ["cam09", "cam11"]


out_folder = {}
imlist = []

feed_folder = Path(conf.out_dir) / "run" / file_num / "feed_2"
if feed_folder.exists():
    shutil.rmtree(str(feed_folder))

feed_folder.mkdir(exist_ok=True)

# get latest log file
list_of_files = glob.iglob("./logs/*.txt")
logfile = max(list_of_files, key=os.path.getctime)
log_data = pd.read_csv(
    logfile, header=None, names=["filenum", "cam", "frame", "msg"]
)


vis_feed = VisFeed()

imlist = []
src_folder = {}
out_folder = {}

for cam in cameras:
    src_folder[cam] = Path(conf.out_dir) / "run" / file_num / cam
    assert src_folder[cam].exists()

    imlist.append(
        utils.get_images_from_dir(
            src_folder[cam],
            skip_init=conf.skip_init,
            skip_end=conf.skip_end,
            delta=conf.delta,
        )
    )


for out1, out2 in tqdm(zip(*imlist)):
    im1, imfile1, _ = out1
    im2, imfile2, _ = out2

    frame_num = int(Path(imfile1).stem) - 1

    # get msg
    msglist = log_data[log_data["frame"] == frame_num]
    if not msglist.empty:
        msglist = msglist.iloc[:, 1:].values
    else:
        msglist = []

    im_feed = vis_feed.draw(im1, im2, frame_num, msglist)

    f_write = feed_folder / (str(frame_num).zfill(4) + ".jpg")
    skimage.io.imsave(str(f_write), im_feed)

cv2.destroyAllWindows()

