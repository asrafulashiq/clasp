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

file_num = '9A'
cameras = ['9', '11']


out_folder = {}
imlist = []

feed_folder = Path(conf.root) / 'output' / file_num / "feed"
feed_folder.mkdir(exist_ok=True)

# get latest log file
list_of_files = glob.iglob('./logs/*.txt')
logfile = max(list_of_files, key=os.path.getctime)
log_data = pd.read_csv(logfile, header=None, names=[
                       'filenum', 'cam', 'frame', 'msg'])


vis_feed = VisFeed()


for cam in cameras:

    out_folder[cam] = Path(conf.root) / 'output' / file_num / cam

    imlist.append(utils.get_images_from_dir(out_folder[cam]))


for out1, out2 in tqdm(zip(*imlist)):
    im1, imfile1, _ = out1
    im2, imfile2, _ = out2

    frame_num = int(Path(imfile1).stem) - 1

    # get msg
    msglist = log_data[log_data['frame']==frame_num]
    if not msglist.empty:
        msglist = msglist.iloc[:, 1:].values
    else:
        msglist = []

    im_feed = vis_feed.draw(im1, im2, frame_num, msglist)

    f_write = feed_folder / (str(frame_num).zfill(4)+'.jpg')
    cv2.imwrite(str(f_write), im_feed)

cv2.destroyAllWindows()

