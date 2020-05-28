"""
This script will convert labelling by MU from MOT format to
COCO format
"""

import json
import os
from pathlib2 import Path
import cv2


FPS = {
    "5a":10, "6a":2, "7a":2, "9a":1, "10a":2
}

_HOME = Path(os.environ['HOME'])
# DATA_ROOT = _HOME / 'dataset' / 'clasp_videos'
DATA_ROOT = Path('/media/ash/New Volume/pax_data/')
GT_FOLDER = Path('./GT_MU/')

def downsample(frame_number, orig_fps, final_fps):
    """convert frame number from one fps to another"""
    return int(frame_number/orig_fps * final_fps)


data = {}
data['categories'] = [
    {"id": 1, "name": 'person', "supercategory": 'person'},
    {"id": 2, "name": 'binEMPTY', "supercategory": 'bin'},
    {"id": 3, "name": 'binFULL', "supercategory": 'bin'},
    {"id": 4, "name": 'dvi', "supercategory": 'dvi'}
]

data['annotations'] = []

dict_im = {}

counter_id = 1
counter_im = 1


for item in GT_FOLDER.iterdir():
    try:
        vfile, camnum = item.stem.split('cam')  # '5a', '11'
    except:
        continue

    print item

    if vfile == '5a':
        continue

    frame_dir = Path(vfile.upper()) / ('cam'+camnum)

    with open(str(item), 'r') as fp:
        for line in fp:
            line = line.rstrip()
            line_split = line.split(",")
            frame = int(line_split[0])
            bbox = [int(i) for i in line_split[2:6]]

            # frame_num = downsample(
            #     frame-1, FPS[vfile], 30.0
            # )
            frame_num = frame
            filename = frame_dir / (str(frame_num).zfill(6)+'.png')
            assert (DATA_ROOT/filename).exists(), \
                '{} does not exist'.format((DATA_ROOT/filename))

            if not filename in dict_im:
                dict_im[filename] = counter_im
                counter_im += 1

            tmp = {
                "id": counter_id,
                "image_id": dict_im[filename],
                "category_id": 1,
                "area": bbox[2] * bbox[3],
                'segmentation': [],
                "bbox": bbox,
                "iscrowd": 0,
            }

            data['annotations'].append(tmp)

            # im = cv2.imread(str((DATA_ROOT/filename)))
            # cv2.rectangle(
            #     im, (bbox[0], bbox[1]),
            #     (bbox[0]+bbox[2], bbox[1]+bbox[3]),
            #     (0, 0, 255), 3
            # )
            # Path('./tmp').mkdir(exist_ok=True)
            # cv2.imwrite('./tmp/'+str(counter_id)+'.jpg', im)

            counter_id += 1

data['images'] = []

for fname in dict_im:

    h, w = cv2.imread(str(DATA_ROOT / fname)).shape[:2]

    data['images'].append(
        {
            'file_name': str(fname),
            'id': dict_im[fname],
            'width': w, 'height': h
        }
    )

with open('pax_detection.json', 'w') as ftarget:
    json.dump(data, ftarget)
