import json
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

json_dir = Path('./annotations/')
anns = []

for file in json_dir.iterdir():
    if str(file).endswith('json'):
        fdata = json.load(file.open())
        anns.extend(fdata)

print(anns[0])


# dict_cat = {'person': 1, 'binEMPTY': 2, 'binFULL': 3}
dict_cat = {'binEMPTY': 2, 'binFULL': 3}
data = {}
data['categories'] = [
    {"id": 1, "name": 'person', "supercategory": 'person'},
    {"id": 2, "name": 'binEMPTY', "supercategory": 'bin'},
    {"id": 3, "name": 'binFULL', "supercategory": 'bin'}
    # {"id": 4, "name": 'dvi', "supercategory": 'dvi'}
]

data['annotations'] = []

dict_im = {}

counter_id = 1
counter_im = 1



for each_ann_file in anns:
    fname = each_ann_file['filename']
    if not fname in dict_im:
        dict_im[fname] = counter_im
        counter_im += 1

    for _ann in each_ann_file['annotations']:
        if _ann['class'] not in dict_cat:
            continue
        tmp = {
            "id": counter_id,
            "image_id": dict_im[fname],
            "category_id": dict_cat[_ann['class']],
            "area": _ann['width'] * _ann['height'],
            'segmentation': [],
            "bbox": [
                _ann['x'], _ann['y'],
                _ann['width'], _ann['height']
            ],
            "iscrowd": 0,
        }
        data['annotations'].append(tmp)
        counter_id += 1

data['images'] = []

for k in dict_im:
    h, w, _ = cv2.imread(k).shape
    data['images'].append(
        {
            'file_name': k,
            'id': dict_im[k],
            'width': w, 'height': h
        }
    )

with open('bin_detection.json', 'w') as ftarget:
    json.dump(data, ftarget)
