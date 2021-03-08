"""python -m pdb coco_format_from_log_gt.py --out_name test --size 1920x1080 --frame_rate []
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import argparse
import os
import parse
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
from PIL import Image
from omegaconf import DictConfig, OmegaConf
import hydra
from utils import read_ata_log_to_df, create_annotation_info, create_image_info
import cv2


def parse_file(fname):
    p = parse.parse('cam{}exp{}-logfile', fname)
    cam = p[0]
    exp = p[1]
    nfname = f'exp{exp}/cam{int(cam):02d}'
    return nfname.replace("training", "train").replace("-", "_")


INFO = {
    "description": "Clasp dataset",
    "version": "0.0.1",
    "year": 2021,
    "contributor": "ashraful",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [{
    "id": 1,
    "name": "Attribution-NonCommercial-ShareAlike License",
    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
}]

CATEGORIES = [{
    "id": 1,
    "name": 'bin_empty',
    "supercategory": 'bin'
}, {
    "id": 2,
    "name": 'item',
    "supercategory": 'bin'
}]


def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, x, y)  # Move it to (x,y)
    cv2.imshow(winname, img)


@hydra.main(config_name='atalog', config_path="conf")
def main(args: DictConfig):
    print(OmegaConf.to_yaml(args, resolve=True))

    Path(args.out).mkdir(exist_ok=True)
    out_name = args.out_name + '_' + ('all' if args.exp is None else '_'.join(
        args.exp)) + '_'.join(args.cam) + '.json'
    out_path = os.path.join(args.out, out_name)

    # concatenate all annotations
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    dict_cat = {x['name']: x['id'] for x in CATEGORIES}

    dict_im_to_id = {}

    counter_id = 1
    counter_im = 1

    if args.test:
        if args.seed:
            np.random.seed(args.seed)

        coco = COCO(out_path)
        # display COCO categories and supercategories
        cats = coco.loadCats(coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        print('COCO categories: \n{}\n'.format(' '.join(nms)))

        catIds = coco.getCatIds(catNms=['item'])
        imgIds = coco.getImgIds(catIds=catIds)
        for _ in range(args.num_test):
            plt.close('all')

            img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

            width, height = img['width'], img['height']

            imfile = Path(args.im_folder) / img['file_name']

            I = cv2.imread(str(imfile))
            # I = cv2.resize(I, (width, height), interpolation=1)

            # load and display instance annotations
            print(imfile)
            annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = coco.loadAnns(annIds)

            for ann in anns:
                bb = ann['bbox']
                bb[2] = bb[0] + bb[2] - 1
                bb[3] = bb[1] + bb[3] - 1
                bb = tuple([int(i) for i in bb])
                cv2.rectangle(I, bb[:2], bb[2:], (0, 255, 0), 5)

            showInMovedWindow(str(imfile), cv2.resize(I, (640, 360)), 30, 30)
            cv2.waitKey(0)
        raise SystemExit

    for exp in args.exp:
        path_dir = Path(args.root) / exp
        ann_files = list(path_dir.iterdir())

        for each_ann_file in tqdm(ann_files):
            basename = Path(str(each_ann_file)).stem
            fcam = parse_file(basename)

            df: pd.DataFrame = read_ata_log_to_df(each_ann_file)

            for _, row in df.iterrows():

                # frame starts from 0, but in the folder it starts from 1
                image_filename_base = os.path.join(fcam,
                                                   eval(args.imfile_format))
                image_filename = os.path.join(args.im_folder,
                                              image_filename_base)
                image = Image.open(image_filename)

                # get image id
                if image_filename_base in dict_im_to_id:
                    image_id = dict_im_to_id[image_filename_base]
                else:
                    image_id = counter_im + 1
                    dict_im_to_id[image_filename_base] = image_id
                    counter_im += 1

                    image_info = create_image_info(image_id,
                                                   image_filename_base,
                                                   image.size)

                    coco_output["images"].append(image_info)

                # filter for associated annotations
                category = 'bin_empty' if row['is_empty'] == 'True' else 'item'
                class_id = dict_cat[category]
                category_info = {'id': class_id, 'is_crowd': 0}

                width, height = image.size
                rat_w, rat_h = 1, 1

                bounding_box = [
                    row['x1'] * rat_w, row['y1'] * rat_h,
                    (row['x2'] - row['x1']) * rat_w,
                    (row['y2'] - row['y1']) * rat_h
                ]
                annotation_info = create_annotation_info(
                    counter_id, image_id, category_info, image.size,
                    bounding_box)
                counter_id += 1

                coco_output["annotations"].append(annotation_info)

    with open(out_path, 'w') as ftarget:
        json.dump(coco_output, ftarget, indent=4)
    print(f"save to {out_path}")


if __name__ == "__main__":
    main()