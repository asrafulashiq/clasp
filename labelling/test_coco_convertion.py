import json
from pathlib import Path
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
import shutil


def add_bbox_ann(I, anns):
    for ann in anns:
        bb = ann['bbox'].copy()
        bb[2] = bb[0] + bb[2] - 1
        bb[3] = bb[1] + bb[3] - 1
        bb = tuple([int(i) for i in bb])
        cv2.rectangle(I, bb[:2], bb[2:], (0, 255, 0), 5)


parser = argparse.ArgumentParser()
parser.add_argument("--ann-file",
                    "-a",
                    type=str,
                    default='annotations/anns_exp1_exp2_traincam9.json')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--data-root",
                    type=str,
                    default=os.environ['HOME'] + '/dataset/ALERT/alert_frames')
parser.add_argument("--write-all", action="store_true")
parser.add_argument("--output", type=str, default="./output_fig")
args = parser.parse_args()

np.random.seed(args.seed)

annFile = args.ann_file
ROOT = Path(args.data_root)
size = (640, 360)

coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['binEMPTY'])
imgIds = coco.getImgIds(catIds=catIds)
#imgIds = coco.getImgIds(imgIds = [324158])
print("START.....")

if not args.write_all:
    while True:
        plt.close('all')
        img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
        # img = coco.loadImgs([80])[0]

        imfile = ROOT / img['file_name']

        I = cv2.imread(str(imfile))
        I = cv2.resize(I, size)

        # load and display instance annotations
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)

        if len(anns) == 0:
            continue

        add_bbox_ann(I, anns)

        plt.imshow(I[..., ::-1])
        plt.title(str(imfile))
        plt.show()
        print(imfile)
else:
    output_dir = Path(args.output)
    if output_dir.exists():
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(exist_ok=True, parents=True)

    for _id in imgIds:
        img = coco.loadImgs(_id)[0]

        imfile = ROOT / img['file_name']

        I = cv2.imread(str(imfile))
        I = cv2.resize(I, size)

        # load and display instance annotations
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)

        if len(anns) == 0:
            continue

        add_bbox_ann(I, anns)
        plt.imsave(str(output_dir / imfile.name), I[..., ::-1])
        print(imfile)