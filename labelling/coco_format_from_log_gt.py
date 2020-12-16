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


def parse_file(fname):
    p = parse.parse('cam{cam:d}exp{exp:d}{other}', fname)
    cam = p['cam']
    exp = p['exp']
    nfname = f'exp{exp}/cam{cam:02d}'
    return nfname


if __name__ == "__main__":
    HOME = os.environ['HOME']
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default='./alert_log_gt')
    parser.add_argument("--out",
                        type=str,
                        help="save path of json",
                        default="annotations")
    # default=HOME+'/dataset/clasp/clasp_annotations')
    parser.add_argument("--out_name", type=str, default="test")
    parser.add_argument("--im_folder",
                        type=str,
                        default=HOME + '/dataset/ALERT/alert_frames')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--exp", type=str, nargs='*', default=["exp"])
    parser.add_argument("--cam", type=str, nargs='*', default=["cam"])
    parser.add_argument('--size',
                        type=str,
                        default=None,
                        help='image size(width x height)')
    parser.add_argument("--frame_rate", type=int, default=30)
    # parser.add_argument("--ann_files", type=str, nargs="*")
    args = parser.parse_args()

    if args.size is not None:
        args.size = tuple(int(x) for x in args.size.split('x'))
    print(args)

    Path(args.out).mkdir(exist_ok=True)
    out_name = args.out_name + '_' + ('all' if args.exp is None else '_'.join(
        args.exp)) + '_'.join(args.cam) + '.json'
    out_path = os.path.join(args.out, out_name)

    if args.test:
        coco = COCO(out_path)

        # display COCO categories and supercategories
        cats = coco.loadCats(coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        print('COCO categories: \n{}\n'.format(' '.join(nms)))

        catIds = coco.getCatIds(catNms=['items'])
        imgIds = coco.getImgIds(catIds=catIds)
        for _ in range(20):
            plt.close('all')
            img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

            width, height = img['width'], img['height']

            imfile = Path(args.im_folder) / img['file_name']

            I = cv2.imread(str(imfile))
            I = cv2.resize(I, (width, height), interpolation=1)

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

            plt.imshow(I[:, :, ::-1])
            plt.axis('off')
            plt.show()
        raise SystemExit

    # concatenate all annotations
    path_dir = Path(args.root)
    ann_files = list(path_dir.iterdir())

    DVI_LINE = (
        "LOC: type: DVI camera-num: {camera} frame: {frame:d} time-offset:"
        " {time_offset} BB: {x1:d}, {y1:d}, {x2:d}, {y2:d} ID: {bin_id} PAX-ID: {pax_id} "
        "{others}")

    dict_cat = {'passengers': 1, 'items': 2}
    data = {}
    data['categories'] = [{
        "id": 1,
        "name": 'passengers',
        "supercategory": 'passengers'
    }, {
        "id": 2,
        "name": 'items',
        "supercategory": 'items'
    }]

    data['annotations'] = []
    data['images'] = []

    dict_im = {}

    counter_id = 1
    counter_im = 1

    for each_ann_file in tqdm(ann_files):
        basename = Path(str(each_ann_file)).stem
        fcam = parse_file(basename)

        with open(os.path.expanduser(each_ann_file), 'r') as fp:
            for line in fp:
                info = parse.parse(DVI_LINE, line.strip())

                if not info:
                    continue

                wt_fps = args.frame_rate / 30
                frame = f"{int((info['frame']+1) / wt_fps+310):06d}.jpg"
                fname = os.path.join(fcam, frame)

                imfile = os.path.join(args.im_folder, fname)
                h, w, _ = cv2.imread(imfile).shape
                if args.size is None:
                    rat_w, rat_h = 1, 1
                else:
                    rat_w, rat_h = args.size[0] / w, args.size[1] / h

                if not fname in dict_im:
                    dict_im[fname] = [counter_im, w, h]
                    counter_im += 1

                tmp = {
                    "id":
                    counter_id,
                    "image_id":
                    dict_im[fname][0],
                    "category_id":
                    dict_cat['items'],
                    "area":
                    w * h,
                    'segmentation': [],
                    "bbox": [
                        info['x1'] * rat_w, info['y1'] * rat_h,
                        (info['x2'] - info['x1']) * rat_w,
                        (info['y2'] - info['y1']) * rat_h
                    ],
                    "iscrowd":
                    0,
                }
                data['annotations'].append(tmp)
                counter_id += 1

    for k in dict_im:
        data['images'].append({
            'file_name': k,
            'id': dict_im[k][0],
            'width': dict_im[k][1],
            'height': dict_im[k][2]
        })

    with open(out_path, 'w') as ftarget:
        json.dump(data, ftarget, indent=4)
