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
    p = parse.parse('cam{cam:d}{exp}', fname)
    cam = p['cam']
    exp = p['exp']
    nfname = f'{exp}/cam{cam:02d}'
    return nfname


if __name__ == "__main__":
    HOME = os.environ['HOME']
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str,
                        default=HOME+'/dataset/ALERT/annotation')
    parser.add_argument("--out", type=str, help="save path of json", default="annotations")
                        # default=HOME+'/dataset/clasp/clasp_annotations')
    parser.add_argument("--out-name", type=str, default="anns")
    parser.add_argument("--im-folder", type=str,
                        default=HOME+'/dataset/ALERT/alert_frames_2')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--exp", default=None, type=str, nargs='*')
    parser.add_argument("--cam", default=None, type=str, nargs='*')
    parser.add_argument('--size', type=str, default='640x360',
                        help='image size(width x height)')

    args = parser.parse_args()

    args.size = tuple(int(x) for x in args.size.split('x'))
    print(args)

    Path(args.out).mkdir(exist_ok=True)
    out_name = args.out_name + '_' + ('all' if args.exp is None
                                      else '_'.join(args.exp)
                                      ) + '_'.join(args.cam) + '.json'
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
    anns = []
    for json_dir in path_dir.iterdir():
        if args.exp is not None and (json_dir.name in args.exp):
            for file in json_dir.iterdir():
                if file.stem in args.cam:
                    if str(file).endswith('json'):
                        fdata = json.load(file.open())
                        anns.extend(fdata)

    print(anns[0])

    dict_cat = {'passengers': 1, 'items': 2}
    data = {}
    data['categories'] = [
        {"id": 1, "name": 'passengers', "supercategory": 'passengers'},
        {"id": 2, "name": 'items', "supercategory": 'items'}
    ]

    data['annotations'] = []
    data['images'] = []

    dict_im = {}

    counter_id = 1
    counter_im = 1

    for each_ann_file in tqdm(anns):
        fcam = parse_file(each_ann_file['camera'])
        frame = f"{int(each_ann_file['frame']):06d}.jpg"
        fname = os.path.join(fcam, frame)

        if not fname in dict_im:
            dict_im[fname] = counter_im
            counter_im += 1

        imfile = os.path.join(args.im_folder, fname)
        h, w, _ = cv2.imread(imfile).shape

        # for _ann in each_ann_file['passengers']:

        #     rat_w, rat_h = args.size[0] / w, args.size[1] / h

        #     tmp = {
        #         "id": counter_id,
        #         "image_id": dict_im[fname],
        #         "category_id": dict_cat['passengers'],
        #         # "area": _ann['size']['width'] * _ann['size']['height'],
        #         "area": args.size[0] * args.size[1],
        #         'segmentation': [],
        #         "bbox": [
        #             _ann['location']['x']*rat_w, _ann['location']['y']*rat_h,
        #             _ann['size']['width']*rat_w, _ann['size']['height']*rat_h
        #         ],
        #         "iscrowd": 0,
        #     }
        #     data['annotations'].append(tmp)
        #     counter_id += 1

        for _ann in each_ann_file['items']:

            rat_w, rat_h = args.size[0] / w, args.size[1] / h

            tmp = {
                "id": counter_id,
                "image_id": dict_im[fname],
                "category_id": dict_cat['items'],
                "area": args.size[0] * args.size[1],
                'segmentation': [],
                "bbox": [
                    _ann['location']['x']*rat_w, _ann['location']['y']*rat_h,
                    _ann['size']['width']*rat_w, _ann['size']['height']*rat_h
                ],
                "iscrowd": 0,
            }
            data['annotations'].append(tmp)
            counter_id += 1

    for k in dict_im:
        # imfile = os.path.join(args.im_folder, k)
        # h, w, _ = cv2.imread(imfile).shape
        data['images'].append(
            {
                'file_name': k,
                'id': dict_im[k],
                # 'width': w, 'height': h
                'width': args.size[0], 'height': args.size[1]
            }
        )

    with open(out_path, 'w') as ftarget:
        json.dump(data, ftarget)
