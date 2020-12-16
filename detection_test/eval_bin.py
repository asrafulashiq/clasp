from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage
from pathlib import Path
import argparse
import os
from parse import parse
import pickle
from tools.nms import multi_nms
import cv2

_HOME = os.environ['HOME']


def show_ann(I, anns, color=(0, 255, 0)):
    if anns is None or len(anns) == 0:
        return I
    for ann in anns:
        bbox = ann['bbox']
        x1, y1 = int(bbox[0]), int(bbox[1])
        x2, y2 = x1 + int(bbox[2]), y1 + int(bbox[3])
        cv2.rectangle(I, (x1, y1), (x2, y2), color=color, thickness=5)
    return I

def plot_all(cocoGt, cocoDt, args, out_folder, cat=None):
    imids = cocoDt.getImgIds(catIds=cat)
    imgs = cocoDt.loadImgs(imids)

    for each_im in imgs:
        I = skimage.io.imread(os.path.join(args.root, each_im['file_name']))
        I = cv2.resize(I, args.size, interpolation=cv2.INTER_LINEAR)

        annid_gt = cocoGt.getAnnIds(imgIds=each_im['id'], catIds=cat)
        ann_gt = cocoGt.loadAnns(annid_gt)
        I = show_ann(I, ann_gt, (0, 255, 0))

        annid_det = cocoDt.getAnnIds(imgIds=each_im['id'], catIds=cat)
        ann_det = cocoDt.loadAnns(annid_det)
        I = show_ann(I, ann_det, (0, 0, 255))

        skimage.io.imsave(
            os.path.join(out_folder, Path(each_im['file_name']).name),
            I)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default=_HOME+'/dataset/ALERT/alert_frames',
                        help='root direcotory of all frames')
    parser.add_argument("path", type=str,
                        help="predicted pkl file")
    parser.add_argument("--gt", type=str, required=True,
                        help="ground truth gt")

    parser.add_argument('--out_dir', '--out',
                        default=_HOME+'/dataset/clasp/clasp_data/output',
                        help='output directory')
    parser.add_argument('--size', type=str, default='640x360',
                        help='image size(width x height)')

    parser.add_argument('--plot', action='store_true', help='to plot the images')

    args = parser.parse_args()
    args.size = tuple(int(x) for x in args.size.split('x'))
    print(args)
    coco = COCO(args.gt)  # gt annotation json file

    dict_im = {}
    for _, val in coco.imgs.items():
        dict_im[val['file_name']] = val['id']

    # parse filename and cameraname
    fparse = parse("{}_{exp}_{cam}.pkl", Path(args.path).name)
    exp_name, cam_name = fparse['exp'], fparse['cam']


    if args.plot:
        out_folder = Path(args.out_dir) / 'out_eval' / exp_name / cam_name
        out_folder.mkdir(parents=True, exist_ok=True)

    data = pickle.load(open(args.path, 'rb'))

    data_np = []
    for key, val in data.items():
        full_fname = val[-1]
        short_fname = f'{exp_name}/{cam_name}/{full_fname.name}'
        imid = dict_im[short_fname]

        if val[0] is not None:

            det, scr, cls = multi_nms(val[0], val[1], val[2])
            val = [det, scr.flatten(), cls.flatten()]

            for i in range(len(val[0])):
                cat_id = coco.getCatIds(catNms=[val[2][i]])[0]
                bb = val[0][i]
                data_np.append(
                    [imid,
                     bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1],
                     val[1][i], cat_id]
                )

    data_np = np.array(data_np)
    cocoDt = coco.loadRes(data_np)

    if args.plot:
        plot_all(coco, cocoDt, args, out_folder, cat=2)

    cocoEval = COCOeval(coco, cocoDt, iouType='bbox')
    cocoEval.params.catIds = [2]
    cocoEval.params.imgIds = list(dict_im.values())
    cocoEval.params.iouThrs = np.array([0.3, 0.5, 0.75])
    cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 50 ** 2], [50 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    cocoEval.params.areaRngLbl = ['all', 'small', 'medium', 'large']


    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
