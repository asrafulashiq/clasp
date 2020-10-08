import torch
import argparse
import os
from os.path import expanduser
import shutil
from torchvision.transforms import functional as F
import cv2
import numpy as np
from pathlib import Path

import model as models

import cv2
import skimage
import skimage.io
from tqdm import tqdm


def draw_rect(img, boxes):
    if len(boxes.shape) > 1:
        for i in range(boxes.shape[0]):
            draw_rect(img, boxes[i])
    else:
        pt1 = (int(boxes[0]), int(boxes[1]))
        pt2 = (int(boxes[2]), int(boxes[3]))
        cv2.rectangle(img, pt1, pt2, (0, 0, 1.), 5)
        return img


if __name__ == "__main__":
    HOME = os.environ["HOME"]
    parser = argparse.ArgumentParser(prog="Detection parser")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thres", type=float, default=0.5)
    parser.add_argument("--ckpt",
                        type=str,
                        help="model output directory",
                        default=HOME +
                        "/dataset/clasp/trained_model_bin/model_bin.pkl")
    parser.add_argument("--folder",
                        type=str,
                        help="Image folder",
                        default=HOME +
                        "/dataset/ALERT/alert_frames/exp2/cam09/")
    parser.add_argument("--write-folder",
                        type=str,
                        help="Image folder",
                        default=HOME + "/dataset/ALERT/out_rcnn/")
    parser.add_argument("--size", type=str, default="640x360")
    args = parser.parse_args()
    args.ckpt = expanduser(args.ckpt)
    args.folder = expanduser(args.folder)
    args.write_folder = expanduser(args.write_folder)
    args.size = tuple(int(x) for x in args.size.split("x"))  # width, height
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cpu_device = torch.device("cpu")

    device = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')

    model = models.get_model(num_classes=4)
    model.to(device)

    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(str(args.ckpt)))
    else:
        raise ValueError(f"{args.ckpt} does not exist")
    model.eval()

    # image folder
    assert os.path.isdir(args.folder)
    imfiles = sorted(os.listdir(args.folder))
    write_folder = Path(args.write_folder)
    if write_folder.exists():
        shutil.rmtree(str(write_folder))
    write_folder.mkdir(exist_ok=True)

    for i in tqdm(range(3000, 6800, 100)):
        filename = Path(args.folder) / f"{i:06d}.jpg"
        if filename.exists():
            im = skimage.io.imread(str(filename))
            im = cv2.resize(im, args.size)
            im = skimage.img_as_float32(im)
            imt = F.to_tensor(im)
            with torch.no_grad():
                output = model([imt.to(device)])
            output = {
                k: v.to(cpu_device).data.numpy()
                for k, v in output[0].items()
            }
            index = (((output["labels"] == 2) | (output["labels"] == 3)) &
                     (output["scores"] > args.thres))
            if index.size > 0:
                boxes = output["boxes"][index]
                draw_rect(im, boxes)
                write_filename = write_folder / f"{i:06d}.jpg"
                skimage.io.imsave(str(write_filename),
                                  skimage.img_as_ubyte(im))
