import skimage.io
from manager.detector import DummyDetector
import os
import argparse
import shutil
import cv2
from tqdm import tqdm

if os.uname()[1] == 'lambda-server':
    _HOME = "/home/rpi"
else:
    _HOME = os.environ["HOME"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default=None, required=True)
    parser.add_argument("--output",
                        "-o",
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument("--size", type=str, default="640x360")
    parser.add_argument(
        "--bin_ckpt",
        type=str,
        default=_HOME + "/dataset/ALERT/trained_model/model_cam9_11_13_14.pkl",
    )
    parser.add_argument("--num", "-n", type=int, default=None)
    parser.add_argument("--delta", "-d", type=int, default=1)

    args = parser.parse_args()
    args.size = tuple(int(x) for x in args.size.split('x'))

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)

    detector = DummyDetector(ckpt=args.bin_ckpt,
                             thres=0.3,
                             labels_to_keep=(2, ))
    list_of_files = sorted(os.listdir(args.input))[::args.delta]

    for i, imfilebase in enumerate(tqdm(list_of_files)):
        imfile = os.path.join(args.input, imfilebase)
        image = cv2.cvtColor(cv2.imread(str(imfile)), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,
                           tuple(args.size),
                           interpolation=cv2.INTER_LINEAR)
        im_with_bb, boxes, scores, classes = detector.predict_box(image,
                                                                  show=True)
        skimage.io.imsave(os.path.join(args.output, imfilebase), im_with_bb)

        if args.num is not None and i >= args.num:
            break
