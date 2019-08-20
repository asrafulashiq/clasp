import numpy as np
import torch
import argparse
import os

_HOME = os.environ['HOME']

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="create_dummy")
    parser.add_argument('--root', type=str,
                        default=_HOME+'/dataset/clasp_data/data',
                        help='root direcotory of all frames')
    parser.add_argument("--file-num", type=str, default='9')
    parser.add_argument("--camera", type=str, default='9')

    args = parser.parse_args()
    print(args)

    path_pkl = args.root + ("bin_" + args.file_num + "_" + args.camera + ".pkl")

    