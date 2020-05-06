import argparse
import os
import sys
from os.path import expanduser

_HOME = os.environ["HOME"]


def get_arg():
    """ get command-line arguments """
    parser = argparse.ArgumentParser(description="Arguments for program")
    parser.add_argument(
        "--root",
        type=str,
        default=_HOME + "/dataset/ALERT/alert_frames",
        help="root direcotory of all frames",
    )
    parser.add_argument(
        "--out_dir",
        "--out",
        default=_HOME + "/dataset/ALERT/clasp_data/output",
        help="output directory",
    )

    parser.add_argument(
        "--bin-ckpt",
        type=str,
        default=_HOME + "/dataset/ALERT/trained_model/model.pkl",
    )

    parser.add_argument("--write", action="store_false")

    parser.add_argument(
        "--size",
        type=str,
        default="640x360",
        help="image size(width x height)",
    )

    parser.add_argument('--run-detector',
                        action='store_true',
                        help='whether to run detector or load pre-run results')

    parser.add_argument(
        "--info",
        default=
        "/data/home/islama6/dataset/ALERT/clasp_data/output/run/info.csv",
        # default=None,
        type=str,
        help="info file to save/load",
    )

    parser.add_argument("--start-frame", type=int, default=10580)
    parser.add_argument("--skip-end", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=12000)
    parser.add_argument("--delta", type=int, default=1)
    parser.add_argument("--plot", action="store_true")

    return parser.parse_args()


conf = get_arg()
conf.size = [int(x) for x in conf.size.split("x")]
conf.root = expanduser(conf.root)
conf.out_dir = expanduser(conf.out_dir)
conf.bin_ckpt = expanduser(conf.bin_ckpt)
conf.info = expanduser(conf.info) if conf.info is not None else None