import argparse
import os
import sys


_HOME = os.environ['HOME']


def get_arg():
    """ get command-line arguments """
    parser = argparse.ArgumentParser(description='Arguments for program')
    parser.add_argument('--root', type=str,
                        default=_HOME+'/dataset/clasp/clasp_data/data',
                        help='root direcotory of all frames')
    parser.add_argument('--out_dir', '--out', default=_HOME+'/dataset/clasp/clasp_videos/output',
                        help='output directory')

    parser.add_argument('--bin_detection_wts', type=str,
                        default=_HOME+'/dataset/clasp/trained_model/model.pkl')

    parser.add_argument('--pax_detection_wts', type=str,
                        default=_HOME+'/dataset/clasp/trained_model/model.pkl')

    return parser.parse_args()

conf = get_arg()
