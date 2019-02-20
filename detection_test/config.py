import argparse
import os
import sys


sys.path.append('../detectron')

_HOME = os.environ['HOME']


def get_arg():
    """ get command-line arguments """
    parser = argparse.ArgumentParser(description='Arguments for program')
    parser.add_argument('--root', type=str,
                        default=_HOME+'/dataset/clasp_videos',
                        help='root direcotory of all frames')
    parser.add_argument('--out_dir', '--out', default=_HOME+'/dataset/clasp_videos/output',
                        help='output directory')

    parser.add_argument('--detection_wts', type=str,
                        default='/home/ash/Desktop/clasp/weights/detection-weights' +
                        '/train/clasp_detect/generalized_rcnn/model_final.pkl')
    parser.add_argument('--detection_cfg', type=str,
                        default='/home/ash/Desktop/clasp/detectron/' +
                        'configs/my_cfg/e2e_faster_rcnn_R-50-C4_1x.yaml')

    return parser.parse_args()

conf = get_arg()
