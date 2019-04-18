import argparse
import os
import sys


sys.path.append('../detectron')

_HOME = os.environ['HOME']


def get_arg():
    """ get command-line arguments """
    parser = argparse.ArgumentParser(description='Arguments for program')
    parser.add_argument('--root', type=str,
                        default=_HOME+'/dataset/clasp_data/data',
                        help='root direcotory of all frames')
    parser.add_argument('--out_dir', '--out', default=_HOME+'/dataset/clasp_videos/output',
                        help='output directory')

    parser.add_argument('--bin_detection_wts', type=str,
                        default=_HOME+'/clasp/labelling/trained_model_bin' +
                        '/train/clasp_bin/generalized_rcnn/model_final.pkl')
    parser.add_argument('--bin_detection_cfg', type=str,
                        default=_HOME+'/clasp/detectron/configs/my_cfg/' +
                        'faster_R-50-FPN_2x.yaml')

    parser.add_argument('--pax_detection_wts', type=str,
                        default=_HOME+'/clasp/labelling/trained_model_pax' +
                        '/train/clasp_pax/generalized_rcnn/model_final.pkl')
    parser.add_argument('--pax_detection_cfg', type=str,
                        default=_HOME+'/clasp/detectron/configs/my_cfg/' +
                        'pax_faster_R-50-FPN_2x.yaml')


    return parser.parse_args()

conf = get_arg()
