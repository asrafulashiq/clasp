"""
Module for utility functions of bounding box processing
"""

import math

def iou_bbox(bbox1, bbox2, ratio_type='comb'):
    """Intersection of two bounding boxes

    Arguments:
        bbox1 {sequence}
        bbox2 {sequence}

    Keyword Arguments:
        ratio_type {str} --  (default: {'min'})

    Returns:
        iou
    """

    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    area_a = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area_b = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # COMBINED AREA
    if ratio_type == 'min':
        area_combined = area_a #min(area_a, area_b)
    elif ratio_type == 'max':
        area_combined = max(area_a, area_b)
    else:
        area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+1e-5)
    return iou


def dist(x1, y1, x2, y2):
    """ distance between two points  """
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


import numpy as np

def get_min_ind(M):
    r, c = M.shape
    Mc = M.copy()
    M_f = M.flatten()
    ind_sort = np.argsort(-M_f)
    row_processed = []

    out = np.ones(r) * (-1)

    for i in range(ind_sort.shape[0]):
        ind = ind_sort[i]
        cr, cc = np.unravel_index(ind, (r, c))
        if Mc[cr, cc] > 0:
            row_processed.append(cr)
            out[cr] = cc
            Mc[cr, :] = 0
            Mc[:, cc] = 0
    return out