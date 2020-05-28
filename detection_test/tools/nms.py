import numpy as np


def multi_nms(dets, scores, classes, thresh=0.1, low_score=0.5):
    uniq_classes = np.unique(classes)

    new_dets = []
    new_scores = []
    new_classes = []
    for each_class in uniq_classes:
        ind = np.where(classes == each_class)

        _det = dets[ind]
        _scr = scores[ind]
        _cls = classes[ind]
        keep = nms(_det, _scr, thresh=thresh)
        # keep = [i for i in _keep if _scr[i] > low_score]

        new_dets.append(_det[keep])
        new_scores.append(_scr[keep])
        new_classes.append(_cls[keep])

    return np.vstack(new_dets), np.vstack(new_scores).flatten(), \
        np.vstack(new_classes).flatten()


def nms(dets, scores, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep