import cv2
import numpy as np

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)
_RED = (255, 10, 10)
_BLUE = (10, 10, 255)


class_to_color = {
    'passengers': _BLUE,
    'items': _GREEN,
    'other': _RED
}

def vis_class_label(img, pos, class_str, label, font_scale=0.5):
    """Visualizes the class."""
    img = img.astype(np.uint8)
    x0, y0 = int(pos[0]), int(pos[1])
    x1, y1 = int(pos[2]), int(pos[3])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0

    color = class_to_color.get(class_str, _RED)
    cv2.rectangle(img, back_tl, back_br, color, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale,
                _GRAY, lineType=cv2.LINE_AA, thickness=1)

    # put label
    if label:
        ((lbl_w, lbl_h), _) = cv2.getTextSize(str(label), font, 3*font_scale, 5)
        lbl_tl = int((x0+x1)/2) - int(0.3 * lbl_w), int((y0+y1)/2) - \
                    int(0 * lbl_h)
        cv2.putText(img, str(label), lbl_tl, font, 3*font_scale,
                    color, thickness=2, lineType=cv2.LINE_AA)

    return img


def vis_bbox(img, bbox, class_str=None, thick=2):
    """Visualizes a bounding box."""
    # img = img.astype(np.uint8)
    (x0, y0, x1, y1) = bbox
    x0, y0 = int(x0), int(y0)
    x1, y1 = int(x1), int(y1)
    color = class_to_color.get(class_str, _RED)

    overlay = img.copy()
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=thick)

    alpha = 0.2
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)  # filled rectangle

    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

    return img


def vis_bins(img, bins):
    for bin in bins:
        bbox = bin.pos
        cls = bin.cls
        label = bin.label
        img = vis_bbox(img, bbox, class_str=cls)
        img = vis_class_label(img, bbox, cls, label)
    return img


def vis_pax(img, pax):
    for person in pax:
        bbox = person.pos
        label = person.label
        img = vis_bbox(img, bbox, class_str=person.cls)
        img = vis_class_label(img, bbox, 'passengers', label)
    return img
