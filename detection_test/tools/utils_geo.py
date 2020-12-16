import numpy as np
from shapely import geometry


def point_in_box(point, box):
    """
        point: (x, y)
        box: [(x, y)] x 4
    """
    point_obj = geometry.Point(point)
    box_obj = geometry.Polygon(box)
    return point_obj.within(box_obj)

