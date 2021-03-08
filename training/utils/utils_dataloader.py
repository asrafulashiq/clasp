import os
import sys
import pandas as pd
import parse
import datetime


def read_ata_log_to_df(filename: str) -> pd.DataFrame:
    """read ata log file and convert it to pandas dataframe

    One sample line:
        LOC: type: DVI camera-num: 13 frame: 1400 time-offset: 140.00 \
        BB: 1739, 184, 1917, 452 ID: P1_A PAX-ID: P1 partial-complete: Partial \
        left-behind: False empty: False \
        description: BROWN CYLINDER ROD, STICK AND A BLACK BAG ON IT
    """
    line_format = (
        "LOC: type: {type:w} camera-num: {camera_num:w} frame: {frame:d} time-offset: {time_offset:f} "
        "BB: {x1:d}, {y1:d}, {x2:d}, {y2:d} ID: {bin_id:w} PAX-ID: {pax_id:w} partial-complete: {}"
        "left-behind: {left_behind:w} empty: {is_empty:w} "
        "description:{}")
    line_parse = parse.compile(line_format)

    memory = []
    columns = [
        'frame', 'camera_num', 'time_offset', 'x1', 'y1', 'x2', 'y2',
        'is_empty', 'type'
    ]
    with open(filename, 'r') as f:
        for line in f:
            res = line_parse.parse(line)
            if res and res['type'] == 'DVI':
                memory.append(list(res[x] for x in columns))
    if not memory:
        raise ValueError("No gt here")
    df = pd.DataFrame(memory, columns=columns)
    # df.replace({'is_empty': {'True': True, 'False': False}})
    return df


def create_image_info(image_id,
                      file_name,
                      image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1,
                      coco_url="",
                      flickr_url=""):

    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[0],
        "height": image_size[1],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url
    }

    return image_info


def create_annotation_info(annotation_id,
                           image_id,
                           category_info,
                           image_size=None,
                           bounding_box=None):

    area = image_size[0] * image_size[1]

    if category_info["is_crowd"]:
        is_crowd = 1
    else:
        is_crowd = 0

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": is_crowd,
        "area": int(area),
        "bbox": bounding_box,
        "segmentation": [],
        "width": image_size[0],
        "height": image_size[1],
    }

    return annotation_info