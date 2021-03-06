""" utility functions """
import pathlib
from matplotlib import pyplot as plt
import matplotlib
import cv2
from loguru import logger
from enum import Enum


class Dummy:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self, *args, **kwargs):
        return None

    def __exit__(self, exc_type, exc_val, traceback):
        if exc_type is not None:
            print(exc_type, exc_val)
        return False

    def __getattr__(self, *args, **kwargs):
        return self


def plot_cv(image, axes=None, show=True, fig_number=None):
    """ plot cv2 images  """
    if axes is None:
        axes = plt.axes()
    if len(image.shape) == 3:
        axes.imshow(image[:, :, ::-1])
    else:
        axes.imshow(image)
    if show:
        matplotlib.use("Qt5Agg")
        if not fig_number is None:
            plt.figure(int(fig_number))
        plt.show()
    return axes


def get_images_from_dir(src_dir,
                        size=(640, 360),
                        start_frame=0,
                        skip_end=0,
                        end_frame=None,
                        delta=1,
                        fmt="{:06d}.jpg",
                        file_only=False):
    """ get images as numpy array from a folder"""
    if end_frame is None:
        files_in_dir = sorted(list(src_dir.iterdir()))
        frame_list = range(start_frame, len(files_in_dir) - skip_end, delta)
    else:
        frame_list = range(start_frame, end_frame + 1, delta)

    for frame_num in frame_list:
        imfile = pathlib.Path(src_dir) / fmt.format(frame_num)
        if not imfile.exists():
            logger.info(imfile, "does not exist")
            continue
        if file_only:
            image = None
        else:
            image = cv2.imread(str(imfile))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image,
                               tuple(size),
                               interpolation=cv2.INTER_LINEAR)
        try:
            yield image, imfile, frame_num
        except Exception as ex:
            print(ex)
            raise Exception


def get_fp_from_dir(src_dir,
                    out_folder,
                    start_frame=1,
                    skip_end=0,
                    end_frame=None,
                    delta=1,
                    fmt="{:06d}.jpg"):
    """ get images as numpy array from a folder"""

    if end_frame is None:
        files_in_dir = sorted(list(src_dir.iterdir()))
        frame_list = range(start_frame, len(files_in_dir) - skip_end, delta)
    else:
        frame_list = range(start_frame, end_frame + 1, delta)

    files = []
    for frame_num in frame_list:
        imfile = pathlib.Path(src_dir) / fmt.format(frame_num)
        if not imfile.exists():
            logger.info(imfile, "does not exist")
            continue
        outfile = str(out_folder / imfile.name)
        files.append(outfile)
    return files


class BinType(int, Enum):
    """Custom type for bin and dvi"""
    BIN_EMPTY = 0
    BIN_DVI = 1

    def __str__(self):
        if self == BinType.BIN_EMPTY:
            return "BIN"
        else:
            return "DVI"

    __repr__ = __str__
    name = __str__

    @property
    def super_type(self):
        return "items"