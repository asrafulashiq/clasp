""" utility functions """
import pathlib
from matplotlib import pyplot as plt
import matplotlib
import cv2
import skimage


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


def get_images_from_dir(
    src_dir, size=(640, 360), skip_init=0, skip_end=0, end_file=None, delta=1,
    fmt="{:06d}.jpg", file_only=False
):
    """ get images as numpy array from a folder"""

    files_in_dir = sorted(list(src_dir.iterdir()))
    if end_file is None:
        frame_list = range(skip_init + 1, len(files_in_dir) - skip_end, delta)
    else:
        frame_list = range(skip_init + 1, end_file, delta)


    for frame_num in frame_list:
        imfile = pathlib.Path(files_in_dir[0].parent) / fmt.format(frame_num)
        if not imfile.exists():
            print(imfile, "does not exist")
            continue
        if file_only:
            image = None
        else:
            image = skimage.io.imread(str(imfile))
            image = cv2.resize(image, tuple(size), interpolation=cv2.INTER_LINEAR)
        yield image, imfile, frame_num


def get_fp_from_dir(
    src_dir, out_folder, skip_init=0, skip_end=0, end_file=None, delta=1,
    fmt="{:06d}.jpg"
):
    """ get images as numpy array from a folder"""

    files_in_dir = sorted(list(src_dir.iterdir()))
    if end_file is None:
        frame_list = range(skip_init + 1, len(files_in_dir) - skip_end, delta)
    else:
        frame_list = range(skip_init + 1, end_file, delta)

    files = []
    for frame_num in frame_list:
        imfile = pathlib.Path(files_in_dir[0].parent) / fmt.format(frame_num)
        if not imfile.exists():
            print(imfile, "does not exist")
            continue
        outfile = str(out_folder / imfile.name)
        files.append(outfile)
    return files