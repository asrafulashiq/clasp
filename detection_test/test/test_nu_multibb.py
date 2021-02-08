import pandas as pd
import os
import numpy as np
from visutils.vis import vis_bbox
import cv2
from tqdm import tqdm


def main():
    nu_bb = pd.read_csv("info/results-multibb-nu.txt", header=None)
    nu_bb = nu_bb.iloc[:, [0, 1, 10, 11, 12, 13]].copy()
    nu_bb = nu_bb.drop_duplicates()
    nu_bb.columns = ['cam', 'frame', 'x1', 'y1', 'x2', 'y2']
    # nu_bb[['x1', 'x2']] *= config.size[0]
    # nu_bb[['y1', 'y2']] *= config.size[1]
    nu_bb = nu_bb.sort_values(by='frame')
    nu_bb.loc[:, 'cam'] = nu_bb['cam'].apply(
        lambda x: x.replace('cam9', 'cam09'))

    src_folder = os.path.expanduser(
        "~/dataset/ALERT/alert_frames/20191024/fps-10/exp2_train")
    out_folder = os.path.expanduser("~/Desktop/test_nubb")
    os.makedirs(out_folder, exist_ok=True)

    camera = "cam09"
    df = nu_bb[nu_bb['cam'] == camera]
    for frame in tqdm(np.unique(nu_bb["frame"].to_numpy())):
        row = df[df['frame'] == frame]
        imfile = os.path.join(src_folder, camera, f"{frame+1:05d}.jpg")
        assert os.path.exists(imfile)
        image = cv2.imread(imfile)

        for _, info in row.iterrows():
            bbox = info[['x1', 'y1', 'x2', 'y2']].to_numpy()
            image = vis_bbox(image, bbox)

        outfile = os.path.join(out_folder, f"{frame:05d}.jpg")
        cv2.imwrite(outfile, image)


if __name__ == "__main__":
    main()