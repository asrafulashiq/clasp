import cv2
import argparse
import os
from tqdm import tqdm
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feed_folder",
        "-f",
        type=str,
        default="/home/rpi/dataset/ALERT/clasp_data/output/run/exp2_train/feed"
    )
    parser.add_argument("--start_frame", "-s", type=int, default=0)
    args = parser.parse_args()

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 1 * 640, 1 * 480)

    i = args.start_frame
    pbar = tqdm(desc="Frame")

    while True:
        imname = os.path.join(args.feed_folder, f"{i:04d}.jpg")
        if os.path.exists(imname):
            image = cv2.imread(imname)
            cv2.imshow("image", image)
            cv2.waitKey(10)
            i += 1
            pbar.update(n=i + 1)
        else:
            time.sleep(0.5)

    pbar.close()