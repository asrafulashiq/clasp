from pathlib2 import Path
import os
import shutil

HOME_DIR = Path('/media/ash/New Volume/shared_folder/all_videos/')
OUT_DIR = Path('./data')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sample_from_folder(folder_name, out_folder, delta=30,
                       skip_initial=0, skip_final=0):
    list_ims = sorted(list(folder_name.iterdir()))


    for i in range(skip_initial, len(list_ims)-skip_final, delta):
        imfile = list_ims[i]
        file_write = OUT_DIR / os.path.sep.join(imfile.parts[-3:-1])
        file_write.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(imfile), str(file_write))


if __name__ == "__main__":
    video_parents = ['7A', '9A']
    camera_names = ['11']

    for vid in video_parents:
        for cam in camera_names:
            src_folder = HOME_DIR / vid / cam
            assert src_folder.exists()
            sample_from_folder(
                src_folder, OUT_DIR,
                delta = 50,
                skip_initial=1200,
                skip_final=400,
            )
