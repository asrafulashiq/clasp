from detectron2.modeling.meta_arch.build import build_model
import skimage.io
from manager.detectron2_utils import DetectorObj, RCNN_Detector
import os
import argparse
import shutil
import cv2
from tqdm import tqdm
import hydra

from detectron2.utils.visualizer import ColorMode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import transforms as T


@hydra.main(config_path="conf", config_name="conf_detector_only")
def main(args):
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)

    # model = RCNN_Detector(args)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.model_zoo))
    cfg.DATASETS.TEST = ()
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.thresh  # set a custom testing threshold
    cfg.MODEL.WEIGHTS = args.bin_ckpt
    predictor = DefaultPredictor(cfg)
    predictor.aug = T.Resize((360, 640))
    model = build_model(cfg)
    DetectionCheckpointer(model).load(args.bin_ckpt)

    list_of_files = sorted(os.listdir(args.input))[::args.delta]

    for i, imfilebase in enumerate(tqdm(list_of_files)):
        imfile = os.path.join(args.input, imfilebase)
        # image = cv2.cvtColor(cv2.imread(str(imfile)), cv2.COLOR_BGR2RGB)
        image = cv2.imread(str(imfile))
        image = cv2.resize(image,
                           tuple(args.size),
                           interpolation=cv2.INTER_LINEAR)
        outputs = predictor(image)
        # outputs = model.predict_one(image)
        v = Visualizer(
            image[:, :, ::-1],
            # scale=0.5,
            instance_mode=ColorMode.
            IMAGE_BW  # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imshow("", out.get_image()[:, :, ::-1])
        # cv2.waitKey(0)

        cv2.imwrite(os.path.join(args.output, f"{i:06d}.jpg"),
                    out.get_image()[:, :, ::-1])


if __name__ == "__main__":
    main()