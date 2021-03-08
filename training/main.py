from itertools import count
import logging
import os
from typing import Dict, List
from detectron2.data.catalog import MetadataCatalog
import torch
import hydra
from detectron2 import model_zoo
from omegaconf import OmegaConf, DictConfig

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluators
from detectron2.data import build_detection_test_loader
from detectron2.data import (DatasetCatalog, DatasetMapper,
                             build_detection_train_loader,
                             build_detection_test_loader)

from detectron2.engine import DefaultTrainer, launch, default_setup, DefaultPredictor
from detectron2.data import transforms as T

from data_loading import clasp_dataset
from utils import visualize_det2, create_train_augmentation, create_test_augmentation


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg,
                               is_train=True,
                               augmentations=create_train_augmentation(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DatasetMapper(cfg,
                               is_train=False,
                               augmentations=create_test_augmentation(cfg))
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, ("bbox", ),
                             False,
                             output_dir=output_folder)


class Predictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.aug = T.Resize((cfg.image_h, cfg.image_w))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(args.model_zoo))
    cfg.DATASETS.TRAIN = (args.train_dataset, )
    cfg.DATASETS.TEST = (args.test_dataset, )
    cfg.DATALOADER.NUM_WORKERS = args.num_workers

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.OUTPUT_DIR = args.output_dir
    cfg.image_w = args.size[0]
    cfg.image_h = args.size[1]

    if args.eval_only is False:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.model_zoo)
        cfg.SOLVER.IMS_PER_BATCH = args.batch_size
        cfg.SOLVER.BASE_LR = args.learning_rate
        cfg.SOLVER.MAX_ITER = args.max_iters
        cfg.SOLVER.WARMUP_ITERS = int(args.max_iters / 10)
        cfg.SOLVER.STEPS = (int(args.max_iters / 2),
                            int(args.max_iters * 2 / 3))
    else:
        cfg.MODEL.WEIGHTS = os.path.join(
            cfg.OUTPUT_DIR,
            "model_final.pth")  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only is False:
        trainer = Trainer(cfg)

        trainer.resume_or_load(resume=args.resume)
        trainer.train()

    else:
        if args.visualize is False:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model,
                                  save_dir=cfg.OUTPUT_DIR).resume_or_load(
                                      cfg.MODEL.WEIGHTS, resume=False)
            res = Trainer.test(cfg, model)
            print(res)
            return res

        else:
            test_dataset: List[Dict] = DatasetCatalog.get(args.test_dataset)
            metadata = MetadataCatalog.get(args.test_dataset)
            predictor = Predictor(cfg)
            visualize_det2(test_dataset,
                           predictor,
                           metadata=metadata,
                           count=args.num_items)


@hydra.main(config_name='configs', config_path='conf')
def hydra_main(args: DictConfig):
    print("Command Line Args:", args)
    clasp_dataset.register_clasp_dataset(args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, ),
    )


if __name__ == "__main__":
    hydra_main()