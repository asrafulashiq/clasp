import logging
import os
import torch
import hydra
from detectron2 import model_zoo
from omegaconf import OmegaConf, DictConfig

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, launch, default_setup, DefaultPredictor


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(args.model_zoo))
    cfg.DATASETS.TRAIN = (args.train_dataset, )
    cfg.DATASETS.TEST = (args.train_dataset, )
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.model_zoo)
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.MAX_ITER = args.max_iters
    cfg.SOLVER.STEPS = (int(args.max_iters / 2), int(args.max_iters * 2 / 3))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.OUTPUT_DIR = args.output_dir

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only is False:
        trainer = DefaultTrainer(cfg)

        trainer.resume_or_load(resume=args.resume)
        trainer.train()

    else:
        cfg.MODEL.WEIGHTS = os.path.join(
            cfg.OUTPUT_DIR,
            "model_final.pth")  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
        # predictor = DefaultPredictor(cfg)
        trainer = DefaultTrainer(cfg)

        evaluator = COCOEvaluator(args.test_dataset, ("bbox", ),
                                  False,
                                  output_dir="./output/")
        val_loader = build_detection_test_loader(cfg, args.test_dataset)
        print(inference_on_dataset(trainer.model, val_loader, evaluator))


@hydra.main(config_name='training', config_path='conf')
def hydra_main(args: DictConfig):
    print("Command Line Args:", args)

    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("clasp_dataset", {}, args.ann_file, args.im_dir)

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