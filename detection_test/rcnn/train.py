import torch
import torch.nn as nn
import torchvision
import argparse
import os
# from torchvision.datasets.coco import CocoDetection
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import numpy as np
from pathlib import Path

import model
import coco_utils


def get_transform(train=True, size=(640, 360)):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.Resize(size))
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


if __name__ == "__main__":
    HOME = os.environ["HOME"]
    parser = argparse.ArgumentParser(prog="Detection parser")
    parser.add_argument(
        "--root",
        type=str,
        default=HOME + "/dataset/ALERT/alert_frames",
        help="root directory where the images are downloaded to")

    parser.add_argument("--ann_file",
                        "-a",
                        type=str,
                        default=HOME + "/dataset/ALERT/annotations/ann.json",
                        help="annotation file")
    parser.add_argument("--out_dir",
                        type=str,
                        help="model output directory",
                        default=HOME + "/dataset/ALERT/trained_model")

    parser.add_argument("--out_name",
                        "-o",
                        type=str,
                        help="output model name",
                        default="model.pkl")

    parser.add_argument("--split",
                        type=float,
                        help="train-test split",
                        default=0.9)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=6)

    parser.add_argument("--size", type=str, default="640x360")
    args = parser.parse_args()

    args.size = tuple(int(x) for x in args.size.split("x"))  # width, height
    args.ann_file = os.path.expanduser(args.ann_file)
    args.root = os.path.expanduser(args.root)
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # output file name
    out_dir = Path(args.out_dir)
    file_write_name = Path(args.out_name)
    out_dir.mkdir(exist_ok=True, parents=True)
    out_file = out_dir / file_write_name

    dataset_train = coco_utils.get_coco(args.root,
                                        args.ann_file,
                                        transforms=get_transform(
                                            True, args.size))
    dataset_test = coco_utils.get_coco(args.root,
                                       args.ann_file,
                                       transforms=get_transform(
                                           False, args.size))

    print(dataset_train[100][1])

    # split into train and test
    train_size = int(args.split * len(dataset_train))
    test_size = len(dataset_train) - train_size
    indices = torch.randperm(len(dataset_train)).tolist()

    #! train is the full dataset
    # dataset_train = torch.utils.data.Subset(
    #     dataset_train, indices[:train_size])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[train_size:])
    # define data loader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn)

    # dataset[0]

    device = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')

    model = model.get_model(num_classes=4, size=args.size)
    model.to(device)

    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=50,
                                                   gamma=0.5)
    # start training
    for epoch in range(args.epoch):
        train_one_epoch(model,
                        optimizer,
                        data_loader_train,
                        device,
                        epoch,
                        print_freq=10)
        print("Finished epoch {}".format(epoch))
        lr_scheduler.step()
        torch.save(model.state_dict(), str(out_file))
