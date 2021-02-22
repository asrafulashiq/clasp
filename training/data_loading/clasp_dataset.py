""" Register CLASP dataset """

from collections import defaultdict
from itertools import chain
from typing import Dict, List
from omegaconf import DictConfig
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
import sklearn.model_selection


def register_clasp_dataset(args: DictConfig):
    register_coco_instances(args.dataset, {}, args.ann_file, args.im_dir)
    dataset: List[Dict] = DatasetCatalog.get(args.dataset)

    # image-id to index
    imid_to_index = defaultdict(list)

    for idx, item in enumerate(dataset):
        imid_to_index[item['image_id']].append(idx)

    train_size = int(args.split * len(imid_to_index))
    test_size = len(imid_to_index) - train_size
    id_train, id_test = sklearn.model_selection.train_test_split(
        list(imid_to_index.keys()),
        test_size=test_size,
        train_size=train_size,
        random_state=args.seed)

    indices_train = chain.from_iterable(imid_to_index[_id] for _id in id_train)
    indices_test = chain.from_iterable(imid_to_index[_id] for _id in id_test)

    train_dict = [dataset[index] for index in indices_train]
    test_dict = [dataset[index] for index in indices_test]

    DatasetCatalog.register(args.dataset + "_test", lambda: test_dict)
    DatasetCatalog.register(args.dataset + "_train", lambda: train_dict)

    thing_classes = MetadataCatalog.get(args.dataset).thing_classes
    thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        args.dataset).thing_dataset_id_to_contiguous_id
    for mode in ["_train", "_test"]:
        MetadataCatalog.get(args.dataset + mode).thing_classes = thing_classes

        MetadataCatalog.get(
            args.dataset + mode
        ).thing_dataset_id_to_contiguous_id = thing_dataset_id_to_contiguous_id
