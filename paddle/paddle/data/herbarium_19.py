import os
from copy import deepcopy

import numpy as np
import paddle

from config import herbarium_dataroot
from data.data_utils import subsample_instances


class HerbariumDataset19(paddle.vision.datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        uq_idx = self.uq_idxs[idx]
        return img, label, uq_idx


def subsample_dataset(dataset, idxs):
    mask = np.zeros(len(dataset)).astype("bool")
    mask[idxs] = True
    dataset.samples = np.array(dataset.samples)[mask].tolist()
    dataset.targets = np.array(dataset.targets)[mask].tolist()
    dataset.uq_idxs = dataset.uq_idxs[mask]
    dataset.samples = [[x[0], int(x[1])] for x in dataset.samples]
    dataset.targets = [int(x) for x in dataset.targets]
    return dataset


def subsample_classes(dataset, include_classes=range(250)):
    cls_idxs = [x for x, l in enumerate(dataset.targets) if l in include_classes]
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i
    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]
    return dataset


def get_train_val_indices(train_dataset, val_instances_per_class=5):
    train_classes = list(set(train_dataset.targets))
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]
        v_ = np.random.choice(cls_idxs, replace=False, size=(val_instances_per_class,))
        t_ = [x for x in cls_idxs if x not in v_]
        train_idxs.extend(t_)
        val_idxs.extend(v_)
    return train_idxs, val_idxs


def get_herbarium_datasets(
    train_transform,
    test_transform,
    train_classes=range(500),
    prop_train_labels=0.8,
    seed=0,
    split_train_val=False,
):
    np.random.seed(seed)
    train_dataset = HerbariumDataset19(
        transform=train_transform, root=os.path.join(herbarium_dataroot, "small-train")
    )
    train_dataset_labelled = subsample_classes(
        deepcopy(train_dataset), include_classes=train_classes
    )
    subsample_indices = subsample_instances(
        train_dataset_labelled, prop_indices_to_subsample=prop_train_labels
    )
    train_dataset_labelled = subsample_dataset(
        train_dataset_labelled, subsample_indices
    )
    if split_train_val:
        train_idxs, val_idxs = get_train_val_indices(
            train_dataset_labelled, val_instances_per_class=5
        )
        train_dataset_labelled_split = subsample_dataset(
            deepcopy(train_dataset_labelled), train_idxs
        )
        val_dataset_labelled_split = subsample_dataset(
            deepcopy(train_dataset_labelled), val_idxs
        )
        val_dataset_labelled_split.transform = test_transform
    else:
        train_dataset_labelled_split, val_dataset_labelled_split = None, None
    unlabelled_indices = set(train_dataset.uq_idxs) - set(
        train_dataset_labelled.uq_idxs
    )
    train_dataset_unlabelled = subsample_dataset(
        deepcopy(train_dataset), np.array(list(unlabelled_indices))
    )
    test_dataset = HerbariumDataset19(
        transform=test_transform,
        root=os.path.join(herbarium_dataroot, "small-validation"),
    )
    unlabelled_classes = list(set(train_dataset.targets) - set(train_classes))
    target_xform_dict = {}
    for i, k in enumerate(list(train_classes) + unlabelled_classes):
        target_xform_dict[k] = i
    test_dataset.target_transform = lambda x: target_xform_dict[x]
    train_dataset_unlabelled.target_transform = lambda x: target_xform_dict[x]
    train_dataset_labelled = (
        train_dataset_labelled_split if split_train_val else train_dataset_labelled
    )
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None
    all_datasets = {
        "train_labelled": train_dataset_labelled,
        "train_unlabelled": train_dataset_unlabelled,
        "val": val_dataset_labelled,
        "test": test_dataset,
    }
    return all_datasets


if __name__ == "__main__":
    np.random.seed(0)
    train_classes = np.random.choice(range(683), size=int(683 / 2), replace=False)
    x = get_herbarium_datasets(
        None, None, train_classes=train_classes, prop_train_labels=0.5
    )
    assert set(x["train_unlabelled"].targets) == set(range(683))
    print("Printing lens...")
    for k, v in x.items():
        if v is not None:
            print(f"{k}: {len(v)}")
    print("Printing labelled and unlabelled overlap...")
    print(
        set.intersection(
            set(x["train_labelled"].uq_idxs), set(x["train_unlabelled"].uq_idxs)
        )
    )
    print("Printing total instances in train...")
    print(
        len(set(x["train_labelled"].uq_idxs)) + len(set(x["train_unlabelled"].uq_idxs))
    )
    print("Printing number of labelled classes...")
    print(len(set(x["train_labelled"].targets)))
    print("Printing total number of classes...")
    print(len(set(x["train_unlabelled"].targets)))
    print(f"Num Labelled Classes: {len(set(x['train_labelled'].targets))}")
    print(f"Num Unabelled Classes: {len(set(x['train_unlabelled'].targets))}")
    print(f"Len labelled set: {len(x['train_labelled'])}")
    print(f"Len unlabelled set: {len(x['train_unlabelled'])}")
