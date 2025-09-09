from copy import deepcopy

import numpy as np
import paddle

from config import cifar_10_root, cifar_100_root
from data.data_utils import subsample_instances


class CustomCIFAR10(paddle.vision.datasets.Cifar10):
    def __init__(self, *args, **kwargs):
        super(CustomCIFAR10, self).__init__(*args, **kwargs)
        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]
        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


class CustomCIFAR100(paddle.vision.datasets.Cifar100):
    def __init__(self, *args, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)
        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]
        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


def subsample_dataset(dataset, idxs):
    if len(idxs) > 0:
        dataset.data = dataset.data[idxs]
        dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]
        return dataset
    else:
        return None


def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):
    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i
    dataset = subsample_dataset(dataset, cls_idxs)
    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):
    train_classes = np.unique(train_dataset.targets)
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(train_dataset.targets == cls)[0]
        v_ = np.random.choice(
            cls_idxs, replace=False, size=(int(val_split * len(cls_idxs)),)
        )
        t_ = [x for x in cls_idxs if x not in v_]
        train_idxs.extend(t_)
        val_idxs.extend(v_)
    return train_idxs, val_idxs


def get_cifar_10_datasets(
    train_transform,
    test_transform,
    train_classes=(0, 1, 8, 9),
    prop_train_labels=0.8,
    split_train_val=False,
    seed=0,
):
    np.random.seed(seed)
    whole_training_set = CustomCIFAR10(
        root=cifar_10_root, transform=train_transform, train=True
    )
    train_dataset_labelled = subsample_classes(
        deepcopy(whole_training_set), include_classes=train_classes
    )
    subsample_indices = subsample_instances(
        train_dataset_labelled, prop_indices_to_subsample=prop_train_labels
    )
    train_dataset_labelled = subsample_dataset(
        train_dataset_labelled, subsample_indices
    )
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(
        deepcopy(train_dataset_labelled), train_idxs
    )
    val_dataset_labelled_split = subsample_dataset(
        deepcopy(train_dataset_labelled), val_idxs
    )
    val_dataset_labelled_split.transform = test_transform
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(
        train_dataset_labelled.uq_idxs
    )
    train_dataset_unlabelled = subsample_dataset(
        deepcopy(whole_training_set), np.array(list(unlabelled_indices))
    )
    test_dataset = CustomCIFAR10(
        root=cifar_10_root, transform=test_transform, train=False
    )
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


def get_cifar_100_datasets(
    train_transform,
    test_transform,
    train_classes=range(80),
    prop_train_labels=0.8,
    split_train_val=False,
    seed=0,
):
    np.random.seed(seed)
    whole_training_set = CustomCIFAR100(
        root=cifar_100_root, transform=train_transform, train=True
    )
    train_dataset_labelled = subsample_classes(
        deepcopy(whole_training_set), include_classes=train_classes
    )
    subsample_indices = subsample_instances(
        train_dataset_labelled, prop_indices_to_subsample=prop_train_labels
    )
    train_dataset_labelled = subsample_dataset(
        train_dataset_labelled, subsample_indices
    )
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(
        deepcopy(train_dataset_labelled), train_idxs
    )
    val_dataset_labelled_split = subsample_dataset(
        deepcopy(train_dataset_labelled), val_idxs
    )
    val_dataset_labelled_split.transform = test_transform
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(
        train_dataset_labelled.uq_idxs
    )
    train_dataset_unlabelled = subsample_dataset(
        deepcopy(whole_training_set), np.array(list(unlabelled_indices))
    )
    test_dataset = CustomCIFAR100(
        root=cifar_100_root, transform=test_transform, train=False
    )
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
    x = get_cifar_100_datasets(
        None,
        None,
        split_train_val=False,
        train_classes=range(80),
        prop_train_labels=0.5,
    )
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
    print(f"Num Labelled Classes: {len(set(x['train_labelled'].targets))}")
    print(f"Num Unabelled Classes: {len(set(x['train_unlabelled'].targets))}")
    print(f"Len labelled set: {len(x['train_labelled'])}")
    print(f"Len unlabelled set: {len(x['train_unlabelled'])}")
