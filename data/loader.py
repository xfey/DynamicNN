import os
import torch

import torchvision.datasets as dset

from core.config import cfg
from data.transforms import *
from data.imagenet16 import ImageNet16
from data.imagenet import ImageFolder


SUPPORTED_DATASETS = [
    "cifar10", 
    "cifar100", 
    "svhn", 
    "imagenet16", 
    "mnist", 
    "fashionmnist"
]

# if you use datasets loaded by imagefolder, you can add it here.
IMAGEFOLDER_FORMAT = ["imagenet"]


def get_dataloader(
    name=None,
    train_batch=None,
    cutout_length=0,
    use_classes=None,
    transforms=None,
    **kwargs
):
    name=cfg.LOADER.DATASET if name is None else name
    train_batch=cfg.LOADER.BATCH_SIZE if train_batch is None else train_batch
    name=cfg.LOADER.DATASET
    datapath=cfg.LOADER.DATAPATH
    test_batch=cfg.LOADER.BATCH_SIZE if cfg.TEST.BATCH_SIZE == -1 else cfg.TEST.BATCH_SIZE
    
    assert (name in SUPPORTED_DATASETS) or (name in IMAGEFOLDER_FORMAT), "dataset not supported."
    assert isinstance(train_batch, int), "normal dataloader using single training batch-size, not list."

    if name in SUPPORTED_DATASETS:
        # get normal dataloaders with train&test subsets.
        train_data, test_data = get_data(name, datapath, cutout_length, use_classes=use_classes, transforms=transforms)
        
        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=train_batch,
            shuffle=True,
            num_workers=cfg.LOADER.NUM_WORKERS,
            pin_memory=cfg.LOADER.PIN_MEMORY,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=test_batch,
            shuffle=False,
            num_workers=cfg.LOADER.NUM_WORKERS,
            pin_memory=cfg.LOADER.PIN_MEMORY,
        )
        return train_loader, test_loader
    elif name in IMAGEFOLDER_FORMAT:
        aug_type = cfg.LOADER.TRANSFORM
        return ImageFolder( # using path of training data of ImageNet as `datapath`
            datapath, batch_size=[train_batch, test_batch],
            augment_type=aug_type, **kwargs
        ).generate_data_loader()



def get_data(name, root, cutout_length, download=True, use_classes=None, transforms=None):
    assert name in SUPPORTED_DATASETS, "dataset not support."
    assert cutout_length >= 0, "cutout_length should not be less than zero."
    root = "./data/" + name if not root else os.path.join(root, name)

    if name == "cifar10":
        train_transform, valid_transform = transforms_cifar10(cutout_length) if transforms is None else transforms
        train_data = dset.CIFAR10(
            root=root, train=True, download=download, transform=train_transform
        )
        test_data = dset.CIFAR10(
            root=root, train=False, download=download, transform=valid_transform
        )
    elif name == "cifar100":
        train_transform, valid_transform = transforms_cifar100(cutout_length) if transforms is None else transforms
        train_data = dset.CIFAR100(
            root=root, train=True, download=download, transform=train_transform
        )
        test_data = dset.CIFAR100(
            root=root, train=False, download=download, transform=valid_transform
        )
    elif name == "svhn":
        train_transform, valid_transform = transforms_svhn(cutout_length) if transforms is None else transforms
        train_data = dset.SVHN(
            root=root, split="train", download=download, transform=train_transform
        )
        test_data = dset.SVHN(
            root=root, split="test", download=download, transform=valid_transform
        )
    elif name == "mnist":
        train_transform, valid_transform = transforms_mnist(cutout_length) if transforms is None else transforms
        train_data = dset.MNIST(
            root=root, train=True, download=download, transform=train_transform
        )
        test_data = dset.MNIST(
            root=root, train=False, download=download, transform=valid_transform
        )
    elif name == "fashionmnist":
        train_transform, valid_transform = transforms_mnist(cutout_length) if transforms is None else transforms
        train_data = dset.FashionMNIST(
            root=root, train=True, download=download, transform=train_transform
        )
        test_data = dset.FashionMNIST(
            root=root, train=False, download=download, transform=valid_transform
        )
    elif name == "imagenet16":
        train_transform, valid_transform = transforms_imagenet16() if transforms is None else transforms
        train_data = ImageNet16(
            root=root,
            train=True,
            transform=train_transform,
            use_num_of_class_only=use_classes,
        )
        test_data = ImageNet16(
            root=root,
            train=False,
            transform=valid_transform,
            use_num_of_class_only=use_classes,
        )
        if use_classes == 120 or use_classes is None:   # Use 120 classes by default.
            assert len(train_data) == 151700 and len(test_data) == 6000
        elif use_classes == 150:
            assert len(train_data) == 190272 and len(test_data) == 7500
        elif use_classes == 200:
            assert len(train_data) == 254775 and len(test_data) == 10000
        elif use_classes == 1000:
            assert len(train_data) == 1281167 and len(test_data) == 50000
    else:
        exit(0)
    return train_data, test_data
