import logging

import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from pathlib import Path
from torchvision import transforms as transform_lib
import torchvision.transforms as T
import os
from hydra.utils import instantiate

from src.data.data_transforms import get_data_transform
from src.data.data_utils import random_split_data_to_clients, split_dataset_train_val, split_subsets_train_val


class ImageNetDataModule():
    def __init__(self,
                 root_dir: str = Path(torch.hub.get_dir()) / f'datasets',
                 batch_size: int = 32,
                 num_workers: int = 1,
                 normalize: bool = True,
                 num_classes: int = 10,
                 seed: int = 42,
                 num_clients: int = 2,
                 shuffle: bool = True,
                 val_split: int = 0.1,
                 pub_size: int = 0.1,
                 transform: bool = False
                 ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.norm = normalize
        self.num_classes = num_classes
        self.seed = seed
        self.num_clients = num_clients
        self.shuffle = shuffle
        self.val_split = val_split
        self.pub_size = pub_size
        self.transform = transform
        self.dataset = ImageNet  ## either Cifar10 or Cifar100
        self.root_dir = (root_dir)  # / str(self.dataset)
        self.current_client_idx = 0
        assert num_classes in (10, 100)  ## raise exception if the number of classes is not 10

    def load_data(self):
        # load data if it does not exist in the specific path
        normalize = self.normalize_data()
        transforms = T.RandomApply(torch.nn.ModuleList([
            T.ColorJitter(),
        ]), p=0.2)

        train_transform, val_transform = get_data_transform('imagenet')

        if not os.path.exists(self.root_dir / str("IMAGENET")):
            print(type(self.root_dir))
            print(self.root_dir)
            train_set = self.dataset(
                self.root_dir,
                split="train",
                download=True,
                transform=train_transform
            )

            val_set = self.dataset(
                self.root_dir,
                split="train",
                download=False,
                transform=val_transform
            )

            transform_train_set = self.dataset(
                self.root_dir, train=True,
                download=False,
                transform=train_transform
            )

            test_set = self.dataset(
                self.root_dir,
                split="val",
                download=True,
                transform=val_transform
            )
        else:
            train_set = self.dataset(
                root=self.root_dir,
                split="train",
                download=False,
                transform=train_transform
            )
            val_set = self.dataset(
                self.root_dir, split="train",
                download=False,
                transform=val_transform
            )

            transform_train_set = self.dataset(
                self.root_dir, split="train",
                download=False,
                transform=train_transform
            )

            test_set = self.dataset(
                root=self.root_dir,
                split="val",
                download=False,
                transform=val_transform
            )
        print(train_set)
        print(val_set)
        print(test_set)
        return train_set, val_set, test_set, transform_train_set

    def normalize_data(self):
        if self.norm:
            normalize = T.Normalize(
                mean=(0.4915, 0.4823, 0.4468),
                std=(0.2470, 0.2435, 0.2616),
            )
        else:
            normalize = T.Normalize(
                mean=(0., 0., 0.),
                std=(1.0, 1.0, 1.0),
            )
        return normalize

    def load_and_split(self, cfg):
        train_set, val_set, test_set, transform_train_set = self.load_data()
        train_set, pub_set = split_dataset_train_val(
            train_dataset=train_set,
            val_split=self.pub_size,
            seed=self.seed,
            val_dataset=val_set,
        )
        # Split the full data with the specified split function
        logging.info(f"split method is {cfg.split}")
        # Split data to train datasets into train and validation
        if cfg.datamodule.transform:
            train_datasets = instantiate(cfg.split, dataset=train_set, transformed_dataset=transform_train_set)
        else:
            train_datasets = instantiate(cfg.split, dataset=train_set)
        datasets_train, datasets_val = split_subsets_train_val(
            train_datasets, self.val_split, self.seed, val_dataset=val_set,
        )

        logging.info(f"train_data {len(datasets_train)} and val_data {len(datasets_val)}")
        return datasets_train, datasets_val, pub_set


    def train_loaders(self, train_set):
        train_loader = DataLoader(
            train_set[self.current_client_idx],
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=None,
        )
        return train_loader

    def val_loaders(self, train_set):
        val_loader = DataLoader(
            train_set[self.current_client_idx],
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=None,
        )
        return val_loader

    # This function loads the global test set
    def test_set(self):

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform = T.Compose([
            T.Resize(256),
            T.RandomResizedCrop(224),
            T.ToTensor(),
            normalize,
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        test_set = self.dataset(
            root=self.root_dir,
            split="val",
            download=True,
            transform=transform,
        )
        test_set_ = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,

        )
        return test_set_

    def public_loader(self, public_data):
        pub_set = DataLoader(
            public_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=None,
        )
        return pub_set

    def list_batches(self, train_loader):
        l = [data_pair for data_pair in train_loader]
        return l

    def set_client(self):
        self.current_client_idx = 0

    def next_client(self):
        self.current_client_idx += 1

    def client_data(self):
        train_set, test_set = self.load_data()
        train_loaders, test_loaders = self.data_loaders(train_set, test_set)
        return train_loaders, test_loaders


    def plot_data(self, train_set):
        dataiter = iter(train_set)
        images, labels = dataiter.next()
        plt.imshow(np.transpose(images[6].numpy(), (1, 2, 0)))
        plt.savefig("a.pdf")