import copy
import logging
from collections import defaultdict

import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from pathlib import Path
from torchvision import transforms as transform_lib
import torchvision.transforms as T
import os
from sklearn.model_selection import train_test_split

from hydra.utils import instantiate
from src.data.data_utils import random_split_data_to_clients, split_dataset_train_val, split_subsets_train_val, \
    customized_test_split, class_frequencies, ru_customized_test_split


class CIFARDataModule():
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
                 transform: bool = False,
                 m: int = 10,
                 available_clients: int = 10,
                 local_steps: int = 5
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
        if self.num_classes == 10:
            self.dataset = CIFAR10  ## either Cifar10 or Cifar100
        else:
            logging.info("CIFAR100")
            self.dataset = CIFAR100
        self.root_dir = (root_dir)  # / str(self.dataset)
        self.current_client_idx = 0
        assert num_classes in (10, 100)  ## raise exception if the number of classes is not 10

    def load_data(self):
        # load data if it does not exist in the specific path
        normalize = self.normalize_data()
        # transforms = T.RandomApply(torch.nn.ModuleList([
        #     T.ColorJitter(),
        # ]), p=0.2)

        if not os.path.exists(self.root_dir / str("CIFAR")):
            print(type(self.root_dir))
            print(self.root_dir)
            train_set = self.dataset(
                self.root_dir,
                train=True,
                download=True,
                transform=T.Compose([
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize,
                ]),
            )

            val_set = self.dataset(
                self.root_dir, train=True,
                download=False,
                transform=T.Compose([
                    T.ToTensor(),
                    normalize,
                ]),
            )

            transform_train_set = self.dataset(
                self.root_dir, train=True,
                download=False,
                transform=T.Compose([
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(10),
                    T.ToTensor(),
                    T.GaussianBlur(kernel_size=(5, 5), sigma=(2, 2)),
                    normalize,
                ]),
            )

            test_set = self.dataset(
                self.root_dir,
                train=False,
                download=True,
                transform=T.Compose([
                    T.ToTensor(),
                    normalize,

                ]),
            )
        else:
            train_set = self.dataset(
                root=self.root_dir,
                train=True,
                download=False,
                transform=T.Compose([
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize,
                ]),
            )
            val_set = self.dataset(
                self.root_dir, train=True,
                download=False,
                transform=T.Compose([
                    T.ToTensor(),
                    normalize,
                ]),
            )

            transform_train_set = self.dataset(
                self.root_dir, train=True,
                download=False,
                transform=T.Compose([
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(10),
                    T.GaussianBlur(kernel_size=(5, 5), sigma=(2, 2)),
                    T.ToTensor(),
                    normalize,

                ]),
            )

            test_set = self.dataset(
                root=self.root_dir,
                train=False,
                download=False,
                transform=T.Compose([
                    T.ToTensor(),
                    normalize,
                ]),
            )
        print(train_set)
        print(val_set)
        print(test_set)
        return train_set, val_set, test_set, transform_train_set

    def normalize_data(self):
        if self.norm:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        else:
            normalize = T.Normalize(
                mean=(0., 0., 0.),
                std=(1.0, 1.0, 1.0),
            )
        return normalize

    def load_and_split(self, cfg, pub=False, local_test=False, val_pub=False, freq=False, transformed_pub=False):
        train_set, val_set, test_set, transform_train_set = self.load_data()
        pub_set = None
        datasets_val_pub = None
        local_test_set = None
        global_test_set = test_set
        frequencies = None
        transformed_public_datasets = None

        if pub:
            train_set, pub_set = split_dataset_train_val(
                train_dataset=train_set,
                val_split=self.pub_size,
                seed=self.seed,
                val_dataset=val_set,
            )
        # Split the full data with the specified split function
        logging.info(f"split method is {cfg.split}")

        if cfg.datamodule.transform:
            train_datasets, train_distributionl, weights = instantiate(cfg.split, dataset=train_set,
                                                                       transformed_dataset=transform_train_set)

        else:
            train_datasets, train_distribution, weights = instantiate(cfg.split, dataset=train_set)

        if transformed_pub:
            public_datasets = copy.deepcopy(train_datasets)
            _, transformed_public_datasets = split_subsets_train_val(
                public_datasets,
                0.2,
                self.seed,
                val_dataset=val_set,
            )

            logging.info(f"transformed pub_data")

            # plot_distibution(train_set, transformed_public_datasets[0], "pub")

        if local_test:

            if train_distribution:
                local_test_set = customized_test_split(test_set, train_distribution, cfg.split.num_clients,
                                                       cfg.split.seed, cfg.split.min_dataset_size)
            else:
                local_test_set, _ = random_split_data_to_clients(test_set, cfg.split.num_clients, self.seed)

        datasets_train, datasets_val = split_subsets_train_val(
            train_datasets, self.val_split, self.seed, val_dataset=val_set,
        )

        # plot_distibution(train_set, datasets_train[0], "train")
        # plot_distibution(train_set, datasets_val[0], "val")

        if val_pub:
            datasets_val_pub, datasets_val = split_subsets_train_val(
                datasets_val, 0.5, self.seed, val_dataset=val_set,
            )

        if freq:
            frequencies = class_frequencies(train_set, train_distribution, cfg.split.seed, cfg.split.min_dataset_size)
        logging.info(
            f"train_data {len(datasets_train)} and val_data {len(datasets_val)} and test is {len(local_test_set)}")

        return datasets_train, datasets_val, pub_set, local_test_set, global_test_set, datasets_val_pub, frequencies, \
               transformed_public_datasets, weights

    def load_and_split_inference(self, cfg, pub=False, local_test=False, val_pub=False, freq=False):
        train_set, val_set, test_set, transform_train_set = self.load_data()

        pub_set = None
        datasets_val_pub = None
        local_test_set = None
        global_test_set = test_set
        frequencies = None

        if pub:
            train_set, pub_set = split_dataset_train_val(
                train_dataset=train_set,
                val_split=self.pub_size,
                seed=self.seed,
                val_dataset=val_set,
            )

        # Split the full data with the specified split function
        logging.info(f"split method is {cfg.split}")

        if cfg.datamodule.transform:
            train_datasets, train_distribution, weights = instantiate(cfg.split, dataset=train_set,
                                                                      transformed_dataset=transform_train_set)

        else:
            train_datasets, train_distribution, weights = instantiate(cfg.split, dataset=train_set)

        if local_test:

            if train_distribution:

                local_test_set = customized_test_split(test_set, train_distribution, cfg.split.num_clients,
                                                       cfg.split.seed, cfg.split.min_dataset_size)

            else:
                local_test_set, _ = random_split_data_to_clients(test_set, cfg.split.num_clients, self.seed)

        datasets_train, datasets_val = split_subsets_train_val(
            train_datasets, self.val_split, self.seed, val_dataset=val_set,
        )
        if val_pub:
            datasets_val_pub, datasets_val = split_subsets_train_val(
                datasets_val, 0.5, self.seed, val_dataset=val_set,
            )

        if freq:
            frequencies = class_frequencies(train_set, train_distribution, cfg.split.seed, cfg.split.min_dataset_size)

        # if number_per_label:
        #     count_per_label = defaultdict(dict)
        #
        #     indices = np.array(train_set.indices) if isinstance(train_set, Subset) else np.arange(len(train_set))
        #     targets = train_set.dataset.targets if isinstance(train_set, Subset) else train_set.targets
        #     targets = targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
        #     df = pd.DataFrame({"target": targets[indices]}, index=indices)
        #     label_to_indices = {}
        #     for label, group in df.groupby('target'):
        #         label_to_indices[label] = group.index
        #     labels, classes_count_ = np.unique(df.target, return_counts=True)
        #     classes_count = defaultdict(int)
        #     for label, count in zip(labels, classes_count_):
        #         classes_count[label] = count
        #     num_classes = len(df.target.unique())
        #     for j in range(cfg.datamodule.num_clients):
        #         for m in range(num_classes):
        #             count_per_label[j][m] = train_distribution[j][m] * classes_count[m]

        logging.info(
            f"train_data {len(datasets_train)} and val_data {len(datasets_val)} and test is {len(local_test_set)}")

        return datasets_train, datasets_val, pub_set, local_test_set, global_test_set, datasets_val_pub, frequencies

    # def load_and_split_inference(self, cfg, pub=False, local_test=False, val_pub=False, freq=False):
    #     # train_set, malicious_train_set, val_set, test_set, transform_train_set = self.load_data()
    #     train_set, val_set, test_set, transform_train_set = self.load_data()
    #
    #     pub_set = None
    #     datasets_val_pub = None
    #     local_test_set = None
    #     global_test_set = test_set
    #     frequencies = None
    #     # train_distribution = None
    #     # test_datasets=None
    #     if pub:
    #         train_set, pub_set = split_dataset_train_val(
    #             train_dataset=train_set,
    #             val_split=self.pub_size,
    #             seed=self.seed,
    #             val_dataset=val_set,
    #         )
    #         # if self.split_train:
    #         #     malicious_train_set, malicious_pub_set = split_dataset_train_val(
    #         #     train_dataset=malicious_train_set,
    #         #     val_split=self.pub_size,
    #         #     seed=self.seed,
    #         #     val_dataset=val_set,
    #         #  )
    #     # Split the full data with the specified split function
    #     logging.info(f"split method is {cfg.split}")
    #
    #     if cfg.datamodule.transform:
    #         train_datasets, train_distribution, weights = instantiate(cfg.split, dataset=train_set,
    #                                                          transformed_dataset=transform_train_set)
    #         # if self.split_train:
    #         # malicious_train_datasets, malicious_train_distribution = instantiate(cfg.split, dataset=malicious_train_set,
    #         #                                                  transformed_dataset=transform_train_set)
    #     else:
    #         train_datasets, train_distribution, weights = instantiate(cfg.split, dataset=train_set)
    #         # if self.split_train:
    #         #     malicious_train_datasets, malicious_train_distribution = instantiate(cfg.split, dataset=malicious_train_set)
    #
    #     if local_test:
    #
    #         if train_distribution:
    #             # local_test_set = ru_customized_test_split(test_set, train_distribution, cfg.split.num_clients,
    #             #                                           cfg.split.seed, cfg.split.min_dataset_size, ru=cfg.ru,
    #             #                                           missing_classes=cfg.split.missing_classes)
    #
    #             local_test_set = customized_test_split(test_set, train_distribution, cfg.split.num_clients,
    #                                                    cfg.split.seed, cfg.split.min_dataset_size)
    #             # if self.split_train:
    #             #     local_malicious_test_set = ru_customized_test_split(test_set, train_distribution, cfg.split.num_clients,
    #             #                                               cfg.split.seed, cfg.split.min_dataset_size, ru=cfg.ru,
    #             #                                               missing_classes=cfg.split.missing_classes)
    #             # #
    #             # local_test_set = customized_test_split(test_set, train_distribution, cfg.split.num_clients,
    #             #                                        cfg.split.seed, cfg.split.min_dataset_size)
    #         else:
    #             local_test_set, _ = random_split_data_to_clients(test_set, cfg.split.num_clients, self.seed)
    #
    #     datasets_train, datasets_val = split_subsets_train_val(
    #         train_datasets, self.val_split, self.seed, val_dataset=val_set,
    #     )
    #     if val_pub:
    #         datasets_val_pub, datasets_val = split_subsets_train_val(
    #             datasets_val, 0.5, self.seed, val_dataset=val_set,
    #         )
    #
    #     if freq:
    #         frequencies = class_frequencies(train_set, train_distribution, cfg.split.seed, cfg.split.min_dataset_size)
    #
    #     logging.info(
    #         f"train_data {len(datasets_train)} and val_data {len(datasets_val)} and test is {len(local_test_set)}")
    #
    #     return datasets_train, datasets_val, pub_set, local_test_set, global_test_set, datasets_val_pub, frequencies

    def train_loaders(self, train_set):
        train_loader = DataLoader(
            train_set[self.current_client_idx],
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=None,
        )
        return train_loader

    def val_loaders(self, val_set):
        val_loader = DataLoader(
            val_set[self.current_client_idx],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=None,
        )
        return val_loader

    def test_loaders(self, test_set):
        test_loader = DataLoader(
            test_set[self.current_client_idx],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=None,
        )
        return test_loader

    def global_test_loader(self, test_set):
        test_set = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,

        )
        return test_set

    # This function loads the global test set
    def test_set(self):
        transform = T.Compose(
            [T.ToTensor(),
             T.Normalize(mean=(0.4915, 0.4823, 0.4468),
                         std=(0.2470, 0.2435, 0.2616))])

        test_set = self.dataset(
            root=self.root_dir,
            train=False,
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
        # assert self.current_client_idx < self.num_clients, "Client number shouldn't excced seleced number of clients"

    def client_data(self):
        train_set, test_set = self.load_data()
        train_loaders, test_loaders = self.data_loaders(train_set, test_set)
        return train_loaders, test_loaders

    def plot_data(self, train_set):
        dataiter = iter(train_set)
        images, labels = dataiter.next()
        plt.imshow(np.transpose(images[6].numpy(), (1, 2, 0)))
        plt.savefig("a.pdf")
