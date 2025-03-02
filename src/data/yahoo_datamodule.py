import copy
import logging
import pandas as pd
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from pathlib import Path
import torchvision.transforms as T
import os

from hydra.utils import instantiate
from src.data.data_utils import random_split_data_to_clients, split_subsets_train_val, \
    customized_test_split, class_frequencies

class CustomDataset(Dataset):
    def __init__(self, data_file):
        data = torch.load(data_file)
        self.data = np.array(data['sents'])
        self.labels = np.array(data['labels'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'sent': self.data[idx],
            'label': self.labels[idx]
        }
        return sample

class YahooDatamodule():
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
        pub_set = None
        datasets_val_pub = None
        local_test_set = None
        frequencies = None
        transformed_public_datasets = None

        csv_file_path = os.path.join(os.path.expanduser('~'), 'yahoo_answers_csv', 'train.csv')
        data_frame = pd.read_csv(csv_file_path)
        logging.info(f"td {data_frame.columns} num is {data_frame.shape[1]}")

        traindata = torch.tensor(data_frame.values)


        csv_file_path = os.path.join(os.path.expanduser('~'), 'yahoo_answers_csv', 'test.csv')
        data_frame = pd.read_csv(csv_file_path)
        testdata = torch.tensor(data_frame.values)

        train_set = CustomDataset(traindata)
        test_set = CustomDataset(testdata)
        global_test_set = test_set

        # train_data = np.array(traindata['sents'])
        # train_label = np.array(traindata['labels'])
        # test_data = np.array(testdata['sents'])
        # test_label = np.array(testdata['labels'])

        train_datasets, train_distribution, weights = instantiate(cfg.split, dataset=train_set)

        datasets_train, datasets_val = split_subsets_train_val(
            train_datasets, self.val_split, self.seed, val_dataset=None,
        )

        if train_distribution:
            local_test_set = customized_test_split(test_set, train_distribution, cfg.split.num_clients,
                                                   cfg.split.seed, cfg.split.min_dataset_size)
        else:
            local_test_set, _ = random_split_data_to_clients(test_set, cfg.split.num_clients, self.seed)


        if freq:
            frequencies = class_frequencies(train_set, train_distribution, cfg.split.seed, cfg.split.min_dataset_size)
        logging.info(
            f"train_data {len(datasets_train)} and val_data {len(datasets_val)} and test is {len(local_test_set)}")

        return datasets_train, datasets_val, pub_set, local_test_set, global_test_set, datasets_val_pub, frequencies, \
               transformed_public_datasets, weights


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
