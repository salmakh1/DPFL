import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

import torch

from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import MNIST
from pathlib import Path
import torchvision.transforms as T
import os
import logging
from hydra.utils import instantiate
from src.data.data_utils import random_split_data_to_clients, split_subsets_train_val

log = logging.getLogger(__name__)

class MNISTDataModule():

    def __init__(self,
                 root_dir: str = Path(torch.hub.get_dir()) / f'datasets',
                 batch_size: int = 32,
                 num_workers: int = 1,
                 normalize: bool = True,
                 num_classes: int = 10,
                 seed: int = 42,
                 num_clients: int = 2,
                 shuffle: bool = True,
                 ):

        logging.info("root_directory {}".format(root_dir))

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.norm = normalize
        self.num_classes = num_classes
        self.seed = seed
        self.num_clients = num_clients
        self.shuffle = shuffle
        self.dataset = MNIST  ## either Cifar10 or Cifar100
        self.root_dir = (root_dir)  #/ f"MNIST"
        self.current_client_idx = 0
        logging.info("root_directory {}".format(root_dir))
        assert num_classes in (10, 100)  ## raise exception if the number of classes is not 10

    def load_data(self):
        # load data if it does not exist in the specific path
        normalize = self.normalize_data()
        if not os.path.exists(self.root_dir / str("MNIST")):
            print(type(self.root_dir))
            print(self.root_dir)
            train_set = self.dataset(
                self.root_dir,
                train=True,
                download=True,
                transform=T.Compose([
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(10),
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
                    T.RandomRotation(10),
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
        return train_set, val_set,test_set


    def normalize_data(self):
        if self.norm:
            normalize = T.Normalize([0.1307], [0.3081])
        else:
            normalize = T.Normalize([0.1307], [0.3081])
        return normalize

    def load_and_split(self, cfg, t=0):
        train_set, val_set, test_set = self.load_data()
        #Split the full data with the specified split fumciton
        train_datasets = instantiate(cfg.split, dataset=train_set)
        #Split data to train datastes into  tain and validation and validation
        datasets_train, datasets_val = split_subsets_train_val(
            train_datasets, self.val_split, self.seed, val_dataset=val_set
        )
        # if cfg.decentralized:
        #If we cant to split the test set between clients we can add this line
        # test_datasets = random_split_data_to_clients(test_set, self.num_clients, self.seed, "_")
        return datasets_train, datasets_val

    def data_loaders(self, train_set):
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

    def test_loader(self, test_set):
        test_loader = DataLoader(
            test_set[self.current_client_idx],
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=None,
        )
        return test_loader


    #This funcrion loads the global test set
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
        normalize = T.Normalize([0.1307], [0.3081])

        inv_normalize = T.Normalize(
            mean=[-m / s for m, s in zip(normalize.mean, normalize.std)],
            std=[1 / s for s in normalize.std]
        )
        index = torch.randint(0, len(train_set), ())
        image, label = train_set[index]

        plt.imshow(TF.to_pil_image(inv_normalize(image)))
        plt.title(f'Label: {label}')
        plt.axis('off')
        plt.show()




