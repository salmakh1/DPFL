import logging
import copy
from pathlib import Path
from typing import Optional
import os
import requests
import tarfile
import shutil
import numpy as np
import torch
from hydra.utils import call
# from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms as transform_lib
from torchvision.datasets import ImageFolder
from hydra.utils import instantiate

from src.data.data_utils import split_subsets_train_val, split_dataset_train_val, add_attrs, customized_test_split, \
    random_split_data_to_clients, class_frequencies


class CINIC10DataModule():
    name = "CINIC10"

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
        self.normalize = normalize
        self.num_classes = num_classes
        self.seed = seed
        self.num_clients = num_clients
        self.shuffle = shuffle
        self.val_split = val_split
        self.pub_size = pub_size
        self.transform = transform
        self.ds_mean = [0.47889522, 0.47227842, 0.43047404]
        self.ds_std = [0.24205776, 0.23828046, 0.25874835]
        self.root_dir = (root_dir)  # / str(self.dataset)
        self.current_client_idx = 0

    def prepare_data(self):
        """Saves CINIC files to `data_dir`"""
        # if self.is_setup:
        #     return

        def download_dataset(url, target_directory):
            # Create the target directory if it doesn't exist
            logging.info(f"the directory is {self.root_dir}")
            os.makedirs(target_directory, exist_ok=True)

            # Check if the dataset is already downloaded
            dataset_file = os.path.join(target_directory, "CINIC-10.tar.gz")
            if os.path.exists(dataset_file):
                print("Dataset already downloaded.")
                return

            # Download the dataset file
            response = requests.get(url)
            with open(dataset_file, "wb") as f:
                f.write(response.content)

            # Extract the contents of the tar file to the target directory
            with tarfile.open(dataset_file, "r:gz") as tar:
                tar.extractall(target_directory)

            # Remove the downloaded tar file
            # os.remove(dataset_file)

        url = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
        target_directory = f"{self.root_dir}/cinic-10"

        # Download the dataset
        logging.info("Downloading dataset...")
        download_dataset(url, target_directory)

        # Create train_val folder
        def combine_directories(source_dir1, source_dir2, destination_dir):
            # Create the destination directory if it doesn't exist
            os.makedirs(destination_dir, exist_ok=True)

            # Copy the contents of source_dir1 to the destination directory
            for root, dirs, files in os.walk(source_dir1):
                relative_dir = os.path.relpath(root, source_dir1)
                destination_subdir = os.path.join(destination_dir, relative_dir)
                os.makedirs(destination_subdir, exist_ok=True)

                for file in files:
                    source_file = os.path.join(root, file)
                    destination_file = os.path.join(destination_subdir, file)
                    shutil.copy2(source_file, destination_file)

            # Copy the contents of source_dir2 to the destination directory
            for root, dirs, files in os.walk(source_dir2):
                relative_dir = os.path.relpath(root, source_dir2)
                destination_subdir = os.path.join(destination_dir, relative_dir)
                os.makedirs(destination_subdir, exist_ok=True)

                for file in files:
                    source_file = os.path.join(root, file)
                    destination_file = os.path.join(destination_subdir, file)
                    shutil.copy2(source_file, destination_file)

        # Define the source directories (train and val) and the destination directory (train_val)
        source_dir1 = f"{target_directory}/train"
        source_dir2 = f"{target_directory}/valid"
        destination_dir = f"{target_directory}/train_val"

        if os.path.exists(destination_dir):
            print("Dataset already combined.")
            return
        # Combine the directories
        logging.info("Combining directories...")
        combine_directories(source_dir1, source_dir2, destination_dir)

    def train_loaders(self, datasets_train):
        loader = DataLoader(
            datasets_train[self.current_client_idx],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_loaders(self, datasets_val):
        loader = DataLoader(
            datasets_val[self.current_client_idx],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return loader

    def test_loaders(self, test_dataset):
        loader = DataLoader(
            test_dataset[self.current_client_idx],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return loader

    def global_test_loader(self, test_set):
        test_set = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,

        )
        return test_set

    def load_data(self):

        self.prepare_data()

        root = f"{self.root_dir}/cinic-10/train_val"
        train_set = ImageFolder(
            root=root, transform=self.aug_transforms
        )
        val_set = ImageFolder(
            root=root, transform=self.default_transforms
        )

        test_set = ImageFolder(
            root=f"{self.root_dir}/cinic-10/test",
            transform=self.default_transforms
        )
        train_set.targets = torch.Tensor(train_set.targets).to(torch.long)
        val_set.targets = torch.Tensor(val_set.targets).to(torch.long)
        test_set.targets = torch.Tensor(test_set.targets).to(torch.long)

        return train_set, val_set, test_set, None


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

        if val_pub:
            datasets_val_pub, datasets_val = split_subsets_train_val(
                datasets_val, 0.5, self.seed, val_dataset=val_set,
            )

        if freq:
            frequencies = class_frequencies(train_set, train_distribution, cfg.split.seed, cfg.split.min_dataset_size)
        logging.info(
            f"train_data {len(datasets_train)} and val_data {len(datasets_val)} and test is {len(local_test_set)}")

        return datasets_train, datasets_val, pub_set, local_test_set, global_test_set, datasets_val_pub, frequencies, \
               transformed_public_datasets ,weights


    #
    # def transfer_setup(self):
    #     root = f"{self.data_dir}/cinic-10/train_val"
    #     self.train_dataset = ImageFolder(
    #         root=root, transform=self.aug_transforms
    #     )
    #     self.val_dataset = ImageFolder(
    #         root=root,
    #         transform=self.default_transforms
    #     )
    #     self.test_dataset = ImageFolder(
    #         root=f"{self.data_dir}/cinic-10/test",
    #         transform=self.default_transforms
    #     )

    def next_client(self):
        self.current_client_idx += 1
        # assert self.current_client_idx < self.num_clients


    def set_client(self):
        self.current_client_idx = 0


    @property
    def default_transforms(self):
        cifar_transforms = [
            transform_lib.ToTensor(),
        ]
        if self.normalize:
            cifar_transforms.append(transform_lib.Normalize(mean=self.ds_mean,
                                                            std=self.ds_std))

        return transform_lib.Compose(cifar_transforms)

    @property
    def aug_transforms(self):
        cifar_transforms = [
            transform_lib.RandomCrop(32, padding=4),
            transform_lib.RandomHorizontalFlip(),
            transform_lib.ToTensor(),

        ]
        if self.normalize:
            cifar_transforms.append(transform_lib.Normalize(mean=self.ds_mean,
                                                            std=self.ds_std))

        return transform_lib.Compose(cifar_transforms)