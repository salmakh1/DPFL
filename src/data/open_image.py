import logging

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
from hydra.utils import instantiate

from src.data.data_patritioner import DataPartitioner
from src.data.data_transforms import get_data_transform
from src.data.data_utils import random_split_data_to_clients, split_dataset_train_val, split_subsets_train_val
from src.data.open_image_utils import OpenImageUtils


class OpenImage():

    def __init__(self, data_dir, seed, num_classes, batch_size):
        self.data_dir = data_dir
        self.seed = seed
        self.num_classes = num_classes
        self.batch_size = batch_size

    # The idea for public datatset is to use only the uniform split for validation

    def load_and_split(self, cfg):
        train_transform, val_transform = get_data_transform('openImg')
        train_dataset = OpenImageUtils(cfg.datamodule.data_dir, dataset='train', transform=train_transform)
        logging.info("Data partitioner starts ...")
        training_sets = DataPartitioner(
            data=train_dataset)
        training_sets.partition_data_helper(
            num_clients=cfg.num_clients,
            data_map_file=os.path.join(cfg.datamodule.data_dir, "client_data_mapping/train.csv"))

        logging.info("Data partitioner completes ...")

        return training_sets

    def pub_data(self, cfg):
        train_transform, val_transform = get_data_transform('openImg')

        public_dataset = OpenImageUtils(cfg.datamodule.data_dir, dataset='validation', transform=val_transform)
        partition_public_set = DataPartitioner(
            data=public_dataset)
        partition_public_set.partition_public(cfg.num_clients)

        return partition_public_set

    def test_set(self,cfg):
        train_transform, test_transform = get_data_transform('openImg')
        test_dataset = OpenImageUtils(self.data_dir, dataset='test', transform=test_transform)

        partition_test_set = DataPartitioner(
            data=test_dataset)
        partition_test_set.partition_data_helper(
            num_clients=cfg.num_clients,
            data_map_file=os.path.join(cfg.datamodule.data_dir, "client_data_mapping/test.csv"))

        #COMMENTED HERE
        # partition_test_set = DataPartitioner(
        #     data=test_dataset)
        # partition_test_set.partition_test()

        # logging.info(f"length of the test set is {partition_test_set.getDataLen()} {partition_test_set}")
        return partition_test_set


    def plot_data(self, train_set):
        dataiter = iter(train_set)
        images, labels = dataiter.next()
        plt.imshow(np.transpose(images[6].numpy(), (1, 2, 0)))
        plt.savefig("a.pdf")
