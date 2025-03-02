from __future__ import print_function

import csv
import logging
import os
import os.path
import warnings

from PIL import Image
from torch.utils.data import DataLoader

from src.data.data_patritioner import DataPartitioner
from src.data.data_transforms import get_data_transform


class OpenImageUtils():
    """
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    classes = []

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return len(self.targets)

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return len(self.targets)

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, data_dir, dataset='train', transform=None, target_transform=None, imgview=False):

        self.root = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data_file = dataset  # 'train', 'test', 'validation'
        logging.info(f"address {self.root}")
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You have to download it')

        self.path = os.path.join(self.processed_folder, self.data_file)
        # load data and targets
        self.data, self.targets = self.load_file(self.path)
        self.imgview = imgview

    # def load_train(self):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        imgName, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(os.path.join(self.path, imgName))

        # avoid channel error
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return self.root

    @property
    def processed_folder(self):
        return self.root

    def _check_exists(self):
        logging.info(f"The file address is {os.path.join(self.processed_folder,self.data_file)}")
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.data_file)))

    def load_meta_data(self, path):
        datas, labels = [], []

        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    datas.append(row[1])
                    labels.append(int(row[-1]))
                line_count += 1

        return datas, labels

    def load_file(self, path):

        # load meta file to get labels
        if self.data_file != "validation":
            datas, labels = self.load_meta_data(
                os.path.join(self.processed_folder, 'client_data_mapping', self.data_file + '.csv'))
        else:
            datas, labels = self.load_meta_data(
                os.path.join(self.processed_folder, 'client_data_mapping', "val" + '.csv'))
        return datas, labels

def select_dataset(rank, partition, batch_size, isTest=False, collate_fn=None):
    """Load data given client Id"""
    # logging.info(f"rank is {rank}")
    partition = partition.use(rank - 1, isTest)
    dropLast = False #if isTest else True
    if isTest:
        num_loaders = 0
    else:
        num_loaders = min(int(len(partition) / batch_size / 2), 2)
    if num_loaders == 0:
        time_out = 0
    else:
        time_out = 60

    if collate_fn is not None:
        return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out,
                          num_workers=num_loaders, drop_last=dropLast, collate_fn=collate_fn)
    return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out,
                      num_workers=num_loaders, drop_last=dropLast)


