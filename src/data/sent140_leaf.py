import copy
import itertools
import json
import logging
import os
from collections import defaultdict, OrderedDict
import random

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset

from src.data.sent140_dataloader import dataloader
from src.data.sent140_utils import process_x, process_y


class Sent140():

    def __init__(self, seed, num_classes, batch_size, num_workers, shuffle, num_clients, m, available_clients,
                 local_steps):
        self.seed = seed
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.current_client_idx = 0
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.num_clients = num_clients
        self.m = m
        self.available_clients = available_clients
        self.local_steps = local_steps

    # The idea for public datatset is to use only the uniform split for validation

    def read_dir(self, data_dir):
        clients = []
        groups = []
        data = defaultdict(lambda: None)

        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith('.json')]
        for f in files:
            file_path = os.path.join(data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            clients.extend(cdata['users'])
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            data.update(cdata['user_data'])

        clients = list(sorted(data.keys()))
        logging.info(f"clients {clients}")
        return clients, groups, data

    def read_data(self, train_data_dir, test_data_dir):
        '''parses data in given train and test data directories
        assumes:
        - the data in the input directories are .json files with
            keys 'users' and 'user_data'
        - the set of train set users is the same as the set of test set users

        Return:
            clients: list of client ids
            groups: list of group ids; empty list if none found
            train_data: dictionary of train data
            test_data: dictionary of test data
        '''
        clients = []
        groups = []
        train_data = {}
        test_data = {}

        # train_files = os.listdir(train_data_dir)
        # train_files = ["mytrain.json"]
        train_files = os.listdir(train_data_dir)
        train_files = [f for f in train_files if f.endswith('.json')]
        for f in train_files:
            file_path = os.path.join(train_data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            clients.extend(cdata['users'])
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            train_data.update(cdata['user_data'])

        # test_files = os.listdir(test_data_dir)
        # test_files = ["mytest.json"]
        test_files = os.listdir(test_data_dir)
        test_files = [f for f in test_files if f.endswith('.json')]
        for f in test_files:
            file_path = os.path.join(test_data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            test_data.update(cdata['user_data'])

        clients = list(train_data.keys())
        test_client = list(test_data.keys())

        logging.info(f"the number of train_clients is {len(clients)}")
        logging.info(f"the number of test_clients is {len(test_client)}")

        lens = []
        for iii, c in enumerate(clients):
            lens.append(len(train_data[c]['x']))

        dict_users_train = list(train_data.keys())
        dict_users_test = list(test_data.keys())
        for c in train_data.keys():
            train_data[c]['y'] = list(np.asarray(train_data[c]['y']).astype('int64'))
            test_data[c]['y'] = list(np.asarray(test_data[c]['y']).astype('int64'))
            # print(c, train_data[c])
        # print(lens)
        # print(clients)
        totoal_numbr_datapoints = 0
        totoal_numbr_test__datapoints = 0

        val_data = OrderedDict()
        # pub_data = OrderedDict()
        pub_data = None
        train_d = OrderedDict()
        test_g = OrderedDict()
        test_d = OrderedDict()
        set_y = set()
        weights = {}
        number_of_clients_per_cls = defaultdict(dict)

        all_federated_trainset = []
        all_federated_testset = []
        for i, dataset in enumerate(train_data.values()):
            all_federated_trainset.append(dataloader(dataset, i))
        for i, dataset in enumerate(test_data.values()):
            all_federated_testset.append(dataloader(dataset, i))
        all_worker_num = len(all_federated_trainset)

        worker_id_list = random.sample(range(all_worker_num), all_worker_num)
        print(worker_id_list)
        federated_trainset = []
        federated_testset = []
        for i in worker_id_list:
            federated_trainset.append(all_federated_trainset[i])
            federated_testset.append(all_federated_testset[i])

        worker_num = len(list(train_data.keys()))
        federated_valset = [None] * worker_num
        for i in range(worker_num):
            n_samples = len(federated_trainset[i])
            if n_samples == 1:
                federated_valset[i] = copy.deepcopy(federated_trainset[i])
            else:
                X = federated_trainset[i].dataset['x']
                y = federated_trainset[i].dataset['y']

                class_counts = defaultdict(int)
                for label in y:
                    class_counts[label] += 1

                # Split the dataset ensuring each class has at least 2 members
                train_indices = []
                val_indices = []

                for label in set(y):
                    indices = [i for i, y in enumerate(y) if y == label]
                    n_samples = len(indices)
                    if n_samples >= 2:
                        train_subset, val_subset = train_test_split(indices, test_size=0.2,
                                                                    stratify=[y[i] for i in indices])
                        train_indices.extend(train_subset)
                        val_indices.extend(val_subset)

                # Use the selected indices to split the data
                train_X = [X[i] for i in train_indices]
                val_X = [X[i] for i in val_indices]
                train_y = [y[i] for i in train_indices]
                val_y = [y[i] for i in val_indices]
                federated_trainset[i] = dataloader({'x': train_X, 'y': train_y}, i)
                federated_valset[i] = dataloader({'x': val_X, 'y': val_y}, i)

                logging.info(f"The train labels for client {i} are {set(federated_trainset[i].dataset['y'])} "
                             f"and val labels are {set(federated_valset[i].dataset['y'])}")

            weights[i] = len(federated_trainset[i].dataset['y'])
        return clients, groups, federated_trainset, federated_testset, pub_data, federated_valset, test_g, weights

    def load_and_split(self, cfg, pub=False, local_test=False, val_pub=False, freq=False, use_val_set=False,
                       transformed_pub=None):
        logging.info(f"function setup data")

        """Instantiates clients based on given train and test data directories.

        Return:
            all_clients: list of Client objects.
        """
        eval_set = 'test' if not use_val_set else 'val'
        train_data_dir = os.path.join(eval("os.path.expanduser('~')"), 'leaf/data', "sent140", 'data', 'train')
        test_data_dir = os.path.join(eval("os.path.expanduser('~')"), 'leaf/data', "sent140", 'data', eval_set)
        logging.info(f"train directory is {train_data_dir}")

        logging.info(f"This is the train data directory {train_data_dir}")
        logging.info(f"This is the test data directory {test_data_dir}")

        clients_ids, groups, train_data, test_data, pub_data, val_data, global_test, weights = self.read_data(
            train_data_dir,
            test_data_dir)

        return train_data, val_data, None, test_data, global_test, pub_data, None, val_data, weights

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

        # val_data = dataloader(val_set, self.current_client_idx)
        val_loader = DataLoader(
            val_set[self.current_client_idx],
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=None,
        )
        return val_loader

    def global_test_loader(self, test_set):
        test_set = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return test_set

    def plot_data(self, train_set):
        dataiter = iter(train_set)
        images, labels = dataiter.next()
        plt.imshow(np.transpose(images[6].numpy(), (1, 2, 0)))
        plt.savefig("a.pdf")

    def set_client(self):
        self.current_client_idx = 0

    def next_client(self):
        self.current_client_idx += 1

# train_size = int(len(federated_trainset[i]) * 0.8)
# val_size = n_samples - train_size
# federated_trainset[i], federated_valset[i] = torch.utils.data.random_split(federated_trainset[i],
#                                                                            [train_size, val_size])
#
# logging.info(f"train is {federated_trainset[i]}")
