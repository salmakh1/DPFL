import itertools
import json
import logging
from collections import defaultdict, OrderedDict
from random import sample

import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

import os
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from src.data.femnist_dataloader import dataloader


class Femnist():

    def __init__(self, seed, num_classes, batch_size, num_workers, shuffle, num_clients, m, available_clients,local_steps):
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

        train_files = os.listdir(train_data_dir)
        train_files = ["mytrain.json"]
        for f in train_files:
            file_path = os.path.join(train_data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            clients.extend(cdata['users'])
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            train_data.update(cdata['user_data'])

        test_files = os.listdir(test_data_dir)
        test_files = ["mytest.json"]
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
        print(lens)
        print(clients)
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
        for i, c in enumerate(list(train_data.keys())):
            train_data[c]['y'] = list(np.asarray(train_data[c]['y']).astype('int64'))
            set_y.update(train_data[c]['y'])
            test_data[c]['y'] = list(np.asarray(test_data[c]['y']).astype('int64'))

            array_x = np.array(train_data[c]['x'])
            array_y = np.array(train_data[c]['y'])
            X_train, X_val, y_train, y_val = train_test_split(array_x, array_y, test_size=0.2, random_state=32)

            val_dic = {}
            val_dic['x'] = X_val
            val_dic['y'] = y_val
            val_data[i] = val_dic

            train_dic = {}
            train_dic['x'] = X_train
            train_dic['y'] = y_train
            train_d[i] = train_dic
            weights[i] = len(y_train)
            array_test_x = np.array(test_data[c]['x'])
            array_test_y = np.array(test_data[c]['y'])

            test_dic = {}
            test_dic['x'] = array_test_x
            test_dic['y'] = array_test_y
            test_d[i] = test_dic

            if len(array_test_y) > 3:
                _, X_g_test, _, y_g_test = train_test_split(array_test_x, array_test_y, test_size=0.1, random_state=32)
                global_test_dic = {}
                global_test_dic['x'] = X_g_test
                global_test_dic['y'] = y_g_test
                test_g[i] = global_test_dic

            totoal_numbr_datapoints += len(train_d[i]['y'])
            totoal_numbr_test__datapoints += len(test_d[i]['y'])

            logging.info(
                f"lables for client {i} are {set(train_d[i]['y'])} and for test {set(test_d[i]['y'])}")
            logging.info(f"client {i} len data {len(train_d[i]['y'])}")
            logging.info(f"client {i} len test data {len(test_d[i]['y'])}")

            for cls in set(train_d[i]['y']):
                number_of_clients_per_cls[i][cls] = len([j for j in train_d[i]['y'] if j == cls])

        test_g = test_g.values()
        test_global = defaultdict(list)
        for d in test_g:
            for key, value in d.items():
                test_global[key].append(value)

        test_global['x'] = list(itertools.chain.from_iterable(test_global['x']))
        test_global['y'] = list(itertools.chain.from_iterable(test_global['y']))

        test_global = dict(test_global)
        test_g = dataloader(test_global["x"], test_global["y"])

        logging.info(f"the total number of train data is {totoal_numbr_datapoints}")
        logging.info(f"the total number of test data is {totoal_numbr_test__datapoints}")

        return clients, groups, train_d, test_d, pub_data, val_data, test_g, weights

    def load_and_split(self, cfg, pub=False, local_test=False, val_pub=False, freq=False, use_val_set=False,
                       transformed_pub=None):
        logging.info(f"function setup data")

        """Instantiates clients based on given train and test data directories.

        Return:
            all_clients: list of Client objects.
        """
        eval_set = 'test' if not use_val_set else 'val'
        train_data_dir = os.path.join(eval("os.path.expanduser('~')"), 'leaf/data', "femnist", 'data', 'train')
        test_data_dir = os.path.join(eval("os.path.expanduser('~')"), 'leaf/data', "femnist", 'data', eval_set)
        logging.info(f"train directory is {train_data_dir}")

        logging.info(f"This is the train data directory {train_data_dir}")
        logging.info(f"This is the test data directory {test_data_dir}")

        clients_ids, groups, train_data, test_data, pub_data, val_data, global_test, weights = self.read_data(train_data_dir,
                                                                                                     test_data_dir)

        return train_data, val_data, None, test_data, global_test, pub_data, None, val_data, weights

    def train_loaders(self, train_set):
        train_set = dataloader(train_set[self.current_client_idx]["x"], train_set[self.current_client_idx]["y"])
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=None,
        )
        return train_loader

    def val_loaders(self, val_set):

        val_data = dataloader(val_set[self.current_client_idx]["x"], val_set[self.current_client_idx]["y"])
        val_loader = DataLoader(
            val_data,
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
