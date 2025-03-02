import itertools
import json
import logging
from collections import defaultdict, OrderedDict
from random import sample

import numpy as np
import matplotlib.pyplot as plt

import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.data.shakespeare_utils import process_x, process_y


class Shakespeare():

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
        logging.info(f"function read data")
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
        train_clients, train_groups, train_data = self.read_dir(train_data_dir)
        test_clients, test_groups, test_data = self.read_dir(test_data_dir)
        logging.info(f"the number of train_clients is {len(train_clients)}")
        logging.info(f"the number of train_data is {len(train_data)}")

        # logging.info(f"the number of data_groups is {train_groups}")
        public_data = OrderedDict()
        val_data = OrderedDict()
        train_dataset = OrderedDict()

        test_g = OrderedDict()
        test_d = OrderedDict()
        train_d = OrderedDict()
        t_d_len = 0
        v_d_len = 0
        weights = {}
        logging.info(f"the number of CLINETS is {len(train_clients)}")
        for i, u in enumerate(train_clients):

            array_x = np.array(train_data[u]['x'])
            array_y = np.array(train_data[u]['y'])

            array_test_x = np.array(test_data[u]['x'])
            array_test_y = np.array(test_data[u]['y'])

            X_train, X_pub, y_train, y_pub = train_test_split(array_x, array_y, test_size=0.001,
                                                              random_state=32)
            X_train, X_val, y_train, y_val = train_test_split(np.array(X_train), np.array(y_train), test_size=0.2,
                                                              random_state=32)

            train_dic = {}
            train_dic['x'] = X_train
            train_dic['y'] = y_train
            train_d[i] = train_dic

            pub_data = {}
            pub_data["x"] = X_pub
            pub_data["y"] = y_pub
            public_data[i] = pub_data
            dic = {}
            dic['x'] = list(X_train)
            dic['y'] = list(y_train)
            weights[i] = len(y_train)
            train_dataset[i] = dic
            val_dic = {}
            val_dic['x'] = X_val
            val_dic['y'] = y_val
            val_data[i] = val_dic

            test_dic = {}
            test_dic['x'] = array_test_x
            test_dic['y'] = array_test_y
            test_d[i] = test_dic

            if len(array_test_y) > 3:
                _, X_g_test, _, y_g_test = train_test_split(array_test_x, array_test_y, test_size=0.1, random_state=32)
                global_test_dic = {'x': X_g_test, 'y': y_g_test}
                test_g[i] = global_test_dic

            logging.info(
                f"lables for client {i} are {set(train_dataset[i]['y'])} and for test {set(test_d[i]['y'])}")
            logging.info(f"client {i} len data {len(train_dataset[i]['y'])}")
            logging.info(f"client {i} len test data {len(test_d[i]['y'])}")
            t_d_len += len(train_dataset[i]['y'])
            v_d_len += len(test_d[i]['y'])
        del train_data

        logging.info(f"the size of the train data is {t_d_len} and the size of the test data is {v_d_len}")
        assert train_clients == test_clients
        assert train_groups == test_groups

        test_g = test_g.values()
        test_global = defaultdict(list)
        for d in test_g:
            for key, value in d.items():
                test_global[key].append(value)

        test_global['x'] = list(itertools.chain.from_iterable(test_global['x']))
        test_global['y'] = list(itertools.chain.from_iterable(test_global['y']))

        # test_global = dict(test_global)
        test_g = self.dataloader(test_global)

        return train_clients, train_groups, train_d, test_d, public_data, val_data, test_g, weights

    def load_and_split(self, cfg, pub=False, local_test=False, val_pub=False, freq=False, use_val_set=False,
                       transformed_pub=None):

        logging.info(f"function setup data")
        """Instantiates clients based on given train and test data directories.

        Return:
            all_clients: list of Client objects.
        """
        eval_set = 'test' if not use_val_set else 'val'

        train_data_dir = os.path.join(eval("os.path.expanduser('~')"), 'leaf/data', "shakespeare", 'data', 'train')
        test_data_dir = os.path.join(eval("os.path.expanduser('~')"), 'leaf/data', "shakespeare", 'data', eval_set)

        # train_data_dir = os.path.join(eval("os.path.expanduser('~')"), 'HetoFL/FairFL/data', "shakespeare", 'data', 'train')
        # test_data_dir = os.path.join(eval("os.path.expanduser('~')"), 'HetoFL/FairFL/data', "shakespeare", 'data', eval_set)

        logging.info(f"This is the train data directory {train_data_dir}")
        logging.info(f"This is the test data directory {test_data_dir}")

        clients_ids, groups, train_data, test_data, pub_data, val_data, global_test, weights = self.read_data(
            train_data_dir, test_data_dir)
        return train_data, val_data, None, test_data, global_test, pub_data, None, val_data, weights

    def plot_data(self, train_set):
        dataiter = iter(train_set)
        images, labels = dataiter.next()
        plt.imshow(np.transpose(images[6].numpy(), (1, 2, 0)))
        plt.savefig("a.pdf")

    def dataloader(self, dataset, client_id=None):
        if client_id is not None:
            data = TensorDataset(process_x(dataset[client_id]["x"]), process_y(dataset[client_id]["y"]))
        else:
            data = TensorDataset(process_x(dataset["x"]), process_y(dataset["y"]))
        return data

    def train_loaders(self, train_set):
        logging.info(f"the cient id is {self.current_client_idx}")
        train_set = self.dataloader(dataset=train_set, client_id=self.current_client_idx)
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
        val_data = self.dataloader(val_set, self.current_client_idx)
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

    def set_client(self):
        self.current_client_idx = 0

    def next_client(self):
        self.current_client_idx += 1
