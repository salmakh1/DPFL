# -*- coding: utf-8 -*-
import csv
import logging

import random
import time
from collections import OrderedDict
from random import Random

import numpy as np
import torch

from torch.utils.data import DataLoader

#set up the data generator to have consistent results
seed = 10
generator = torch.Generator()
generator.manual_seed(seed)

def seed_worker(worker_id):
    worker_seed = seed #torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Partition(object):
    """ Dataset partitioning helper """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """Partition data by trace or random"""

    def __init__(self, data, seed=10, isTest=False, isVal=False):
        self.partitions = []
        self.rng = Random()
        self.rng.seed(seed)

        self.data = data
        self.labels = self.data.targets

        self.isTest = isTest
        self.isVal = isVal
        np.random.seed(seed)

        self.data_len = len(self.data)
        # logging.info(f"the length of the data is {self.data_len}")
        self.numOfLabels = len(set(self.data.targets))
        self.filter_class=1

        self.targets = OrderedDict()
        self.indexToLabel = {}

        self.usedSamples = -1
        for index, label in enumerate(self.labels):
            if label not in self.targets:
                self.targets[label] = []

            self.targets[label].append(index)
            self.indexToLabel[index] = label

    def getNumOfLabels(self):
        return self.numOfLabels

    def getDataLen(self):
        return self.data_len

    def trace_partition(self, data_map_file, ratio=1.0):
        """Read data mapping from data_map_file. Format: <client_id, sample_name, sample_category, category_id>"""
        logging.info(f"Partitioning data by profile {data_map_file}...")

        clientId_maps = {}
        unique_clientIds = {}
        # load meta data from the data_map_file
        with open(data_map_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            read_first = True
            sample_id = 0

            for row in csv_reader:
                # if 'data/raw_data/by_write/hsf_0/' in row[1]:
                if read_first:
                    logging.info(f'Trace names are {", ".join(row)}')
                    read_first = False
                else:
                    client_id = row[0]

                    if client_id not in unique_clientIds:
                        unique_clientIds[client_id] = len(unique_clientIds)

                    clientId_maps[sample_id] = unique_clientIds[client_id]
                    sample_id += 1

        # Partition data given mapping
        self.partitions = [[] for _ in range(len(unique_clientIds))]

        for idx in range(len(self.data.data)):
            self.partitions[clientId_maps[idx]].append(idx)

        # for i in range(len(unique_clientIds)):
        #     self.rng.shuffle(self.partitions[i])
        #     takelen = max(0, int(len(self.partitions[i]) * ratio))
        #     self.partitions[i] = self.partitions[i][:takelen]
        #     logging.info(f"############### client_id = {i} have datapoints {len(self.partitions[i])}")

    def partition_data_helper(self, num_clients, data_map_file=None):

        # read mapping file to partition trace
        logging.info(f"data map file {data_map_file}" )
        if data_map_file is not None:
            self.trace_partition(data_map_file)
        else:
            self.uniform_partition(num_clients=num_clients)

    def uniform_partition(self, num_clients):
        # random partition
        numOfLabels = self.getNumOfLabels()
        data_len = self.getDataLen()
        logging.info(f"Randomly partitioning data, {data_len} samples...")

        indexes = list(range(data_len))
        self.rng.shuffle(indexes)

        for _ in range(num_clients):
            part_len = int(1./num_clients * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]
        return self.partitions[0]

    def create_mapping(self, sizes):
        numOfLabels = self.getNumOfLabels()

        # ratioOfClassWorker = None
        # if self.args.partitioning == 1:
        ratioOfClassWorker = np.random.rand(len(sizes), numOfLabels).astype(np.float32)
        # elif self.args.partitioning == 2:
        #     ratioOfClassWorker = np.random.zipf(self.args.zipf_param, [len(sizes), numOfLabels]).astype(np.float32)
        # elif self.args.partitioning == 3:
        #     ratioOfClassWorker = np.ones((len(sizes), numOfLabels)).astype(np.float32)

        if self.filter_class > 0:
            for w in range(len(sizes)):
                # randomly filter classes by forcing zero samples
                wrandom = self.rng.sample(range(numOfLabels), self.filter_class)
                for wr in wrandom:
                    ratioOfClassWorker[w][wr] = 0.001

        logging.info("==== Class per worker p:{} s:{} l:{} c:{} ====\n {} \n".format(1, len(sizes),
                                                                                     numOfLabels, np.count_nonzero(
                ratioOfClassWorker), repr(ratioOfClassWorker)))
        return ratioOfClassWorker


    def getTargets(self):
        tempTarget = self.targets.copy()

        #TODO:why the temp targets are reshuffled each time getTargets is called?
        for key in tempTarget:
            self.rng.shuffle(tempTarget[key])

        return tempTarget

    def custom_partition(self, num_clients):
        # custom partition
        numOfLabels = self.getNumOfLabels()
        data_len = self.getDataLen()
        sizes = [1.0 / num_clients for _ in range(num_clients)]

        # get # of samples per worker
        # get number of samples per worker
        if self.usedSamples <= 0:
            self.usedSamples = int(data_len / num_clients)
        self.usedSamples = max(self.usedSamples, self.numOfLabels)

        # get targets
        targets = self.getTargets()
        keyDir = {key: int(key) for i, key in enumerate(targets.keys())}
        keyLength = [0] * numOfLabels
        for key in keyDir.keys():
            keyLength[keyDir[key]] = len(targets[key])

        # logging.info(
        #     f"Custom partitioning data, {data_len} samples of {numOfLabels} labels on {num_clients} clients ...")

        ratioOfClassWorker = self.create_mapping(sizes)
        if ratioOfClassWorker is None:
            return self.uniform_partition(num_clients=num_clients)

        sumRatiosPerClass = np.sum(ratioOfClassWorker, axis=1)
        for worker in range(len(sizes)):
            ratioOfClassWorker[worker, :] = ratioOfClassWorker[worker, :] / float(sumRatiosPerClass[worker])

        # classPerWorker -> Rows are workers and cols are classes
        tempClassPerWorker = np.zeros([len(sizes), numOfLabels])
        # split the classes
        for worker in range(len(sizes)):
            self.partitions.append([])
            # enumerate the ratio of classes it should take
            for c in list(targets.keys()):
                # takeLength = min(int(ceil(keyLength[keyDir[c]] * ratioOfClassWorker[worker][keyDir[c]])), len(targets[c]))
                takeLength = min(int(self.usedSamples * ratioOfClassWorker[worker][keyDir[c]]), keyLength[keyDir[c]])
                self.rng.shuffle(targets[c])
                self.partitions[-1] += targets[c][0:takeLength]
                tempClassPerWorker[worker][keyDir[c]] += takeLength
                logging.info(f"the length of this partition is {len(self.partitions[-1]) }")
            self.rng.shuffle(self.partitions[-1])
        logging.info(f"the number of partitions is {len(self.partitions)}")
        del tempClassPerWorker

    def partition_test(self):
        data_len = self.getDataLen()
        logging.info(f"Randomly partitioning data, {data_len} samples...")

        indexes = list(range(data_len))
        self.rng.shuffle(indexes)

        part_len = int(data_len)
        logging.info(f"the length is {part_len} {len(indexes[0:part_len])}")
        self.partitions.append(indexes[0:part_len])




    def partition_public(self,num_clients):
        data_len = self.getDataLen()
        logging.info(f"Randomly partitioning data, {data_len} samples...")

        indexes = list(range(data_len))
        self.rng.shuffle(indexes)

        part_len_pub = int((0.5) * data_len)
        logging.info(f"the length is {part_len_pub} {len(indexes[0:part_len_pub])}")
        self.partitions.append(indexes[0:part_len_pub])

        part_len = int(1.*(data_len-part_len_pub) / num_clients )
        logging.info(f"{part_len}, {num_clients}, {data_len-part_len_pub}")
        indexes = indexes[part_len_pub:]
        logging.info(f"{len(indexes)}")
        for i in range(num_clients):
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]
            # logging.info(len(indexes))

        # logging.info(self.partitions)
    def use(self, partition, istest):
        # if istest:
            # logging.info(f"#######The number of partions is ###### {len(self.partitions)}")
        resultIndex = self.partitions[partition]

        exeuteLength = len(resultIndex) if not istest else int(
            len(resultIndex) * 1)
        resultIndex = resultIndex[:exeuteLength]
        self.rng.shuffle(resultIndex)

        return Partition(self.data, resultIndex)

