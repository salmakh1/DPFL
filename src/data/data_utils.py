import copy
import itertools
import random
import time
from collections import defaultdict

import torch
from torch.utils.data import random_split, Subset, Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
import scipy.stats as stats
from itertools import chain
import logging
from sklearn.model_selection import train_test_split
import random
from PIL import Image

import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


# This function will split the full data to all clients

def new_getattr(self, name):
    """Search recursively for attributes under self.dataset."""
    dataset = self
    if name[:2] == "__":
        raise AttributeError(name)
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
        if hasattr(dataset, name):
            return getattr(dataset, name)
    raise AttributeError(name)


def split_dataset_train_val(train_dataset, val_split, seed, val_dataset=None):
    targets = train_dataset.targets
    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split, stratify=targets, random_state=seed
    )
    train_subset = Subset(train_dataset, indices=train_idx)
    val_subset = Subset(val_dataset if val_dataset else train_dataset, indices=val_idx)
    train_subset.__class__.__getattr__ = new_getattr
    val_subset.__class__.__getattr__ = new_getattr  # trick to get the attrs of the original dataset
    return train_subset, val_subset


# def split_subsets_train_val(subsets, val_precent, seed, val_dataset: Dataset = None):
#     """
#     split clients subsets into train/val sets
#     Args:
#         val_dataset: give if you have a val dataset that have different transforms than the train dataset
#     """
#     train_sets = []
#     val_sets = []
#     for subset in subsets:
#         # logging.info(f"{len(subset.indices)}")
#         # train_indices, val_indices = train_test_split(subset.indices, test_size=val_precent, random_state=seed)
#         labels = [subset.dataset.targets[i] for i in subset.indices] if hasattr(subset.dataset, 'targets') else None
#         train_indices, val_indices = train_test_split(subset.indices, test_size=val_precent, stratify=labels, random_state=seed)
#         train_subset = copy.deepcopy(subset)
#         train_subset.indices = train_indices
#
#         if val_dataset:
#             val_subset = Subset(val_dataset, indices=val_indices)
#         else:
#             val_subset = copy.deepcopy(subset)
#             val_subset.indices = val_indices
#
#         train_subset.__class__.__getattr__ = new_getattr
#         val_subset.__class__.__getattr__ = new_getattr  # trick to get the attrs of the original dataset
#
#         train_sets.append(train_subset)
#         val_sets.append(val_subset)
#     # ensure that all indices are disjoints and splits are correct
#     assert all((set(p0).isdisjoint(set(p1))) for p0, p1 in itertools.combinations([s.indices for s in train_sets], 2))
#     assert all((set(p0).isdisjoint(set(p1))) for p0, p1 in itertools.combinations([s.indices for s in val_sets], 2))
#     assert all(
#         (set(p0).isdisjoint(set(p1))) for p0, p1 in itertools.combinations(
#             [s.indices for s in itertools.chain(train_sets, val_sets)],
#             2
#         )
#     )
#     return train_sets, val_sets


from torch.utils.data import Subset, Dataset
import copy
import itertools
from sklearn.model_selection import train_test_split


def split_subsets_train_val(subsets, val_percent, seed, val_dataset: Dataset = None):
    train_sets = []
    val_sets = []

    for subset in subsets:
        labels = [subset.dataset.targets[i] for i in subset.indices] if hasattr(subset.dataset, 'targets') else None

        # Calculate the minimum number of samples per class
        min_samples_per_class = min(labels.count(label) for label in set(labels))

        # Ensure that each class has at least two samples
        if min_samples_per_class < 2:
            # Skip this subset if any class has less than two samples
            train_indices, val_indices = train_test_split(subset.indices, test_size=val_percent,
                                                          random_state=seed)
        else:
            # Ensure that test_size is at least equal to the number of classes
            adjusted_val_percent = max(val_percent, min_samples_per_class / len(subset.indices))

            # Split indices based on labels to ensure the same labels and proportions in train and val sets
            train_indices, val_indices = train_test_split(subset.indices, test_size=val_percent, stratify=labels,
                                                          random_state=seed)

        # Create train subset
        train_subset = copy.deepcopy(subset)
        train_subset.indices = train_indices

        # Create val subset
        if val_dataset:
            val_subset = Subset(val_dataset, indices=val_indices)
        else:
            val_subset = copy.deepcopy(subset)
            val_subset.indices = val_indices

        train_subset.__class__.__getattr__ = new_getattr
        val_subset.__class__.__getattr__ = new_getattr  # trick to get the attrs of the original dataset

        train_sets.append(train_subset)
        val_sets.append(val_subset)

    # Ensure that all indices are disjoint and splits are correct
    # assert all((set(p0).isdisjoint(set(p1))) for p0, p1 in itertools.combinations([s.indices for s in train_sets], 2))
    # assert all((set(p0).isdisjoint(set(p1))) for p0, p1 in itertools.combinations([s.indices for s in val_sets], 2))
    # assert all(
    #     (set(p0).isdisjoint(set(p1))) for p0, p1 in itertools.combinations(
    #         [s.indices for s in itertools.chain(train_sets, val_sets)],
    #         2
    #     )
    # )

    return train_sets, val_sets


def random_split_data_to_clients(dataset, num_clients, seed, min_dataset_size=0):
    """
    Plain random data split amoung clients
    args:
    dataset: pytorch dataset object
    num_clients: int
    seed: int for fixing the splits
    Returns:
    List of Dataset subset object of length=num_clients
    """
    cls_frequencies = None
    weights = {}
    percentage_client_indices_per_class = defaultdict(lambda: defaultdict(int))
    idx_batch = [[] for _ in range(num_clients)]
    indices = np.array(dataset.indices) if isinstance(dataset, Subset) else np.arange(len(dataset))
    targets = dataset.dataset.targets if isinstance(dataset, Subset) else dataset.targets
    targets = targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
    df = pd.DataFrame({"target": targets[indices]}, index=indices)
    label_to_indices = {}
    for label, group in df.groupby('target'):
        label_to_indices[label] = group.index
    labels, classes_count_ = np.unique(df.target, return_counts=True)
    classes_count = defaultdict(int)
    for label, count in zip(labels, classes_count_):
        classes_count[label] = count
    num_classes = len(df.target.unique())

    if min_dataset_size == 0:
        logging.info(f"here is  0")
        ds_len = len(dataset)
        split_sizes = [
            ds_len // num_clients if i != num_clients - 1 else ds_len - (ds_len // num_clients * (num_clients - 1))
            for i in range(num_clients)
        ]
        weights = {i: split_sizes[i] for i in range(len(split_sizes))}
        logging.info(f"split_sizes {split_sizes}")
        assert ds_len == sum(split_sizes)
        gen = torch.Generator().manual_seed(seed)  # to preserve the same split everytime
        datasets = random_split(dataset=dataset, lengths=split_sizes, generator=gen)

        for k, subset in enumerate(datasets):
            idx_batch[k] = [subset.indices]
        class_dis = np.zeros((num_clients, num_classes))
        for j in range(num_clients):
            for m in range(num_classes):
                class_dis[j, m] = int((np.array(targets[idx_batch[j]]) == m).sum())
                # logging.info(f"class dis {class_dis[j, m]}")
                percentage_client_indices_per_class[j][m] = class_dis[j, m] / classes_count[m]
            logging.info(
                f"the classes for client {j} are {np.nonzero(class_dis[j, :])[0]}"
                f"and the number is {class_dis[j, np.nonzero(class_dis[j, :])[0]]}")

        assert all(
            (set(p0).isdisjoint(set(p1))) for p0, p1 in itertools.combinations([ds.indices for ds in datasets], 2))
    else:
        a = min_dataset_size * num_clients
        indices = np.random.permutation(len(dataset))[:a]
        datasets = Subset(dataset, indices)
    logging.info(f"len of datasets is {len(datasets)}")
    return datasets, percentage_client_indices_per_class, weights


####### the parameter missing will identify the number of missing classes##########
# Salma: if missing = number_classes -1 then each client will have only one class.
# we should split the train in this way but keep the test random split.


def generate_zipf_distribution(train_data, num_clients, missing, a=1.1, size=0):
    targets = train_data.dataset.targets
    x = np.arange(1, max(list(targets)) + 1 - missing)
    weights = x ** (-a)
    weights /= weights.sum()
    bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))
    # TODO: make the number of datapoints smaller for every client 250 - 500 datapoints
    if size == 0:
        size = int(len(train_data) / num_clients)
        sample = bounded_zipf.rvs(size=size)
    else:
        sample = bounded_zipf.rvs(size=size)
    return sample


def count_elements_from_distr(sample, targets, missing):
    elemet_counts = []
    sample = list(sample)
    for i in range(1, targets + 1 - missing):
        elemet_counts.append(sample.count(i))
    return elemet_counts


def generate_permutation(num_classes, num_clients, missing):
    labels = [*range(0, num_classes, 1)]
    perms = []
    for i in range(num_clients):
        random.seed(i + 10)
        perms.append(random.sample(labels, len(labels) - missing))
    # perm = itertools.permutations(np.arange(0, num_classes))
    print(perms)
    return perms


def split_data_by_idx(dataset):
    indices = np.array(dataset.indices) if isinstance(dataset, Subset) else np.arange(len(dataset))
    targets = dataset.dataset.targets if isinstance(dataset, Subset) else dataset.targets
    targets = targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
    df = pd.DataFrame({"target": targets[indices]}, index=indices)
    label_to_indices = {}
    # Map indices to classes (labels, targets)
    for label, group in df.groupby('target'):
        label_to_indices[label] = group.index
    return label_to_indices


def split_nniid(dataset, num_clients, missing, a, data_points):
    logging.info(f"here  {dataset}")
    targets = torch.tensor(dataset.dataset.targets) if isinstance(dataset, Subset) else dataset.targets
    num_classes = len(targets.unique())
    label_to_indices = split_data_by_idx(dataset)
    logging.info("label_to_indices {}".format(label_to_indices))
    perm = generate_permutation(num_classes, num_clients, missing)
    datasets = []
    clients_data = []
    # d={}
    if data_points == 0:
        data_points = len(targets) / num_clients
    for i in range(num_clients):
        samples = generate_zipf_distribution(dataset, num_clients, missing, a, data_points)
        count_samples = count_elements_from_distr(samples, num_classes, missing)
        log.info("elemet_counts {} for client {}".format(count_samples, i))
        client_data = []
        for j in range(len(count_samples)):
            np.random.seed(i + 10 + j)
            sub = np.random.choice(label_to_indices[perm[i][j]], size=count_samples[j], replace=False)
            sub = list(sub)
            logging.info(sub)
            client_data.append(sub)
        client_data_ = list(chain.from_iterable(client_data))
        clients_data.append(client_data_)

    for data in clients_data:
        datasets.append(Subset(dataset, data))
    print("datasets", datasets)
    return datasets


def split_three_clients(
        dataset, num_clients, seed, min_dataset_size, transformed_dataset=None, alpha=0.5, missing_classes=0.5
):
    """
    Splits a dataset into three clients with specified class distributions.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        seed (int): Random seed for reproducibility.

    Returns:
        list of torch.utils.data.Subset: List containing subsets of the dataset for each client.
        dict: Percentage of samples per class for each client.
        dict: Number of samples per client.
    """
    np.random.seed(seed)

    # Define class distributions for each client
    client_class_distribution = {
        0: {"classes": [0, 1, 2, 3, 4], "num_samples": [500, 500, 500, 500, 500]},
        1: {"classes": [0, 1, 2, 5, 6], "num_samples": [100, 100, 100, 400, 400]},
        2: {"classes": [3, 4, 5, 6, 7], "num_samples": [400, 400, 400, 100, 100]}
    }

    # Initialize clients data and class percentages
    clients_data = []
    percentage_client_indices_per_class = defaultdict(dict)
    client_weights = {}

    # Loop through each client
    for client_name, client_info in client_class_distribution.items():
        client_classes = client_info["classes"]
        num_samples_per_class = client_info["num_samples"]

        client_indices = []
        class_counts = defaultdict(int)

        # Sample data for each class in the client
        for class_idx, class_id in enumerate(client_classes):
            class_indices = np.where(np.array(dataset.targets) == class_id)[0]
            class_indices = np.random.choice(class_indices, num_samples_per_class[class_idx], replace=False)
            client_indices.extend(class_indices)
            class_counts[class_id] = num_samples_per_class[class_idx]

        # Calculate percentage of samples per class for the client
        total_samples = sum(num_samples_per_class)
        for class_id, class_count in class_counts.items():
            percentage_client_indices_per_class[client_name][class_id] = class_count / total_samples

        # Store client data
        clients_data.append(torch.utils.data.Subset(dataset, client_indices))
        client_weights[client_name] = len(client_indices)

    logging.info(f"client_weights {client_weights}")

    return clients_data, percentage_client_indices_per_class, client_weights


# def split_three_clients(
#         dataset, num_clients, seed, min_dataset_size, transformed_dataset=None, alpha=0.5, missing_classes=0.5
# ):
#     print("split_three_clients function")
#     np.random.seed(seed)
#     weights = {}
#
#     indices = np.array(dataset.indices) if isinstance(dataset, Subset) else np.arange(len(dataset))
#     targets = dataset.dataset.targets if isinstance(dataset, Subset) else dataset.targets
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
#
#     percentage_client_indices_per_class = defaultdict(dict)
#
#     number_of_samples = len(targets)
#     indices = list(range(number_of_samples))
#     np.random.shuffle(indices)
#
#     used_indices = indices[:int(0.8* number_of_samples)]
#     used_indices_ = torch.utils.data.Subset(dataset, used_indices)
#     classes = np.random.choice(range(10), size=10, replace=False)
#     client1_indices = []
#     client2_indices = []
#     client3_indices = []
#     logging.info(f"classes are {classes}")
#     number_of_ind_per_label = {i: 0 for i in range(len(classes))}
#     class_counts = {i: {class_idx: 0 for class_idx in classes} for i in range(3)}
#
#     for idx in used_indices_.indices:
#         _, label = dataset[idx]
#         if label in classes:
#             # if label == 4 or label == 8 or label == 2 and number_of_ind_per_label[label] <= 180:
#             #     client1_indices.append(idx)
#             #     class_counts[0][label] += 1
#             #     number_of_ind_per_label[label] += 1
#
#             if label == 0 or label == 1 or label == 2 or label == 3 or label == 4 and number_of_ind_per_label[label] <= 600:
#                 client1_indices.append(idx)
#                 class_counts[0][label] += 1
#                 number_of_ind_per_label[label] += 1
#
#
#     logging.info(f"number_of_ind_per_label {number_of_ind_per_label}")
#     remaining_indices = list(set(used_indices) - set(client1_indices))
#     remaining_data = torch.utils.data.Subset(dataset, remaining_indices)
#
#     np.random.shuffle(remaining_data.indices)
#
#     # Separate data for each selected class
#
#
#     number_of_ind_per_label = {i: 0 for i in range(len(classes))}
#
#     for idx in remaining_data.indices:
#         _, label = dataset[idx]
#         if label in classes:
#             # print(label)
#             if label == 0 or label == 1 or label == 2 and number_of_ind_per_label[label] <= 200:
#                 client2_indices.append(idx)
#                 class_counts[1][label] += 1
#                 number_of_ind_per_label[label] += 1
#
#             if label == 5 or label == 6 and number_of_ind_per_label[label] <= 500:
#                 client2_indices.append(idx)
#                 class_counts[1][label] += 1
#                 number_of_ind_per_label[label] += 1
#
#     logging.info(f"number_of_ind_per_label {number_of_ind_per_label}")
#     number_of_ind_per_label = {i: 0 for i in range(len(classes))}
#     remaining_indices = list(set(remaining_data.indices) - set(client2_indices))
#     remaining_data = torch.utils.data.Subset(dataset, remaining_indices)
#
#     np.random.shuffle(remaining_data.indices)
#     for idx in remaining_data.indices:
#         _, label = dataset[idx]
#         if label in classes:
#             if label == 3 or label == 4 or label == 5 and number_of_ind_per_label[label] <= 500:
#                 client3_indices.append(idx)
#                 class_counts[2][label] += 1
#                 number_of_ind_per_label[label] += 1
#
#             elif label == 6 or label == 7 and number_of_ind_per_label[label] <= 200:
#                 client3_indices.append(idx)
#                 class_counts[2][label] += 1
#                 number_of_ind_per_label[label] += 1
#
#
#
#     weights[0] = len(client1_indices)
#     weights[1] = len(client2_indices)
#     weights[2] = len(client3_indices)
#
#     client1_data = torch.utils.data.Subset(dataset, client1_indices)
#     client2_data = torch.utils.data.Subset(dataset, client2_indices)
#     client3_data = torch.utils.data.Subset(dataset, client3_indices)
#
#     # label_counts = {label: 0 for label in range(10)}
#     datasets = [client1_data, client2_data, client3_data]
#     for i, dataloader in enumerate(datasets):
#         # for data, label in dataloader:
#         #     label_counts[label] += 1
#         # unique_labels = list(label_counts.keys())
#         for label in classes:
#             percentage_client_indices_per_class[i][label] = class_counts[i][label] / classes_count[label]
#
#             logging.info(
#                 f"the classes for client {i}"
#                 f"and {label} the number is {class_counts[i][label]}")
#
#     log.info({f"client_{i}": len(ds.indices) for i, ds in enumerate(datasets)})
#
#     return datasets, percentage_client_indices_per_class, weights


# def split_three_clients(
#         dataset, num_clients, seed, min_dataset_size, transformed_dataset=None, alpha=0.5, missing_classes=0.5
# ):
#     print("split_three_clients function")
#     np.random.seed(seed)
#     weights = {}
#
#     indices = np.array(dataset.indices) if isinstance(dataset, Subset) else np.arange(len(dataset))
#     targets = dataset.dataset.targets if isinstance(dataset, Subset) else dataset.targets
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
#
#     percentage_client_indices_per_class = defaultdict(dict)
#
#     number_of_samples = len(targets)
#     indices = list(range(number_of_samples))
#     np.random.shuffle(indices)
#
#     used_indices = indices[:int(0.8* number_of_samples)]
#     used_indices_ = torch.utils.data.Subset(dataset, used_indices)
#     classes = np.random.choice(range(10), size=10, replace=False)
#     client1_indices = []
#     logging.info(f"classes are {classes}")
#     number_of_ind_per_label = {i: 0 for i in range(len(classes))}
#     class_counts = {i: {class_idx: 0 for class_idx in classes} for i in range(3)}
#
#     for idx in used_indices_.indices:
#         _, label = dataset[idx]
#         if label in classes:
#             # if label == 4 or label == 8 or label == 2 and number_of_ind_per_label[label] <= 180:
#             #     client1_indices.append(idx)
#             #     class_counts[0][label] += 1
#             #     number_of_ind_per_label[label] += 1
#
#             if label == 3 or label == 5 and number_of_ind_per_label[label] <= 150:
#                 client1_indices.append(idx)
#                 class_counts[0][label] += 1
#                 number_of_ind_per_label[label] += 1
#             # if label % 2 == 0 and number_of_ind_per_label[label] <= 150:
#             #     if label == 2:
#             #         continue
#             #     if label == 4 or label == 8:
#             #         if number_of_ind_per_label[label] > 50:
#             #             continue
#             #         else:
#             #             class_counts[0][label] += 1
#             #             number_of_ind_per_label[label] += 1
#             #             client1_indices.append(idx)
#             #     else:
#             #         class_counts[0][label] += 1
#             #         number_of_ind_per_label[label] += 1
#             #         client1_indices.append(idx)
#
#     # client1_indices = indices[:number_of_samples // 2]
#
#     # Divide the remaining data into two parts for clients 2 and 3
#     # remaining_indices = indices[number_of_samples // 2:]
#
#     remaining_indices = list(set(used_indices) - set(client1_indices))
#     remaining_data = torch.utils.data.Subset(dataset, remaining_indices)
#
#     np.random.shuffle(remaining_data.indices)
#
#     # Separate data for each selected class
#     client2_indices = []
#     client3_indices = []
#
#     number_of_ind_per_label = {i: 0 for i in range(len(classes))}
#
#     for idx in remaining_data.indices:
#         _, label = dataset[idx]
#         if label in classes:
#             # print(label)
#             if label == 3 and number_of_ind_per_label[label] <= 200:
#                 client2_indices.append(idx)
#                 class_counts[1][label] += 1
#                 number_of_ind_per_label[label] += 1
#
#             if label == 0 and number_of_ind_per_label[label] <= 150:
#                 client2_indices.append(idx)
#                 class_counts[1][label] += 1
#                 number_of_ind_per_label[label] += 1
#
#             # elif label == 1 and number_of_ind_per_label[label] <= 150:
#             #     client2_indices.append(idx)
#             #     class_counts[1][label] += 1
#             #     number_of_ind_per_label[label] += 1
#             #
#             # elif label == 3 and number_of_ind_per_label[label] <= 100:
#             #     client2_indices.append(idx)
#             #     class_counts[1][label] += 1
#             #     number_of_ind_per_label[label] += 1
#
#
#     number_of_ind_per_label = {i: 0 for i in range(len(classes))}
#
#     for idx in remaining_data.indices:
#         _, label = dataset[idx]
#         if label in classes:
#             if label == 5 and number_of_ind_per_label[label] <= 200:
#                 client3_indices.append(idx)
#                 class_counts[2][label] += 1
#                 number_of_ind_per_label[label] += 1
#
#             # if label == 6 and number_of_ind_per_label[label] <= 200:
#             #     client3_indices.append(idx)
#             #     class_counts[2][label] += 1
#             #     number_of_ind_per_label[label] += 1
#
#             elif label == 0 and number_of_ind_per_label[label] <= 150:
#                 client3_indices.append(idx)
#                 class_counts[2][label] += 1
#                 number_of_ind_per_label[label] += 1
#
#
#
#             # elif label == 5 and number_of_ind_per_label[label] <= 150:
#             #     client3_indices.append(idx)
#             #     class_counts[2][label] += 1
#             #     number_of_ind_per_label[label] += 1
#
#
#
#     # for idx in remaining_data.indices:
#     #     _, label = dataset[idx]
#     #     if label in classes:
#     #         # print(label)
#     #         if label == 0 and number_of_ind_per_label[label] <= 300:
#     #             client2_indices.append(idx)
#     #             class_counts[1][label] += 1
#     #             number_of_ind_per_label[label] += 1
#     #         elif label == 6 and number_of_ind_per_label[label] <= 300:
#     #             client2_indices.append(idx)
#     #             class_counts[1][label] += 1
#     #             number_of_ind_per_label[label] += 1
#     #         elif label == 1 and number_of_ind_per_label[label] <= 200:
#     #             client2_indices.append(idx)
#     #             class_counts[1][label] += 1
#     #             number_of_ind_per_label[label] += 1
#     #         elif label == 3 and number_of_ind_per_label[label] <= 200:
#     #             client2_indices.append(idx)
#     #             class_counts[1][label] += 1
#     #             number_of_ind_per_label[label] += 1
#     #         elif label == 4 and number_of_ind_per_label[label] <= 300:
#     #             client3_indices.append(idx)
#     #             class_counts[2][label] += 1
#     #             number_of_ind_per_label[label] += 1
#     #         elif label == 8 and number_of_ind_per_label[label] <= 300:
#     #             client3_indices.append(idx)
#     #             class_counts[2][label] += 1
#     #             number_of_ind_per_label[label] += 1
#     #         elif label == 5 and number_of_ind_per_label[label] <= 200:
#     #             client3_indices.append(idx)
#     #             class_counts[2][label] += 1
#     #             number_of_ind_per_label[label] += 1
#     #         elif label == 7 and number_of_ind_per_label[label] <= 200:
#     #             client3_indices.append(idx)
#     #             class_counts[2][label] += 1
#     #             number_of_ind_per_label[label] += 1
#
#     # for idx in remaining_data.indices:
#     #     _, label = dataset[idx]
#     #     if label in classes:
#     #         # print(label)
#     #         if label == 0 and number_of_ind_per_label[label] <= 100:
#     #             client2_indices.append(idx)
#     #             class_counts[1][label] += 1
#     #             number_of_ind_per_label[label] += 1
#     #         elif label == 6 and number_of_ind_per_label[label] <= 100:
#     #             client2_indices.append(idx)
#     #             class_counts[1][label] += 1
#     #             number_of_ind_per_label[label] += 1
#     #         elif label == 1 and number_of_ind_per_label[label] <= 100:
#     #             client2_indices.append(idx)
#     #             class_counts[1][label] += 1
#     #             number_of_ind_per_label[label] += 1
#     #         elif label == 3 and number_of_ind_per_label[label] <= 100:
#     #             client2_indices.append(idx)
#     #             class_counts[1][label] += 1
#     #             number_of_ind_per_label[label] += 1
#     #         elif label == 9 and number_of_ind_per_label[label] <= 100:
#     #             client3_indices.append(idx)
#     #             class_counts[2][label] += 1
#     #             number_of_ind_per_label[label] += 1
#     #         elif label == 5 and number_of_ind_per_label[label] <= 100:
#     #             client3_indices.append(idx)
#     #             class_counts[2][label] += 1
#     #             number_of_ind_per_label[label] += 1
#     #         elif label == 2 and number_of_ind_per_label[label] <= 100:
#     #             client3_indices.append(idx)
#     #             class_counts[2][label] += 1
#     #             number_of_ind_per_label[label] += 1
#     #         elif label == 7 and number_of_ind_per_label[label] <= 200:
#     #             client3_indices.append(idx)
#     #             class_counts[2][label] += 1
#     #             number_of_ind_per_label[label] += 1
#
#             # elif label == 6 and number_of_ind_per_label[6] <= 100:
#             #     client3_indices.append(idx)
#             #     class_counts[2][label] += 1
#             #     number_of_ind_per_label[label] += 1
#             #
#             #
#             # elif label == 1 and number_of_ind_per_label[1] <= 220:
#             #     client2_indices.append(idx)
#             #     class_counts[1][label] += 1
#             #     number_of_ind_per_label[1] += 1
#             # elif label == 3 and number_of_ind_per_label[3] <= 220:
#             #     client2_indices.append(idx)
#             #     class_counts[1][label] += 1
#             #     number_of_ind_per_label[3] += 1
#             # elif label == 9 and number_of_ind_per_label[9] <= 100:
#             #     client3_indices.append(idx)
#             #     class_counts[2][label] += 1
#             #     number_of_ind_per_label[9] += 1
#             # elif label == 5 and number_of_ind_per_label[5] <= 150:
#             #     client3_indices.append(idx)
#             #     class_counts[2][label] += 1
#             #     number_of_ind_per_label[5] += 1
#
#     # for idx in remaining_data.indices:
#     #     _, label = dataset[idx]
#     #     if label in classes:
#     #         # print(label)
#     #         if label % 2 == 0:
#     #             client2_indices.append(idx)
#     #         else:
#     #             client3_indices.append(idx)
#     #         class_counts[label] += 1
#
#     weights[0] = len(client1_indices)
#     weights[1] = len(client2_indices)
#     weights[2] = len(client3_indices)
#     client1_data = torch.utils.data.Subset(dataset, client1_indices)
#     client2_data = torch.utils.data.Subset(dataset, client2_indices)
#     client3_data = torch.utils.data.Subset(dataset, client3_indices)
#
#     # label_counts = {label: 0 for label in range(10)}
#     datasets = [client1_data, client2_data, client3_data]
#     for i, dataloader in enumerate(datasets):
#         # for data, label in dataloader:
#         #     label_counts[label] += 1
#         # unique_labels = list(label_counts.keys())
#         for label in classes:
#             percentage_client_indices_per_class[i][label] = class_counts[i][label] / classes_count[label]
#
#             logging.info(
#                 f"the classes for client {i}"
#                 f"and {label} the number is {class_counts[i][label]}")
#
#     log.info({f"client_{i}": len(ds.indices) for i, ds in enumerate(datasets)})
#
#     return datasets, percentage_client_indices_per_class, weights


def dirichlet_split(
        dataset, num_clients, seed, min_dataset_size, transformed_dataset=None, alpha=0.5
):
    np.random.seed(seed)
    freq = None
    min_size = 0
    indices = np.array(dataset.indices) if isinstance(dataset, Subset) else np.arange(len(dataset))
    targets = dataset.dataset.targets if isinstance(dataset, Subset) else dataset.targets
    targets = targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
    df = pd.DataFrame({"target": targets[indices]}, index=indices)
    label_to_indices = {}
    for label, group in df.groupby('target'):
        label_to_indices[label] = group.index
    labels, classes_count_ = np.unique(df.target, return_counts=True)
    classes_count = defaultdict(int)
    for label, count in zip(labels, classes_count_):
        classes_count[label] = count
    num_classes = len(df.target.unique())

    N = len(df.target)
    net_dataidx_map = {}
    percentage_client_indices_per_class = defaultdict(dict)

    while min_size < min_dataset_size:
        logging.info(f"min size is smaller than min_dataset_size")
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            # freq[k] = proportions
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            print(min_size)
    sum = 0
    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        # print(len(net_dataidx_map[j]))
        sum += len(net_dataidx_map[j])
    class_dis = np.zeros((num_clients, num_classes))

    for j in range(num_clients):
        for m in range(num_classes):
            class_dis[j, m] = int((np.array(targets[idx_batch[j]]) == m).sum())
            percentage_client_indices_per_class[j][m] = class_dis[j, m] / classes_count[m]

        logging.info(
            f"the classes for client {j} are {np.nonzero(class_dis[j, :])[0]}"
            f"and the number is {class_dis[j, np.nonzero(class_dis[j, :])[0]]}")

        # print(class_dis.astype(int))

    datasets = []
    weights = {}
    for client_idx in range(num_clients):
        indices = net_dataidx_map[client_idx]
        weights[client_idx] = len(indices)
        if isinstance(dataset, Subset):
            subset = copy.deepcopy(dataset)
            subset.indices = indices
            datasets.append(subset)
        else:
            subset = Subset(dataset=dataset, indices=indices)
            # subset.__class__.__getattr__ = new_getattr  # trick to get the attrs of the original dataset
            datasets.append(subset)
    #
    # logging.info(f"the percentages are {percentage_client_indices_per_class}")
    # logging.info(f"the sum is {sum}")
    log.info({f"client_{i}": len(ds.indices) for i, ds in enumerate(datasets)})

    return datasets, percentage_client_indices_per_class, weights


def distribute_cifar10_single_class(dataset, num_clients=10, seed=42, min_dataset_size=0, transformed_dataset=None,
                                    alpha=0.5,
                                    missing_classes=2
                                    ):
    assert num_clients == 5, "For CIFAR-10, the number of clients must be 10."

    np.random.seed(seed)
    min_size = 0
    indices = np.array(dataset.indices) if isinstance(dataset, Subset) else np.arange(len(dataset))
    targets = dataset.dataset.targets if isinstance(dataset, Subset) else dataset.targets
    targets = targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
    df = pd.DataFrame({"target": targets[indices]}, index=indices)
    label_to_indices = {}
    for label, group in df.groupby('target'):
        label_to_indices[label] = group.index
    labels, classes_count_ = np.unique(df.target, return_counts=True)
    classes_count = defaultdict(int)
    for label, count in zip(labels, classes_count_):
        classes_count[label] = count
    num_classes = len(df.target.unique())

    N = len(df.target)
    net_dataidx_map = {}
    percentage_client_indices_per_class = defaultdict(dict)

    # Initialize variables
    indices = np.arange(len(dataset))
    targets = np.array(dataset.targets)

    # Dictionary to store indices of each class
    class_to_indices = {i: [] for i in range(10)}

    # Populate the dictionary with indices for each class
    for idx, target in enumerate(targets):
        class_to_indices[target].append(idx)


    # Shuffle the indices within each class
    for class_idx in class_to_indices:
        np.random.shuffle(class_to_indices[class_idx])
        class_to_indices[class_idx] = class_to_indices[class_idx][:50]

    # Distribute the data to each client
    net_dataidx_map = {}
    classes_per_client=2
    for client_idx in range(num_clients):
        start_class = client_idx * classes_per_client
        end_class = start_class + classes_per_client
        client_indices = []
        for class_idx in range(start_class, end_class):
            client_indices.extend(class_to_indices[class_idx])
        net_dataidx_map[client_idx] = client_indices
        # net_dataidx_map[client_idx] = class_to_indices[client_idx]

    class_dis = np.zeros((num_clients, num_classes))
    for j in range(num_clients):
        for m in range(num_classes):
            class_dis[j, m] = int((np.array(targets[net_dataidx_map[j]]) == m).sum())
            percentage_client_indices_per_class[j][m] = class_dis[j, m] / classes_count[m]

    # Prepare the datasets for each client
    datasets = []
    weights = {}
    for client_idx in range(num_clients):
        indices = net_dataidx_map[client_idx]
        subset = Subset(dataset=dataset, indices=indices)
        datasets.append(subset)
        logging.info(f"Client {client_idx} has {len(indices)} samples of class {client_idx}")
        weights[client_idx] = len(indices)
    log.info({f"client_{i}": len(ds.indices) for i, ds in enumerate(datasets)})

    return datasets, percentage_client_indices_per_class, weights


# def dirichlet_split(
#         dataset, num_clients, seed, min_dataset_size, transformed_dataset=None, alpha=0.5
# ):
#     np.random.seed(seed)
#     freq = None
#     min_size = 0
#     indices = np.array(dataset.indices) if isinstance(dataset, Subset) else np.arange(len(dataset))
#     targets = dataset.dataset.targets if isinstance(dataset, Subset) else dataset.targets
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
#
#     N = len(df.target)
#     net_dataidx_map = {}
#     percentage_client_indices_per_class = defaultdict(dict)
#
#     while min_size < min_dataset_size:
#         logging.info(f"min size is smaller than min_dataset_size")
#         idx_batch = [[] for _ in range(num_clients)]
#         for k in range(num_classes):
#             idx_k = np.where(targets == k)[0]
#             np.random.shuffle(idx_k)
#
#             proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
#             proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
#             proportions = proportions / proportions.sum()
#             # freq[k] = proportions
#             proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
#             idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
#             min_size = min([len(idx_j) for idx_j in idx_batch])
#     sum = 0
#     for j in range(num_clients):
#         np.random.shuffle(idx_batch[j])
#         net_dataidx_map[j] = idx_batch[j]
#         # print(len(net_dataidx_map[j]))
#         sum += len(net_dataidx_map[j])
#     class_dis = np.zeros((num_clients, num_classes))
#
#     for j in range(num_clients):
#         for m in range(num_classes):
#             class_dis[j, m] = int((np.array(targets[idx_batch[j]]) == m).sum())
#             percentage_client_indices_per_class[j][m] = class_dis[j, m] / classes_count[m]
#
#         logging.info(
#             f"the classes for client {j} are {np.nonzero(class_dis[j, :])[0]}"
#             f"and the number is {class_dis[j, np.nonzero(class_dis[j, :])[0]]}")
#
#             # print(class_dis.astype(int))
#
#
#     datasets = []
#     for client_idx in range(num_clients):
#         indices = net_dataidx_map[client_idx]
#         if isinstance(dataset, Subset):
#             subset = copy.deepcopy(dataset)
#             subset.indices = indices
#             datasets.append(subset)
#         else:
#             subset = Subset(dataset=dataset, indices=indices)
#             # subset.__class__.__getattr__ = new_getattr  # trick to get the attrs of the original dataset
#             datasets.append(subset)
#     #
#     logging.info(f"the percentages are {percentage_client_indices_per_class}")
#     # logging.info(f"the sum is {sum}")
#     log.info({f"client_{i}": len(ds.indices) for i, ds in enumerate(datasets)})
#
#     return datasets, percentage_client_indices_per_class


def class_frequencies(dataset, train_distribution, seed, min_dataset_size):
    logging.info(f"{min_dataset_size}")
    np.random.seed(seed)
    logging.info(f" the train  distribution is {train_distribution}")
    indices = np.array(dataset.indices) if isinstance(dataset, Subset) else np.arange(len(dataset))
    targets = dataset.dataset.targets if isinstance(dataset, Subset) else dataset.targets
    targets = targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
    df = pd.DataFrame({"target": targets[indices]}, index=indices)

    label_to_indices = {}
    # Map indices to classes (labels, targets)
    for label, group in df.groupby('target'):
        label_to_indices[label] = group.index

    labels, classes_count_ = np.unique(df.target, return_counts=True)
    classes_count = defaultdict(int)
    for label, count in zip(labels, classes_count_):
        classes_count[label] = count
    logging.info(f"classes_count {classes_count}")

    # collect frequencies:
    all_frequencies = []
    for cls_idx in df.target.unique():
        freq = []
        for client_id in list(train_distribution.keys()):
            if train_distribution[client_id].get(cls_idx) is not None:
                freq.append(train_distribution[client_id][cls_idx])
            else:
                freq.append(0.)
            all_frequencies.append(freq)

        logging.info(f" For the class {cls_idx} length {len(freq)} and {sum(freq)}")

    return all_frequencies


# def plot_distibution(dataset, subset, s):
#     logging.info(f"s is {s}")
#     subset_labels = [dataset[i][1] for i in
#                      subset.indices]  # Assuming labels are at index 1 in your dataset tuples
#
#     logging.info(f"subset labels is {subset_labels} and length is {len(subset_labels)}")
#     # subset_labels = [label.item() for label in subset_labels]  # Convert labels to Python integers
#
#     num_classes = len(set(subset_labels))
#     plt.hist(subset_labels, bins=range(num_classes + 1), align='left',
#              rwidth=0.8)  # Adjust bins and appearance as needed
#     plt.xlabel('Label')
#     plt.ylabel('Frequency')
#     plt.title('Distribution of Labels in Subset')
#     plt.xticks(range(num_classes))  # Adjust if necessary
#     plt.savefig(f"distribution_{s}.pdf")


class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.original_labels = dataset.targets

    def __getitem__(self, idx):
        data_item, label = super().__getitem__(idx)
        original_label = self.original_labels[self.indices[idx]]
        return data_item, original_label


def customized_test_split(dataset, train_distribution, num_clients, seed, min_dataset_size, transformed_dataset=None):
    logging.info(f"{min_dataset_size}")
    np.random.seed(seed)
    # logging.info(f" the train  distribution is {train_distribution}")
    indices = np.array(dataset.indices) if isinstance(dataset, Subset) else np.arange(len(dataset))
    logging.info(f"the number of indices is {len(indices)}")
    targets = dataset.dataset.targets if isinstance(dataset, Subset) else dataset.targets
    targets = targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
    df = pd.DataFrame({"target": targets[indices]}, index=indices)

    label_to_indices = {}

    for idx, label in enumerate(targets):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    labels, classes_count_ = np.unique(df.target, return_counts=True)
    classes_count = defaultdict(int)
    for label, count in zip(labels, classes_count_):
        classes_count[label] = count
    logging.info(f"classes_count {classes_count}")
    client_indices = defaultdict(list)
    client_indices_per_class = defaultdict(dict)

    # collect frequencies:

    for cls_idx in df.target.unique():
        freq = []

        for client_id in list(train_distribution.keys()):
            if train_distribution[client_id].get(cls_idx) is not None:
                if sum(freq) + round(train_distribution[client_id][cls_idx] * classes_count[cls_idx]) <= \
                        classes_count[cls_idx]:
                    freq.append(round(train_distribution[client_id][cls_idx] * classes_count[cls_idx]))
                else:
                    freq.append(classes_count[cls_idx] - sum(freq))
            else:
                freq.append(0)

        logging.info(f"frequency is {freq[0]}")

        idx_k = copy.deepcopy(np.where(df['target'] == cls_idx)[0])

        np.random.shuffle(idx_k)
        remaining_indices = idx_k.copy()

        client_having_cls_idx = []

        for client_idx, client_cls_indicies in enumerate(freq):
            if freq[client_idx] != 0:
                selected_indices = remaining_indices[:client_cls_indicies]
                client_indices[client_idx].extend(selected_indices)
                client_having_cls_idx.append(client_idx)
                client_indices_per_class[client_idx][cls_idx] = selected_indices
                remaining_indices = remaining_indices[client_cls_indicies:]

        for index, client_idx in zip(remaining_indices, client_having_cls_idx):
            if index is None:
                logging.info(f"break done in index {index}")
                break
            client_indices[client_idx].append(index)

    logging.info(f"the length {len([idx for _, indices in client_indices.items() for idx in indices])}")
    # assert len(df) == len([idx for _, indices in client_indices.items() for idx in indices])
    #
    # # assert that there is no intersection between clients indices!
    # assert all((set(p0).isdisjoint(set(p1))) for p0, p1 in
    #            itertools.combinations([indices for _, indices in client_indices.items()], 2))
    datasets = []
    for client_idx in range(num_clients):
        # logging.info(
        #     f"customized test split:the classes for client {client_idx} are {list(client_indices_per_class[client_idx].keys())} "
        #     f"and the values are {list(client_indices_per_class[client_idx].values())}")

        indices = client_indices[client_idx]

        if isinstance(dataset, Subset):

            subset = copy.deepcopy(dataset)
            subset.indices = indices
            for index in range(len(subset)):
                subset_index = indices[index]  # Get the corresponding index from the indices array
                data_sample, label = dataset[subset_index]  # Extract data_sample and label from the dataset
                subset.dataset.targets[index] = label  # Update the label in the Subset
                logging.info(
                    f"the label is {label} and labels for that client are {list(client_indices_per_class[client_idx].keys())}")

            unique_labels = []
            for data_item, label in subset.dataset:
                if label not in unique_labels:
                    unique_labels.append(label)
            # logging.info(f"the labels are {unique_labels}")
            datasets.append(subset)

        else:
            subset = Subset(dataset=dataset, indices=indices)
            unique_labels = []
            for data_item, label in subset:
                if label not in unique_labels:
                    unique_labels.append(label)
            # logging.info(f"the labels are {unique_labels}")
            # subset.__class__.__getattr__ = new_getattr  # trick to get the attrs of the original dataset
            datasets.append(subset)

    return datasets


def ru_customized_test_split(dataset, train_distribution, num_clients, seed, min_dataset_size, transformed_dataset=None,
                             ru=0, missing_classes=0):
    np.random.seed(seed)
    indices = np.array(dataset.indices) if isinstance(dataset, Subset) else np.arange(len(dataset))
    targets = dataset.dataset.targets if isinstance(dataset, Subset) else dataset.targets
    targets = targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
    df = pd.DataFrame({"target": targets[indices]}, index=indices)

    label_to_indices = {}

    # Map indices to classes (labels, targets)
    for label, group in df.groupby('target'):
        label_to_indices[label] = group.index

    labels, classes_count_ = np.unique(df.target, return_counts=True)
    classes_count = defaultdict(int)

    for label, count in zip(labels, classes_count_):
        classes_count[label] = count

    client_indices = defaultdict(list)
    client_indices_per_class = defaultdict(dict)

    for cls_idx in df.target.unique():
        freq = []

        for client_id in list(train_distribution.keys()):
            if train_distribution[client_id].get(cls_idx) is not None:
                if sum(freq) + round(train_distribution[client_id][cls_idx] * classes_count[cls_idx]) <= \
                        classes_count[cls_idx]:
                    freq.append(round(train_distribution[client_id][cls_idx] * classes_count[cls_idx]))
                else:
                    freq.append(classes_count[cls_idx] - sum(freq))
            else:
                freq.append(0)

        idx_k = np.where(targets == cls_idx)[0]
        np.random.shuffle(idx_k)
        remaining_indices = idx_k.copy()

        all_indices = idx_k.copy()
        np.random.shuffle(all_indices)
        client_having_cls_idx = []

        for client_idx, client_cls_indices in enumerate(freq):
            if freq[client_idx] != 0:
                client_having_cls_idx.append(client_idx)
                selected_indices = remaining_indices[:client_cls_indices]
                client_indices[client_idx].extend(selected_indices)
                client_indices_per_class[client_idx][cls_idx] = selected_indices
                remaining_indices = remaining_indices[client_cls_indices:]

        current_min_size = min([len(client_indices[i]) for i in range(num_clients)])

        for index, client_idx in zip(remaining_indices, client_having_cls_idx):
            logging.info(f"client idx {client_idx}")

            if index is None:
                logging.info(f"break done in index {index}")
                break
            client_indices[client_idx].append(index)
        for index, client_idx in zip(remaining_indices, client_having_cls_idx):
            if index is None:
                break
            client_indices[client_idx].append(index)

    # Calculate the total number of RU samples
    total_ru_samples = int(ru * len(indices))

    # Distribute RU samples across clients
    rng = random.Random(seed)
    np.random.shuffle(indices)
    for client_idx in range(num_clients):
        client_indices[client_idx].extend(rng.sample(list(indices), total_ru_samples // num_clients))

    datasets = []
    for client_idx in range(num_clients):
        indices = client_indices[client_idx]
        if transformed_dataset:
            if isinstance(transformed_dataset, Subset):
                subset = copy.deepcopy(transformed_dataset)
                subset.indices = indices
                datasets.append(subset)
            else:
                subset = Subset(dataset=transformed_dataset, indices=indices)
                datasets.append(subset)
        else:
            if isinstance(dataset, Subset):
                subset = copy.deepcopy(dataset)
                subset.indices = indices
                datasets.append(subset)
            else:
                subset = Subset(dataset=dataset, indices=indices)
                datasets.append(subset)

    return datasets


def complex_dirichlet_client_data_split(
        dataset, num_clients, seed,
        min_dataset_size,
        missing_classes,
        alpha,
        dist_of_missing_classes="uniform",
        **kwargs
):
    """
    Data split with specific class distribution among clients
    args:
    dataset: pytorch dataset object, should have targets attribute that return list of that contain the labels in a list,
        this is very important since the split will depend on this list.
    num_clients: int
    seed: int for fixing the splits
    dist_of_missing_classes: how to sample the missing classes. Options are uniform, or
        weighted by the inverse of the frequency of classes
    Returns:
    List of Dataset subset object of length=num_clients
    """

    np.random.seed(seed)
    s = 0
    indices = np.array(dataset.indices) if isinstance(dataset, Subset) else np.arange(len(dataset))
    targets = dataset.dataset.targets if isinstance(dataset, Subset) else dataset.targets

    targets = targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets)

    df = pd.DataFrame({"target": targets[indices]}, index=indices)

    label_to_indices = {}
    # Map indices to classes (labels, targets)
    for label, group in df.groupby('target'):
        label_to_indices[label] = set(group.index)

    labels, classes_count = np.unique(df.target, return_counts=True)

    logging.info(f" class count is {classes_count}")
    num_classes = len(labels)
    N = len(df.target)
    # assert num_clients >= num_classes * 3, "To use the complex random split number clients should be equal or higher " \
    #                                        "than number of classes * 3 "

    classes = [i for i in range(num_classes)]
    classes_to_miss = [random.sample(classes, k=missing_classes) for _ in range(num_clients)]

    logging.info(f"{classes_to_miss} {len(classes_to_miss)}")

    flat_classes_to_miss = [cls_idx for clients in classes_to_miss for cls_idx in clients]
    labels, number_of_times_classes_is_missing = np.unique(flat_classes_to_miss, return_counts=True)
    # logging.info(f"flat_classes_to_miss {flat_classes_to_miss} and its length is {len(flat_classes_to_miss)}")
    logging.info(
        f"number_of_times_classes_is_missing {number_of_times_classes_is_missing} and the sum is {sum(number_of_times_classes_is_missing)}")

    client_indices = defaultdict(list)
    client_indices_per_class = defaultdict(dict)
    len_of_client_indices_per_class = defaultdict(dict)

    percentage_client_indices_per_class = defaultdict(dict)
    for cls_idx in df.target.unique():
        num_of_clients_missing_cls_idx = number_of_times_classes_is_missing[cls_idx]
        logging.info(f"num_of_clients_missing_cls_idx {num_of_clients_missing_cls_idx}")
        client_who_have_cls_idx = [client_idx for client_idx in range(num_clients) if
                                   cls_idx not in classes_to_miss[client_idx]]
        num_of_clients_for_cls_idx = len(client_who_have_cls_idx)

        assert num_of_clients_missing_cls_idx + len(client_who_have_cls_idx) == num_clients

        min_size = 0
        while min_size < min_dataset_size:
            idx_batch = [[] for _ in range(len(client_who_have_cls_idx))]
            idx_k = np.where(targets == cls_idx)[0]
            np.random.shuffle(idx_k)

            proportions = np.random.dirichlet(np.repeat(alpha, len(client_who_have_cls_idx)))
            proportions = np.array(
                [p * (len(idx_j) < N / len(client_who_have_cls_idx)) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

            min_size = min([len(idx_j) for idx_j in idx_batch])

            if min_size >= min_dataset_size:
                logging.info(f" the length of the proportions is {len(proportions)} and the sum is {sum(proportions)}")

                for x in idx_batch:
                    if x == []:
                        logging.info(f"x issss {x}")

        s = 0
        for client_idx, indices in zip(client_who_have_cls_idx, idx_batch):
            client_indices_per_class[client_idx][cls_idx] = list(indices)
            len_of_client_indices_per_class[client_idx][cls_idx] = len(list(indices))
            client_indices[client_idx].extend(list(indices))
            s += len(client_indices_per_class[client_idx][cls_idx])

        logging.info(f" the total number of datapoints used from cls {cls_idx} is {s}")
    datasets = []
    # assert that we distributed all the indices!
    assert len(df) == len([idx for _, indices in client_indices.items() for idx in indices])
    # assert that there is no intersection between clients indices!
    assert all((set(p0).isdisjoint(set(p1))) for p0, p1 in
               itertools.combinations([indices for _, indices in client_indices.items()], 2))

    indices_sum = 0
    weights = {}
    datapoints_per_client = {client_id: 0 for client_id in range(num_clients)}
    for client_idx in range(num_clients):
        logging.info(
            f"the classes for client {client_idx} are {list(len_of_client_indices_per_class[client_idx].keys())} "
            f"and the values are {list(len_of_client_indices_per_class[client_idx].values())}")
        datapoints_per_client[client_idx] = sum(list(len_of_client_indices_per_class[client_idx].values()))
        indices = client_indices[client_idx]
        indices_sum += len(indices)
        weights[client_idx] = len(indices)
        for cls_idx in list(client_indices_per_class[client_idx].keys()):
            percentage_client_indices_per_class[client_idx][cls_idx] = len(
                client_indices_per_class[client_idx][cls_idx]) / \
                                                                       classes_count[
                                                                           cls_idx]

        if isinstance(dataset, Subset):
            subset = copy.deepcopy(dataset)
            subset.indices = indices
            datasets.append(subset)
        else:
            subset = Subset(dataset=dataset, indices=indices)
            # subset.__class__.__getattr__ = new_getattr  # trick to get the attrs of the original dataset
            datasets.append(subset)
    log.info(f"total number of datapoints used is {indices_sum}")

    log.info({f"client_{i}": len(ds.indices) for i, ds in enumerate(datasets)})
    time.sleep(5)
    return datasets, percentage_client_indices_per_class, weights


def disjoint_uniform_split(indices, num_clients, seed):
    np.random.seed(seed)
    random.shuffle(indices)
    indices_per_client = len(indices) // num_clients
    remainder = len(indices) % num_clients  # Calculate the remainder

    client_indices = []
    start = 0

    for i in range(num_clients):
        extra = 1 if remainder > 0 else 0  # Distribute the remainder
        end = start + indices_per_client + extra
        if i == num_clients - 1:  # If it's the last client, add one more index
            end += 1
        client_indices.append(indices[start:end])
        start = end
        remainder -= 1

    return client_indices


# def complex_uniform_client_data_split(
#         dataset, num_clients, seed,
#         min_dataset_size,
#         missing_classes,
#         dist_of_missing_classes="uniform",
#         **kwargs
# ):
#     """
#     Data split with specific class distribution among clients
#     args:
#     dataset: pytorch dataset object, should have targets attribute that return list of that contain the labels in a list,
#         this is very important since the split will depend on this list.
#     num_clients: int
#     seed: int for fixing the splits
#     dist_of_missing_classes: how to sample the missing classes. Options are uniform, or
#         weighted by the inverse of the frequency of classes
#     Returns:
#     List of Dataset subset object of length=num_clients
#     """
#
#     np.random.seed(seed)
#     s = 0
#     indices = np.array(dataset.indices) if isinstance(dataset, Subset) else np.arange(len(dataset))
#     targets = dataset.dataset.targets if isinstance(dataset, Subset) else dataset.targets
#
#     targets = targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
#
#     df = pd.DataFrame({"target": targets[indices]}, index=indices)
#
#     label_to_indices = {}
#     # Map indices to classes (labels, targets)
#     for label, group in df.groupby('target'):
#         label_to_indices[label] = set(group.index)
#
#     labels, classes_count = np.unique(df.target, return_counts=True)
#
#     num_classes = len(labels)
#     N = len(df.target)
#     assert num_clients >= num_classes * 3, "To use the complex random split number clients should be equal or higher " \
#                                            "than number of classes * 3 "
#
#     classes = [i for i in range(num_classes)]
#     classes_to_miss = [random.sample(classes, k=missing_classes) for _ in range(num_clients)]
#
#     logging.info(f"{classes_to_miss} {len(classes_to_miss)}")
#
#     flat_classes_to_miss = [cls_idx for clients in classes_to_miss for cls_idx in clients]
#     labels, number_of_times_classes_is_missing = np.unique(flat_classes_to_miss, return_counts=True)
#     logging.info(f"flat_classes_to_miss {flat_classes_to_miss} and its length is {len(flat_classes_to_miss)}")
#     logging.info(
#         f"number_of_times_classes_is_missing {number_of_times_classes_is_missing} and the sum is {sum(number_of_times_classes_is_missing)}")
#
#     percentage_client_indices_per_class = defaultdict(dict)
#     client_indices = defaultdict(list)
#     client_indices_per_class = defaultdict(dict)
#     percentage_client_indices_per_class = defaultdict(dict)
#     len_of_client_indices_per_class = defaultdict(dict)
#
#     for cls_idx in df.target.unique():
#         num_of_clients_missing_cls_idx = number_of_times_classes_is_missing[cls_idx]
#         logging.info(f"num_of_clients_missing_cls_idx {num_of_clients_missing_cls_idx}")
#         client_who_have_cls_idx = [client_idx for client_idx in range(num_clients) if
#                                    cls_idx not in classes_to_miss[client_idx]]
#         num_of_clients_for_cls_idx = len(client_who_have_cls_idx)
#
#         assert num_of_clients_missing_cls_idx + len(client_who_have_cls_idx) == num_clients
#
#         min_size = 0
#         # while min_size < min_dataset_size:
#         logging.info(f"min_size for cls {cls_idx}")
#         idx_k = np.where(targets == cls_idx)[0]
#         idx_batch = disjoint_uniform_split(idx_k, len(client_who_have_cls_idx), seed)
#         min_size = min([len(idx_j) for idx_j in idx_batch])
#
#         s = 0
#         for client_idx, indices in zip(client_who_have_cls_idx, idx_batch):
#             client_indices_per_class[client_idx][cls_idx] = list(indices)
#             len_of_client_indices_per_class[client_idx][cls_idx] = len(list(indices))
#
#             client_indices[client_idx].extend(list(indices))
#             s += len(client_indices_per_class[client_idx][cls_idx])
#
#         logging.info(f" the total number of datapoints used from cls {cls_idx} is {s}")
#     for client_idx in range(num_clients):
#         logging.info(
#             f"the classes for client {client_idx} are {list(len_of_client_indices_per_class[client_idx].keys())} "
#             f"and the values are {list(len_of_client_indices_per_class[client_idx].values())}")
#         # logging.info(f"classes for clients are {list(client_indices_per_class[client_idx].keys())}")
#
#     datasets = []
#     # assert that we distributed all the indices!
#     assert len(df) == len([idx for _, indices in client_indices.items() for idx in indices])
#     # assert that there is no intersection between clients indices!
#     assert all((set(p0).isdisjoint(set(p1))) for p0, p1 in
#                itertools.combinations([indices for _, indices in client_indices.items()], 2))
#
#     indices_sum = 0
#     for client_idx in range(num_clients):
#         indices = client_indices[client_idx]
#         indices_sum += len(indices)
#
#         for cls_idx in list(client_indices_per_class[client_idx].keys()):
#             percentage_client_indices_per_class[client_idx][cls_idx] = len(
#                 client_indices_per_class[client_idx][cls_idx]) / \
#                                                                        classes_count[
#                                                                            cls_idx]
#
#         if isinstance(dataset, Subset):
#             logging.info(f"is istance of subset")
#             subset = copy.deepcopy(dataset)
#             subset.indices = indices
#             datasets.append(subset)
#
#             # unique_labels = []
#             #
#             # # Iterate through the subset to collect unique labels
#             # for data_item, label in subset:
#             #     if label not in unique_labels:
#             #         unique_labels.append(label)
#             # logging.info(f"the labels are {unique_labels}")
#         else:
#             logging.info(f" the indices are {indices} with length {len(indices)}")
#             subset = Subset(dataset=dataset, indices=indices)
#
#             datasets.append(subset)
#
#             unique_labels = []
#             # Iterate through the subset to collect unique labels
#             for data_item, label in subset:
#                 if label not in unique_labels:
#                     unique_labels.append(label)
#             logging.info(f"the labels are {unique_labels}")
#
#     log.info(f"total number of datapoints used is {indices_sum}")
#
#     log.info({f"client_{i}": len(ds.indices) for i, ds in enumerate(datasets)})
#     time.sleep(5)
#     return datasets, percentage_client_indices_per_class

def complex_uniform_client_data_split(
        dataset, num_clients, seed,
        min_dataset_size,
        missing_classes,
        dist_of_missing_classes="uniform",
        **kwargs
):
    """
    Data split with specific class distribution among clients
    args:
    dataset: pytorch dataset object, should have targets attribute that return list of that contain the labels in a list,
        this is very important since the split will depend on this list.
    num_clients: int
    seed: int for fixing the splits
    dist_of_missing_classes: how to sample the missing classes. Options are uniform, or
        weighted by the inverse of the frequency of classes
    Returns:
    List of Dataset subset object of length=num_clients
    """

    np.random.seed(seed)
    s = 0
    indices = np.array(dataset.indices) if isinstance(dataset, Subset) else np.arange(len(dataset))
    targets = dataset.dataset.targets if isinstance(dataset, Subset) else dataset.targets

    targets = targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets)

    df = pd.DataFrame({"target": targets[indices]}, index=indices)

    label_to_indices = {}
    # Map indices to classes (labels, targets)
    for label, group in df.groupby('target'):
        label_to_indices[label] = set(group.index)

    labels, classes_count = np.unique(df.target, return_counts=True)

    num_classes = len(labels)
    N = len(df.target)
    assert num_clients >= num_classes * 3, "To use the complex random split number clients should be equal or higher " \
                                           "than number of classes * 3 "

    classes = [i for i in range(num_classes)]
    classes_to_miss = [random.sample(classes, k=missing_classes) for _ in range(num_clients)]

    logging.info(f"{classes_to_miss} {len(classes_to_miss)}")

    flat_classes_to_miss = [cls_idx for clients in classes_to_miss for cls_idx in clients]
    labels, number_of_times_classes_is_missing = np.unique(flat_classes_to_miss, return_counts=True)
    logging.info(f"flat_classes_to_miss {flat_classes_to_miss} and its length is {len(flat_classes_to_miss)}")
    logging.info(
        f"number_of_times_classes_is_missing {number_of_times_classes_is_missing} and the sum is {sum(number_of_times_classes_is_missing)}")

    percentage_client_indices_per_class = defaultdict(dict)
    client_indices = defaultdict(list)
    client_indices_per_class = defaultdict(dict)
    percentage_client_indices_per_class = defaultdict(dict)
    len_of_client_indices_per_class = defaultdict(dict)

    for cls_idx in df.target.unique():
        num_of_clients_missing_cls_idx = number_of_times_classes_is_missing[cls_idx]
        logging.info(f"num_of_clients_missing_cls_idx {num_of_clients_missing_cls_idx}")
        client_who_have_cls_idx = [client_idx for client_idx in range(num_clients) if
                                   cls_idx not in classes_to_miss[client_idx]]
        num_of_clients_for_cls_idx = len(client_who_have_cls_idx)

        assert num_of_clients_missing_cls_idx + len(client_who_have_cls_idx) == num_clients

        min_size = 0
        # while min_size < min_dataset_size:
        logging.info(f"min_size for cls {cls_idx}")
        idx_k = np.where(targets == cls_idx)[0]
        idx_batch = disjoint_uniform_split(idx_k, len(client_who_have_cls_idx), seed)
        min_size = min([len(idx_j) for idx_j in idx_batch])

        s = 0
        for client_idx, indices in zip(client_who_have_cls_idx, idx_batch):
            client_indices_per_class[client_idx][cls_idx] = list(indices)
            len_of_client_indices_per_class[client_idx][cls_idx] = len(list(indices))

            client_indices[client_idx].extend(list(indices))
            s += len(client_indices_per_class[client_idx][cls_idx])

        logging.info(f" the total number of datapoints used from cls {cls_idx} is {s}")
    for client_idx in range(num_clients):
        logging.info(
            f"the classes for client {client_idx} are {list(len_of_client_indices_per_class[client_idx].keys())} "
            f"and the values are {list(len_of_client_indices_per_class[client_idx].values())}")
        # logging.info(f"classes for clients are {list(client_indices_per_class[client_idx].keys())}")

    datasets = []
    # assert that we distributed all the indices!
    assert len(df) == len([idx for _, indices in client_indices.items() for idx in indices])
    # assert that there is no intersection between clients indices!
    assert all((set(p0).isdisjoint(set(p1))) for p0, p1 in
               itertools.combinations([indices for _, indices in client_indices.items()], 2))

    indices_sum = 0
    for client_idx in range(num_clients):
        indices = client_indices[client_idx]
        indices_sum += len(indices)

        for cls_idx in list(client_indices_per_class[client_idx].keys()):
            percentage_client_indices_per_class[client_idx][cls_idx] = len(
                client_indices_per_class[client_idx][cls_idx]) / \
                                                                       classes_count[
                                                                           cls_idx]

        if isinstance(dataset, Subset):
            subset = copy.deepcopy(dataset)
            subset.indices = indices
            datasets.append(subset)
        else:
            subset = Subset(dataset=dataset, indices=indices)
            # subset.__class__.__getattr__ = new_getattr  # trick to get the attrs of the original dataset
            unique_labels = []
            # Iterate through the subset to collect unique labels
            for data_item, label in subset:
                if label not in unique_labels:
                    unique_labels.append(label)
            logging.info(f"the labels are {unique_labels}")
            datasets.append(subset)
    log.info(f"total number of datapoints used is {indices_sum}")

    log.info({f"client_{i}": len(ds.indices) for i, ds in enumerate(datasets)})
    time.sleep(5)
    return datasets, percentage_client_indices_per_class


def sanity_check(dataset, num_clients, seed, missing_classes, num_classes, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # np.random.seed(seed)
    num_users = num_clients
    shard_per_user = missing_classes
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    count = 0

    ## this part search for the indices for each label
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1

    shard_per_class = int(shard_per_user * num_users / num_classes)
    logging.info(f"shard per class is {shard_per_class}")
    samples_per_user = int(count / num_users)
    logging.info(f"number of samples per user is {samples_per_user}")

    # whether to sample more test samples per user
    if (samples_per_user < 100):
        double = True
    else:
        double = False

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x
        logging.info(f"idxs_dict {len(idxs_dict[label])}")

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        logging.info(f"rand set all {rand_set_all}")
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        if double:
            rand_set_label = list(rand_set_all[i]) * 50
        else:
            rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)

            if (samples_per_user < 100 and False):
                rand_set.append(idxs_dict[label][idx])

            else:
                rand_set.append(idxs_dict[label].pop(idx))
                # logging.info(f"len of random set is {len(rand_set)} for idx{idx}")

        dict_users[i] = np.concatenate(rand_set)
        logging.info(f"number of indices for client {i} are {len(dict_users[i])}")

    assert len(dataset) == len([idx for _, indices in dict_users.items() for idx in indices])
    # assert that there is no intersection between clients indices!
    assert all((set(p0).isdisjoint(set(p1))) for p0, p1 in
               itertools.combinations([indices for _, indices in dict_users.items()], 2))
    datasets = []
    for client_idx in range(num_users):

        if isinstance(dataset, Subset):
            subset = copy.deepcopy(dataset)
            subset.indices = dict_users[client_idx]
            datasets.append(subset)
        else:
            subset = Subset(dataset=dataset, indices=dict_users[client_idx])
            # subset.__class__.__getattr__ = new_getattr  # trick to get the attrs of the original dataset
            datasets.append(subset)

    return datasets, None


def add_attrs(*given_subsets: [Subset]):
    for subsets in given_subsets:
        for subset in subsets:
            subset.__class__.__getattr__ = new_getattr
