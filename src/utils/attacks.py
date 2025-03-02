import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# class FlippedLabelsDataset(Dataset):
#     def __init__(self, original_dataset, num_classes):
#         self.original_dataset = original_dataset
#         # self.malicious_indices = malicious_indices
#         self.num_classes = num_classes
#
#     def __len__(self):
#         return len(self.original_dataset)
#
#     def __getitem__(self, index):
#         data, label = self.original_dataset[index]
#         label = np.random.randint(0, self.num_classes)
#         return data, label
#
# def label_flip_attack(num_classes, train_loader):
#     for batch_idx, (data, labels) in enumerate(train_loader):
#         labels_flip = np.random.randint(0, num_classes, len(labels))
#         # logging.info(f"initial labels are {labels} and flipped labels are {labels_flip}")
#         start_idx = batch_idx * train_loader.batch_size
#         end_idx = start_idx + len(labels)
#         logging.info(f" start index is {start_idx} and end_idx is {end_idx}")
#         train_loader.dataset.targets[start_idx:end_idx] = labels_flip
#     return train_loader


class FlippedLabelsDataset(Dataset):
    def __init__(self, original_dataset, num_classes):
        self.original_dataset = original_dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        data, label = self.original_dataset[index]
        flipped_label = self.num_classes - 1 - label
        # logging.info(f"initial labels are {label} and flipped labels are {flipped_label}")
        return data, flipped_label


def label_flip_attack(num_classes, train_loader):
    for batch_idx, (data, labels) in enumerate(train_loader):
        labels_flip = num_classes - 1 - labels
        start_idx = batch_idx * train_loader.batch_size
        end_idx = start_idx + len(labels)
        train_loader.dataset.targets[start_idx:end_idx] = labels_flip
    return train_loader


def gradient_boost_attack(client, boost_factor):
    for param_name, param_tensor in client.model.state_dict().items():
        if param_tensor.ndimension() > 1:  # Only scale weights, not biases
            scaled_weights = boost_factor * param_tensor
            client.model.state_dict()[param_name].copy_(scaled_weights)
    client.model.load_state_dict(client.model.state_dict())


# def random_update_attack(client):
#     for param_name, param_tensor in client.model.named_parameters():
#         if param_tensor.ndimension() > 1:
#             random_weights = torch.randn_like(param_tensor) * 1.0
#             param_tensor.data.copy_(random_weights)
#
#
# def same_model_attack(client):
#     for param_name, param_tensor in client.model.state_dict().items():
#         if param_tensor.ndimension() > 1:
#             random_weights = torch.ones_like(param_tensor)
#             client.model.state_dict()[param_name].copy_(random_weights)
#     client.model.load_state_dict(client.model.state_dict())
#
#
# def sign_flip_model_attack(client):
#     for param_name, param_tensor in client.model.state_dict().items():
#         if param_tensor.ndimension() > 1:
#             random_weights = -param_tensor
#             client.model.state_dict()[param_name].copy_(random_weights)
#     client.model.load_state_dict(client.model.state_dict())


def random_update_attack(client):
    for param_name, param_tensor in client.model.named_parameters():
        if param_tensor.ndimension() > 1:
            random_weights = torch.randn_like(param_tensor) * 1.0
            new_param_tensor = random_weights.detach().clone()
            client.model.state_dict()[param_name].copy_(new_param_tensor)


def same_model_attack(client):
    for param_name, param_tensor in client.model.state_dict().items():
        if param_tensor.ndimension() > 1:
            random_weights = torch.ones_like(param_tensor)
            new_param_tensor = random_weights.detach().clone()
            client.model.state_dict()[param_name].copy_(new_param_tensor)


def sign_flip_model_attack(client):
    for param_name, param_tensor in client.model.state_dict().items():
        if param_tensor.ndimension() > 1:
            random_weights = -param_tensor
            new_param_tensor = random_weights.detach().clone()
            client.model.state_dict()[param_name].copy_(new_param_tensor)
