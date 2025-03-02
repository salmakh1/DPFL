import logging

import numpy as np
import torch
from torch.utils.data import Dataset


class dataloader(Dataset):

    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx]
        image = np.reshape(image, (28, 28)).astype(np.float32)
        image = torch.tensor(np.expand_dims(image, axis=0))
        label = torch.tensor(self.y[idx])
        return image, label
