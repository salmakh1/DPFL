import torch

class dataloader(torch.utils.data.Dataset):
    def __init__(self, dataset, worker_id):
        self.dataset = dataset
        self.id = worker_id

    def __getitem__(self, index):
        data = self.dataset['x'][index]
        target = self.dataset['y'][index]
        return data, target

    def __len__(self):
        return len(self.dataset['x'])

