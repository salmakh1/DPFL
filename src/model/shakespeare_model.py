import logging

import torch.nn as nn


# class RNNShakespeare(nn.Module):
#     def __init__(self, num_classes, n_hidden):
#         super(RNNShakespeare, self).__init__()
#         self.num_classes = num_classes
#         self.n_hidden = n_hidden
#         self.embedding = nn.Embedding(self.num_classes, 8)
#         self.lstm = nn.LSTM(8, self.n_hidden, batch_first=True, num_layers=2)
#         self.fc = nn.Linear(self.n_hidden, self.num_classes)
#         logging.info(f"model instantiation")
#
#     def forward(self, x):
#         x = self.embedding(x)
#         self.lstm.flatten_parameters()  # Add this line
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])
#         return out
#
#
# class RNNShakespeare(nn.Module):
#
#     def __init__(self, input_size=8, hidden_size=256, **kwargs):
#         super().__init__()
#         self.embedding = nn.Embedding(num_embeddings=self.num_classes,
#                                       embedding_dim=input_size, padding_idx=0)
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
#                             batch_first=True, bidirectional=False, **kwargs)
#         self.linear = nn.Linear(hidden_size, self.num_classes)
#
#     def forward(self, X, lengths):
#         X = self.embedding(X)
#         X = rnn.pack_padded_sequence(X, lengths, batch_first=True)
#         self.lstm.flatten_parameters()
#         X, _ = self.lstm(X)
#         X, _ = rnn.pad_packed_sequence(X, batch_first=True)
#         return self.linear(X[:, -1])


# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils import rnn
#
# class RNNBase(nn.Module):
#
#     def __init__(self, input_size=8, hidden_size=256,num_classes=80, **kwargs):
#         super().__init__()
#         self.embedding = nn.Embedding(num_embeddings=num_classes,
#                                       embedding_dim=input_size, padding_idx=0)
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
#                             batch_first=True, bidirectional=False, **kwargs)
#
#     def forward(self, X):
#         lengths = X.shape[1] - (X == 0).sum(1)
#         sorted_lengths, sorted_indices = lengths.sort(0, descending=True)
#         X = X[sorted_indices]
#         X = self.embedding(X)
#         X = rnn.pack_padded_sequence(X, sorted_lengths.cpu(), batch_first=True)
#         self.lstm.flatten_parameters()
#         X, _ = self.lstm(X)
#         X, _ = rnn.pad_packed_sequence(X, batch_first=True)
#         return X[:, -1]
#
# class RNNClassifier(nn.Module):
#
#     def __init__(self, input_size, num_classes):
#         super().__init__()
#         self.linear = nn.Linear(input_size, num_classes)
#
#     def forward(self, x):
#         return self.linear(x)
#
# class RNNShakespeare(nn.Module):
#
#     def __init__(self, num_classes, n_hidden,  **kwargs):
#     # def __init__(self, input_size=8, hidden_size=256, num_classes=10, **kwargs):
#         super().__init__()
#         self.base = RNNBase(input_size=8, hidden_size=n_hidden,num_classes=num_classes, **kwargs)
#         self.classifier = RNNClassifier(input_size=n_hidden, num_classes=num_classes)
#
#     def forward(self, X):
#         x_base = self.base(X)
#         c = self.classifier(x_base)
#         return c



class RNNBase(nn.Module):
    def __init__(self, num_classes, n_hidden):
        super(RNNBase, self).__init__()
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        self.embedding = nn.Embedding(self.num_classes, 8)
        self.lstm = nn.LSTM(8, self.n_hidden, batch_first=True, num_layers=2)

    def forward(self, x):
        x = self.embedding(x)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        return out[:, -1, :]

class RNNClassifier(nn.Module):
    def __init__(self, hidden_dims, num_classes):
        super(RNNClassifier, self).__init__()
        self.fc = nn.Linear(hidden_dims, num_classes)

    def forward(self, x):
        return self.fc(x)

class RNNShakespeare(nn.Module):
    def __init__(self, num_classes, n_hidden):
        super(RNNShakespeare, self).__init__()
        self.base = RNNBase(num_classes, n_hidden)
        self.classifier = RNNClassifier(n_hidden, num_classes)

    def forward(self, x):
        base_output = self.base(x)
        return self.classifier(base_output)

