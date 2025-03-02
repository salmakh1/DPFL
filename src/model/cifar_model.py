from torch import nn
import torch.nn.functional as F
from torch.quantization import quantize, QuantStub, DeQuantStub


# class CNNCifar(nn.Module):
#     def __init__(self, num_classes):
#         super(CNNCifar, self).__init__()
#         self.num_classes = num_classes
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, self.num_classes)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)


class CNNCifar(nn.Module):
    def __init__(self, num_classes):
        super(CNNCifar, self).__init__()
        self.base = FE(16 * 5 * 5, [120, 84])
        self.classifier = Classifier(84, num_classes)

    def forward(self, x):
        return self.classifier((self.base(x)))


class FE(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(FE, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        # self.quant = QuantStub()
        # self.dequant = DeQuantStub()

    def forward(self, x):
        # x = self.quant(x)
        x = self.pool(self.relu(self.conv1(x)))
        print("here ", x.shape)
        x = self.pool(self.relu(self.conv2(x)))
        print("here 1", x.shape)
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.dequant(x)
        return x


class Classifier(nn.Module):
    def __init__(self, hidden_dims, output_dim=10):
        super(Classifier, self).__init__()
        self.fc3 = nn.Linear(hidden_dims, output_dim)

    def forward(self, x):
        x = self.fc3(x)
        return x

# import logging

# import torch.nn as nn
# import torch.nn.functional as F
#
# class CNNCifar(nn.Module):
#     def __init__(self, num_classes):
#         super(CNNCifar, self).__init__()
#         self.num_classes = num_classes
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.bn1 = nn.BatchNorm2d(6)  # BatchNorm after the first convolutional layer
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.bn2 = nn.BatchNorm2d(16)  # BatchNorm after the second convolutional layer
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.bn3 = nn.BatchNorm1d(120)  # BatchNorm after the first fully connected layer
#         self.fc2 = nn.Linear(120, 84)
#         self.bn4 = nn.BatchNorm1d(84)  # BatchNorm after the second fully connected layer
#         self.fc3 = nn.Linear(84, self.num_classes)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))  # BatchNorm after the first convolutional layer
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))  # BatchNorm after the second convolutional layer
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.bn3(self.fc1(x)))  # BatchNorm after the first fully connected layer
#         x = F.relu(self.bn4(self.fc2(x)))  # BatchNorm after the second fully connected layer
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)
