from torch import nn
import torch.nn.functional as F

class LinearVehicle(nn.Module):
    def __init__(self, zero_init=True, input_size=101):
        super(LinearVehicle, self).__init__()
        self.zero_init = zero_init
        self.linear = nn.Linear(input_size, 1)
        if zero_init:
            self.linear.bias.data.fill_(0)
        # self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        if self.zero_init:
            x = F.linear(x, self.linear.weight)
        else:
            x = self.linear(x)
        return x
