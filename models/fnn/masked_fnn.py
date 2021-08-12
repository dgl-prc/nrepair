import torch.nn as nn
from collections import OrderedDict
from utils.constant import DataType


class MaskedNet(nn.Module):
    def __init__(self, num_hidden_layers=7, hidden_size=50, datatype=DataType.ACASXU):
        super(MaskedNet, self).__init__()
        if datatype == DataType.MNIST:
            input_dim = 784
            output_class = 10
        elif datatype == DataType.CIFAR10:
            input_dim = 3072
            output_class = 10
        elif datatype == DataType.ACASXU:
            input_dim = 5
            output_class = 5
        layers = [('h0', nn.Linear(input_dim, hidden_size)), ('relu0', nn.ReLU()),
                  ("MSK0", nn.Linear(hidden_size,hidden_size))]
        for i in range(num_hidden_layers - 1):
            layers.append((f'h{i + 1}', nn.Linear(hidden_size, hidden_size)))
            layers.append((f'relu{i + 1}', nn.ReLU()))
            layers.append((f"MSK{i + 1}", nn.Linear(hidden_size, hidden_size)))
        layers.append((f'output', nn.Linear(hidden_size, output_class)))
        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.model(x)
