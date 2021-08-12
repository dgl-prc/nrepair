import torch.nn as nn
from collections import OrderedDict


class MnistNet(nn.Module):

    def __init__(self, num_hidden_layers, hidden_size):
        super(MnistNet, self).__init__()
        output_class = 10
        input_dim = 784
        layers = [('h0', nn.Linear(input_dim, hidden_size)), ('relu0', nn.ReLU())]
        for i in range(num_hidden_layers - 1):
            layers.append((f'h{i + 1}', nn.Linear(hidden_size, hidden_size)))
            layers.append((f'relu{i + 1}', nn.ReLU()))
        layers.append((f'output', nn.Linear(hidden_size, output_class)))
        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.model(x)


class Cifar10Net(nn.Module):

    def __init__(self, num_hidden_layers, hidden_size):
        super(Cifar10Net, self).__init__()
        output_class = 10
        input_dim = 3072
        layers = [('h0', nn.Linear(input_dim, hidden_size)), ('relu0', nn.ReLU())]
        for i in range(num_hidden_layers - 1):
            layers.append((f'h{i + 1}', nn.Linear(hidden_size, hidden_size)))
            layers.append((f'relu{i + 1}', nn.ReLU()))
        layers.append((f'output', nn.Linear(hidden_size, output_class)))
        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.model(x)


class ACASXu(nn.Module):
    def __init__(self):
        super(ACASXu, self).__init__()

        self.linear1 = nn.Linear(5, 50)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(50, 50)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(50, 50)
        self.relu3 = nn.ReLU()

        self.linear4 = nn.Linear(50, 50)
        self.relu4 = nn.ReLU()

        self.linear5 = nn.Linear(50, 50)
        self.relu5 = nn.ReLU()

        self.linear6 = nn.Linear(50, 50)
        self.relu6 = nn.ReLU()

        self.linear7 = nn.Linear(50, 5)

    def forward(self, input):
        x = self.linear1(input)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.relu5(x)
        x = self.linear6(x)
        x = self.relu6(x)
        x = self.linear7(x)
        return x
