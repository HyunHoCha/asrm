import torch
import torch.nn as nn


class MLP_net_noNorm_double(nn.Module):
    # two hidden layers

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, C):
        super(MLP_net_noNorm_double, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.C = C
        self.num_distances = int(C * (C - 1) / 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
