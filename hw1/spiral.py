# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from math import sqrt, atan2

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.layer1 = torch.nn.Linear(2, 16)
        self.layer2 = torch.nn.Linear(16, 1)

    def forward(self, input1):
        list_1 = []
        for input in input1:
            r = sqrt(input[0] * input[0] + input[1]*input[1])
            a = atan2(input[1], input[0])
            output = [r, a]
            list_1.append(output)
        output = torch.tensor(list_1)
        layer1 = torch.tanh(self.layer1(output))
        layer2 = self.layer2(layer1)
        return torch.sigmoid(layer2)

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.layer1 = torch.nn.Linear(2, num_hid)
        self.layer2 = torch.nn.Linear(num_hid, num_hid)
        self.layer3 = torch.nn.Linear(num_hid, 1)
        # INSERT CODE HERE

    def forward(self, input):
        layer1 = torch.tanh(self.layer1(input))
        layer2 = torch.tanh(self.layer2(layer1))
        layer3 = self.layer3(layer2)
        return torch.sigmoid(layer3)

def graph_hidden(net, layer, node):
    plt.clf()
    # INSERT CODE HERE
