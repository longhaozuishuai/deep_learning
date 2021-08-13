# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        self.layer1 = nn.Linear(784, 100)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = torch.relu(self.layer1(x))
        return F.log_softmax(output,dim=1)         


class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.layer1= torch.nn.Linear(784, 300)
        self.layer2 = torch.nn.Linear(300, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        layer1 = self.layer1(x)
        hidden = torch.tanh(layer1)
        output = self.layer2(hidden)
        return   F.log_softmax(output, dim=1)

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.conv1 = torch.nn.Conv2d(1, 40, (5, 5))
        self.conv2 = torch.nn.Conv2d(40, 60, (5, 5))
        self.conv2_drop = nn.Dropout2d()
        self.layer1 = torch.nn.Linear(960, 300)
        self.layer2 = torch.nn.Linear(300, 10)

    def forward(self, x):
        x = torch.relu(F.max_pool2d(self.conv1(x), 2, 2))
        x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2, 2))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        x = F.dropout(x, training=self.training)
        x = self.layer2(x)
        return F.log_softmax(x, dim=1) # CHANGE CODE HERE
