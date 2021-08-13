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
        self.output = layer2
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
        self.output = layer3
        return torch.sigmoid(layer3)

def graph_hidden(net, layer, node):

    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout
        output = net(grid)
        net.train() # toggle batch norm, dropout back again
        pred = (output >= 0.5).float()
        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')

