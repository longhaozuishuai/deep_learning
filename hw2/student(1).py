#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import torch.nn.functional as F
# import numpy as np
# import sklearn

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """

    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

stopWords = {}
wordVectors = GloVe(name='6B', dim=50)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """

    
    output1 = torch.argmax(ratingOutput, dim=1)
    output2 = torch.argmax(categoryOutput, dim=1)
    return output1, output2

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        self.lstm = tnn.LSTM(50, 100, num_layers=2, batch_first=True, dropout=0.5)
        self.Linear1 = tnn.Linear(100, 70)
        self.Relu = tnn.ReLU()
        self.Linear_2 = tnn.Linear(70, 2)
        self.Linear_5 = tnn.Linear(70, 5)
        self.LogSoftmax = tnn.LogSoftmax(dim=1)

    def forward(self, input, length):
        shape = input.shape[0]
        out, (h_n, c_n) = self.lstm(input)
        output = out[torch.arange(shape),length-1]
        output = self.Linear1(output)
        output = self.Relu(output)
        x1 = self.Linear_2(output)
        x2 = self.Linear_5(output)
        
        #return torch.squeeze(x1), torch.squeeze(x2)
        return self.LogSoftmax(x1), self.LogSoftmax(x2)

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training. 
    """

    def __init__(self):
        super(loss, self).__init__()
        #self.lossf= tnn.L1Loss()
        self.lossf = tnn.CrossEntropyLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        return (self.lossf(ratingOutput, ratingTarget) + self.lossf(categoryOutput,categoryTarget)) / 2

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################

################################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.005)
