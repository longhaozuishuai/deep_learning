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

"""


Our group made a program with a network consist of two layers of bidirectional LSTM(input_size:100, hidden_size:128, dropout:0.12), 
two layers of fully connected layers and output with LogSoftmax function. For rating of positive and negative the output dimension 
is two and for bussiness rating the output dimension is 5. In addition we choose CrossEntropyLoss as loss function and the weight of 
bussiness rate set to 1.5. For processing of input data, we removed error words and digits. And we use Adam instead of SGD as optimiser
function. At last, the weight score can be more than 84%.

Firstly, I tried convolutional layers and fully connected layers, I failed and the weighted score is very low, especially the
bussiness category rate, then I tried LSTM with sigmod, relu, tahn function and one or two or three full connected layers, 
this time the weighted socre can improved to more than 60% and relu function proferom the best. Then I tried to change the
parameters and optimiser to admn function, this time the weighted socre can reach around 80% but the bussiness category rate 
is always low. After that, I tried to change the processing of input data, I changed the stopwords and it does not work at all.
But I removed the digits and error words, the weighted score can be more than 83%.

"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import torch.nn.functional as F
import re
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
    for i in range(len(sample)):
        sample[i] = re.sub(r'[^a-zA-Z0-9#$@*&()?!><:\']', '', sample[i])
    
    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

stopWords = {'i', 'me', 'my', 'the', 'a', 'an', 'where', 'that', 'was', 'did', 'where', 'were'}
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
        return self.lossf(ratingOutput, ratingTarget) + self.lossf(categoryOutput,categoryTarget) * 1.5

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################

################################################################################

trainValSplit = 1.0
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.005)
