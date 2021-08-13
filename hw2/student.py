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
import re
import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import numpy as np
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

def preprocessing(s):
    for i in range(len(s)):
        s[i] = re.sub(r'[^a-zA-Z0-9#$@*&()?!><:\']', '', s[i])
        s[i] = re.sub(r'\d+', '', s[i])
    """
    Called after tokenising but before numericalising.
    """
    return s

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

stopWords = {}
wordVectors = GloVe(name='6B', dim=100)

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
    # ratingOutput, categoryOutput = ratingOutput.cpu().tolist(), categoryOutput.cpu().tolist()
    # for i in range(len(ratingOutput)):
    #     ratingOutput[i] = np.argmax(ratingOutput[i])
    #     categoryOutput[i] = np.argmax(categoryOutput[i])
    ratingOutput = torch.argmax(ratingOutput, dim=-1)
    categoryOutput = torch.argmax(categoryOutput, dim=-1)
    return ratingOutput, categoryOutput
    # return torch.LongTensor(ratingOutput).to(device), torch.LongTensor(categoryOutput).to(device)

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
        self.hidden_dim = 128
        self.hidden_dim2 = 70
        self.rating_dim = 2
        self.category_dim = 5

        self.Relu = tnn.ReLU()

        self.lstm = tnn.LSTM(100, self.hidden_dim, batch_first=True, bidirectional=True, dropout=0.12, num_layers=2)
        self.fc = tnn.Linear(self.hidden_dim*2, self.hidden_dim2)
        self.rating_fc = tnn.Linear(self.hidden_dim2, self.rating_dim)
        self.category_fc = tnn.Linear(self.hidden_dim2, self.category_dim)
        self.LogSoftmax = tnn.LogSoftmax(dim=1)
        self.Softmax = tnn.Softmax(dim=1)


    def forward(self, input, length):
        # print(input.shape)
        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        # final_hidden_state = final_hidden_state.permute(1, 0, 2)
        # atten_out = self.attention_net_with_w(output, final_hidden_state)
        output = output[:,-1,:]
        # output = self.linear(output)
        # print(output.shape)
        # print(output.shape)
        output = self.Relu(output)
        output = self.fc(output)
        rating = self.rating_fc(output)
        # rating = rating[:,-1,:]
        # print(rating.shape)
        # print(rating.shape)
        category = self.category_fc(output)
        # category = category[:,-1,:]
        # print(category[:,-1].shape)
        
        # return_rating = self.LogSoftmax(rating)
        # return_category = self.LogSoftmax(category)
        # print(return_rating[:,-1])
        # print(return_category[:,-1])
        return rating, category

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.loss = tnn.CrossEntropyLoss()
    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        output = self.loss(ratingOutput, ratingTarget) + self.loss(categoryOutput, categoryTarget)*1.5
        return output

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.01)
