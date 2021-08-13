from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

conv = nn.Conv1d(1, 1, 3, bias = False)
sample = torch.randn(1, 1, 7)
conv(Variable(sample))
print(conv.weight)