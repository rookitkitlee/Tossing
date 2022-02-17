import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable
from model.CommonUtil import *

class Attention(nn.Module):

    def __init__(self, feature_size, compute_size):
        super(Attention, self).__init__()

        self.feature_size = feature_size
        self.compute_size = compute_size

        self.weight_w = nn.Parameter(torch.randn(self.feature_size * 2, self.compute_size))
        self.weight_p = nn.Parameter(torch.randn(self.compute_size, 1))

        self.weight_v = nn.Parameter(torch.randn(self.feature_size, self.compute_size))


    def forward(self, d1, d2, d2_size):

        SEQ_LENGTH = d2.size()[0]
        BATCH_LENGTH = d2.size()[1]

        d3 = d2.view(-1, self.feature_size)

        d4 = [d1 for _ in range(0, SEQ_LENGTH)]
        d4 = torch.cat(d4, dim=0)

        d5 = torch.cat([d3, d4], dim=1)
        d5 = d5.view(SEQ_LENGTH, BATCH_LENGTH, -1)

        us = torch.tanh(torch.matmul(d5, self.weight_w))
        atts = torch.matmul(us, self.weight_p).view(SEQ_LENGTH, -1).permute(1, 0)
        att_scores = F.softmax(atts, dim=1).view(BATCH_LENGTH, SEQ_LENGTH, -1)

        vs = torch.tanh(torch.matmul(d2, self.weight_v)).permute(1, 0, 2)
        vs = vs * att_scores
        vs = torch.sum(vs, dim=1)

        return vs


