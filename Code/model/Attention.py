import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable
from model.CommonUtil import *
from model.Param import *

class Attention(nn.Module):

    def __init__(self, feature_size, compute_size):
        super(Attention, self).__init__()

        self.feature_size = feature_size
        self.compute_size = compute_size

        self.linear1 = nn.Sequential(
            nn.Linear(FEATHER_SIZE, self.compute_size),
            nn.ReLU()
        )

        self.cc = nn.Sequential(
            nn.Linear(self.compute_size * HEADS, 128),
            nn.ReLU()
        )

        self.ws = [nn.Parameter(torch.randn(self.compute_size * 2, self.compute_size)).cuda() for _ in range(0, HEADS)]
        self.ps = [nn.Parameter(torch.randn(self.compute_size, 1)).cuda() for _ in range(0, HEADS)]
        #
        # self.weight_w1 = nn.Parameter(torch.randn(self.compute_size * 2, self.compute_size))
        # self.weight_p1 = nn.Parameter(torch.randn(self.compute_size, 1))
        # self.weight_w2 = nn.Parameter(torch.randn(self.compute_size * 2, self.compute_size))
        # self.weight_p2 = nn.Parameter(torch.randn(self.compute_size, 1))
        # self.weight_w3 = nn.Parameter(torch.randn(self.compute_size * 2, self.compute_size))
        # self.weight_p3 = nn.Parameter(torch.randn(self.compute_size, 1))
        # self.weight_w4 = nn.Parameter(torch.randn(self.compute_size * 2, self.compute_size))
        # self.weight_p4 = nn.Parameter(torch.randn(self.compute_size, 1))
        # self.weight_w5 = nn.Parameter(torch.randn(self.compute_size * 2, self.compute_size))
        # self.weight_p5 = nn.Parameter(torch.randn(self.compute_size, 1))
        # self.weight_w6 = nn.Parameter(torch.randn(self.compute_size * 2, self.compute_size))
        # self.weight_p6 = nn.Parameter(torch.randn(self.compute_size, 1))
        # self.weight_w7 = nn.Parameter(torch.randn(self.compute_size * 2, self.compute_size))
        # self.weight_p7 = nn.Parameter(torch.randn(self.compute_size, 1))
        # self.weight_w8 = nn.Parameter(torch.randn(self.compute_size * 2, self.compute_size))
        # self.weight_p8 = nn.Parameter(torch.randn(self.compute_size, 1))

        self.weight_v = nn.Parameter(torch.randn(self.feature_size, self.compute_size))




    def head(self, qk, v, w, p, SEQ_LENGTH, BATCH_LENGTH):

        us = torch.tanh(torch.matmul(qk, w))
        atts = torch.matmul(us, p).view(SEQ_LENGTH, -1).permute(1, 0)
        att_scores = F.softmax(atts, dim=1).view(BATCH_LENGTH, SEQ_LENGTH, -1)

        vs = v * att_scores
        vs = torch.sum(vs, dim=1)
        return vs


    def forward(self, d1, d2, d2_size):

        SEQ_LENGTH = d2.size()[0]
        BATCH_LENGTH = d2.size()[1]

        d3 = d2.view(-1, self.feature_size)
        d3 = self.linear1(d3)

        d4 = [d1 for _ in range(0, SEQ_LENGTH)]
        d4 = torch.cat(d4, dim=0)
        d4 = self.linear1(d4)

        d5 = torch.cat([d3, d4], dim=1)
        d5 = d5.view(SEQ_LENGTH, BATCH_LENGTH, -1)

        tv = torch.tanh(torch.matmul(d2, self.weight_v)).permute(1, 0, 2)

        # print(d5.size(), tv.size())


        vs = []
        for i in range(0, HEADS):
            w = self.ws[i]
            p = self.ps[i]
            v = self.head(d5, tv, w, p, SEQ_LENGTH, BATCH_LENGTH)
            vs.append(v)

        # v1 = self.head(d5, tv, self.weight_w1, self.weight_p1, SEQ_LENGTH, BATCH_LENGTH)
        # v2 = self.head(d5, tv, self.weight_w2, self.weight_p2, SEQ_LENGTH, BATCH_LENGTH)
        # v3 = self.head(d5, tv, self.weight_w3, self.weight_p3, SEQ_LENGTH, BATCH_LENGTH)
        # v4 = self.head(d5, tv, self.weight_w4, self.weight_p4, SEQ_LENGTH, BATCH_LENGTH)
        # v5 = self.head(d5, tv, self.weight_w5, self.weight_p5, SEQ_LENGTH, BATCH_LENGTH)
        # v6 = self.head(d5, tv, self.weight_w6, self.weight_p6, SEQ_LENGTH, BATCH_LENGTH)
        # v7 = self.head(d5, tv, self.weight_w7, self.weight_p7, SEQ_LENGTH, BATCH_LENGTH)
        # v8 = self.head(d5, tv, self.weight_w8, self.weight_p8, SEQ_LENGTH, BATCH_LENGTH)
        #
        #
        # # vs = torch.cat([v1, v2, v3, v4], dim=1)
        # vs = torch.cat([v1, v2, v3, v4, v5, v6, v7, v8], dim=1)
        # vs = torch.cat([v1, v2], dim=1)

        vs = torch.cat(vs, dim=1)
        return self.cc(vs)

    # def forward(self, d1, d2, d2_size):
    #
    #     SEQ_LENGTH = d2.size()[0]
    #     BATCH_LENGTH = d2.size()[1]
    #
    #     d3 = d2.view(-1, self.feature_size)
    #
    #     d4 = [d1 for _ in range(0, SEQ_LENGTH)]
    #     d4 = torch.cat(d4, dim=0)
    #
    #     d5 = torch.cat([d3, d4], dim=1)
    #     d5 = d5.view(SEQ_LENGTH, BATCH_LENGTH, -1)
    #
    #     us = torch.tanh(torch.matmul(d5, self.weight_w))
    #     atts = torch.matmul(us, self.weight_p).view(SEQ_LENGTH, -1).permute(1, 0)
    #     att_scores = F.softmax(atts, dim=1).view(BATCH_LENGTH, SEQ_LENGTH, -1)
    #
    #     vs = torch.tanh(torch.matmul(d2, self.weight_v)).permute(1, 0, 2)
    #     vs = vs * att_scores
    #     vs = torch.sum(vs, dim=1)
    #
    #     return vs


