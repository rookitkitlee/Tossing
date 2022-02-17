import torch
import torch.nn as nn

from torch.autograd import Variable
from model.Attention import Attention
from model.Classifier import Classifier
from model.Param import *


class TransformModel(nn.Module):

    def __init__(self):
        super(TransformModel, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(FEATHER_SIZE, COMPUTE_SIZE),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(FEATHER_SIZE, COMPUTE_SIZE),
            nn.ReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(FEATHER_SIZE, COMPUTE_SIZE),
            nn.ReLU()
        )
        self.linear4 = nn.Sequential(
            nn.Linear(FEATHER_SIZE, COMPUTE_SIZE),
            nn.ReLU()
        )
        self.cc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
            # nn.Linear(COMPUTE_SIZE * 2, 8)
        )


        self.attn11 = Attention(FEATHER_SIZE, COMPUTE_SIZE)
        self.attn12 = Attention(FEATHER_SIZE, COMPUTE_SIZE)
        self.attn21 = Attention(FEATHER_SIZE, COMPUTE_SIZE)
        self.attn22 = Attention(FEATHER_SIZE, COMPUTE_SIZE)
        self.attn31 = Attention(FEATHER_SIZE, COMPUTE_SIZE)
        self.attn32 = Attention(FEATHER_SIZE, COMPUTE_SIZE)
        self.attn41 = Attention(FEATHER_SIZE, COMPUTE_SIZE)
        self.attn42 = Attention(FEATHER_SIZE, COMPUTE_SIZE)

        self.classifier = Classifier(8)


    #  data1 BATCH * COMPUTE_SIZE
    #  data2 LENGTH * BATCH * COMPUTE_SIZE
    #  data2_size 1 * BATCH
    #  data3s LENGTH * BATCH * COMPUTE_SIZE
    #  data3_sizes 1 * BATCH
    def constract_data(self, data1, data2, data2_size, data3, data3_size):

        # d1 = self.linear1(data1)
        # d2 = self.linear2(data1)
        # d3 = self.linear3(data1)
        # d4 = self.linear4(data1)

        p21 = self.attn11(data1, data2, data2_size)
        # p22 = self.attn21(data1, data2, data2_size)
        # p23 = self.attn31(data1, data2, data2_size)
        # p24 = self.attn41(data1, data2, data2_size)

        p31 = self.attn12(data1, data3, data3_size)
        # p32 = self.attn22(data1, data3, data3_size)
        # p33 = self.attn32(data1, data3, data3_size)
        # p34 = self.attn42(data1, data3, data3_size)

        # operate_data = torch.cat([d1, d2, d3, d4, p21, p22, p23, p24, p31, p32, p33, p34], dim=1)
        # operate_data = torch.cat([d1, p21, p31], dim=1)
        operate_data = torch.cat([p21, p31], dim=1)
        # operate_data = torch.cat([p21, p22, p23, p24, p31, p32, p33, p34], dim=1)

        return operate_data

    def forward_part(self, data1, data2, data2_size, data3, data3_size):
        operate_data = self.constract_data(data1, data2, data2_size, data3, data3_size)
        # print(operate_data.size())
        return self.cc(operate_data)

    def forward(self, data1, data2, data2_size, data3, data3_size):

        operate_data = self.forward_part(data1, data2, data2_size, data3, data3_size)
        return self.classifier.forward(operate_data)

    def forward_and_loss(self, data1, data2, data2_size, data3, data3_size, labels, weights):

        operate_data = self.forward_part(data1, data2, data2_size, data3, data3_size)
        return self.classifier.forward_and_loss(operate_data, labels, weights)
