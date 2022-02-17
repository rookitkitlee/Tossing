import torch
import torch.nn as nn

from model.Param import *

class Classifier(nn.Module):

    def __init__(self, base_size):
        super(Classifier, self).__init__()
        self.cc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):

        out = self.cc(x)

        # out = out.permute(1, 0)
        # out = out[1]
        # out = out.view(-1, 1)
        # print('---------------------')
        # print(out.size())
        return self.sigmoid(out)
        # return out

    def forward_and_loss(self, operate_data, labels, weights):
        out = self.cc(operate_data)
        # classifier_loss = - torch.mean(weights * torch.log(out))

        # print('out')
        # print(out)
        # print('log')
        # print(torch.log(out))
        # print('weights')
        # print(weights)
        # print('mu')
        # print(weights * torch.log(out))
        # print('loss')
        # print(classifier_loss)
        classifier_loss = self.loss_fn(out, labels)
        return out, classifier_loss

# 参考 https://blog.csdn.net/m0_37306360/article/details/79307818
# https://www.jb51.net/article/167880.htm