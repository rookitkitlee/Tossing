import torch
import torch.nn as nn

from torch.autograd import Variable


def try_cuda(target):
    if torch.cuda.is_available():
        return target.cuda()
    else:
        return target

def create_0_value():
    if torch.cuda.is_available():
        return Variable(torch.FloatTensor([0.]).cuda())
    else:
        return Variable(torch.FloatTensor([0.]))

def create_0_varibale(s1):
    if torch.cuda.is_available():
        return Variable(torch.zeros(s1).cuda(), requires_grad=True)
    else:
        return Variable(torch.zeros(s1), requires_grad=True)

def create_0_matrix(s1, s2):
    if torch.cuda.is_available():
        return Variable(torch.zeros(s1, s2).cuda(), requires_grad=True)
    else:
        return Variable(torch.zeros(s1, s2), requires_grad=True)

def create_variable_by_Tensor(data):
    if torch.cuda.is_available():
        return Variable(data.cuda())
    else:
        return Variable(data)
