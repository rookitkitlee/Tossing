
import torch
import heapq
import torch.nn as nn

from model.Model import TransformModel
from model.CommonUtil import *
from model.DataUtil import *

class ModelComplex(nn.Module):

    def __init__(self, user_num, transform_model):
        super(ModelComplex, self).__init__()

        self.transform_model = transform_model
        self.user_num = user_num
        self.loss_fn = nn.CrossEntropyLoss()
        self.cc = nn.Sequential(

            nn.ReLU(),
            nn.Linear(80, 10)
        )


    def constract_data(self, datas, weight, multi, counts):

        data1s = []
        data2s = []
        data2_sizes = []
        data3s = []
        data3_sizes = []

        for i in range(0, LOG_RETAIN_LENGTH):
            data2s.append([])
            data3s.append([])

        for d in datas:

            data1 = d['data1']
            data2 = d['data2']
            data2_size = d['data2_size']
            data3 = d['data3']
            data3_size = d['data3_size']

            data1 = create_float_value(data1)
            data2 = create_float_value(data2)
            data3 = create_float_value(data3)

            data1s.append(data1)
            data2_sizes.extend(data2_size)
            data3_sizes.extend(data3_size)

            for i in range(0, LOG_RETAIN_LENGTH):
                data2s[i].append(data2[i])
                data3s[i].append(data3[i])

        data1s = torch.cat(data1s, dim=0)
        for i in range(0, LOG_RETAIN_LENGTH):
            data2s[i] = torch.cat(data2s[i], dim=0)
            data3s[i] = torch.cat(data3s[i], dim=0)

        data2s = torch.cat(data2s, dim=0).view(LOG_RETAIN_LENGTH, -1, FEATHER_SIZE)
        data3s = torch.cat(data3s, dim=0).view(LOG_RETAIN_LENGTH, -1, FEATHER_SIZE)

        return data1s, data2s, data2_sizes, data3s, data3_sizes

    def forward_train(self, datas, weight, multi, counts):

        data1s, data2s, data2_sizes, data3s, data3_sizes = self.constract_data(datas, weight, multi, counts)

        predicts = self.transform_model.forward_part(data1s, data2s, data2_sizes, data3s, data3_sizes)
        predicts = predicts * multi  # softmax * 10 ?

        zs = []
        for i in range(0, len(datas)):


            w = predicts[i * 100: (i + 1) * 100] #列表的切片操作 共100个
            w = w.view(10, -1) # 10个10维的列表 10*10

            w = self.cc(w)
            w = w.view(10, -1)

            w = torch.softmax(w, dim=1)  #Top 10做softmax
            zs.append(w)

        zs = torch.cat(zs, dim=0)
        return zs


    def forward_predict(self, datas, weight, multi, counts):

        data1s, data2s, data2_sizes, data3s, data3_sizes = self.constract_data(datas, weight, multi, counts)

        predicts = self.transform_model.forward(data1s, data2s, data2_sizes, data3s, data3_sizes)
        predicts = predicts * multi

        zs = []
        for i in range(0, len(datas)):
            d = datas[i]
            probabilitiesTopK = d['ptopk']
            v = create_float_value(probabilitiesTopK).view(1, -1)

            w = predicts[i * 100: (i + 1) * 100]
            w = w.view(10, -1)

            # w = self.cc(w)
            # w = w.view(10, -1)

            w = torch.softmax(w, dim=1)

            if counts > 1:  #阿尔法
                ww = w
                for _ in range(0, counts - 1):
                    w = w.mm(ww)  # w的阿尔法次方

            for m in range(0, len(w)): #β 不超过1
                w.data[m][m] = w.data[m][m] + weight

            z = v.mm(w)

            zs.append(z)

        zs = torch.cat(zs, dim=0)
        return zs


    def forward_and_loss(self, datas, label, weight, multi, counts):


        predict = self.forward_train(datas, weight, multi, counts)
        # print(predict.size(), label.size())
        loss = self.loss_fn(predict, label)
        return predict, loss



