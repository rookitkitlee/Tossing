
import numpy as np
import heapq
from numpy import *
from common.file_util import fu_load_json
from common.file_util import fu_save_json
from common.file_util import fu_save_csv
from model.Param import *
from model.DataUtil import *
from model.CommonUtil import *
from model.Model import *
from model.ModelComplex import *



def run(fold, weight, multi, counts):

    feature_index = get_feature_index()

    complex_model_data = get_complex_model_data(fold, 5)

    complex_model = torch.load('../data_model/fold_complex_' + str(fold-1) + '.pth')
    # complex_model = torch.load('../data_model/fold_complex_75.pth')
    complex_model.eval()

    loader = du_create_loader(len(complex_model_data), 8)
    for step, (batch_x, batch_y) in enumerate(loader):

        batch_datas = [complex_model_data[int(i)] for i in batch_x]
        datas = []

        for d in batch_datas:

            top10_probabilities = d['top10_probabilities']
            top10_categories = d['top10_categories']

            data1, data2, data2_size, data3, data3_size = du_get_complex_data(d, feature_index)

            item = {'ptopk': top10_probabilities, 'ctopk': top10_categories, 'data1': data1, 'data2': data2,
                    'data2_size': data2_size, 'data3': data3, 'data3_size': data3_size}
            datas.append(item)


        predict = complex_model.forward_predict(datas, weight, multi, counts)  #counts 阿尔法

        for i in range(0, len(datas)):

            d = batch_datas[i]
            p = predict[i]
            ctopk = datas[i]['ctopk']
            original_predict_order = heapq.nlargest(5, range(len(p)), p.__getitem__)
            adjust_order = []
            for o in original_predict_order:
                adjust_order.append(ctopk[o])
            d['adjust_predict'] = adjust_order

            # print(d['label'], d['predict'], d['adjust_predict'])


    prs = []
    for j in range(1, 6):
        num = 0
        for md in complex_model_data:
            ap = md['adjust_predict'][0: j]
            if md['label'] in ap:
                num = num + 1

        pr = num / len(complex_model_data)
        prs.append(pr)
    print(fold, weight, prs)


for i in range(PRE_TRAIN_LENGTH + 1, 101):

    # run(i, 1, 1, 10)
    # run(i, 1, 2, 10)
    # run(i, 1, 3, 10)
    # run(i, 1, 4, 10)
    # run(i, 1, 5, 10)
    # run(i, 1, 6, 10)
    # run(i, 1, 7, 10)
    # run(i, 1, 8, 10)
    # run(i, 1, 9, 10)
    # run(i, 1, 10, 10)

    # run(i, 1, 5, 10)
    # run(i, 1, 10, 10)
    #
    #
    # run(i, 0.5, 5, 10)
    run(44, 0.5, 10, 10)
    run(44, 1, 10, 10)




    # run(44, 1, 10, 5)
    # run(44, 1, 9, 5)
    # run(44, 1, 8, 5)
    # run(44, 1, 7, 5)
    # run(44, 1, 6, 5)
    # run(44, 1, 5, 5)
    # run(44, 1, 4, 5)
    # run(44, 1, 3, 5)
    # run(44, 1, 2, 5)
    # run(44, 1, 1, 5)
    # run(44, 1, 5, 1)
    # run(44, 1, 5, 2)
    # run(44, 1, 5, 3)
    # run(44, 1, 5, 4)
    # run(44, 1, 5, 5)
    # run(44, 1, 5, 6)
    # run(44, 1, 5, 7)
    # run(44, 1, 5, 8)
    # run(44, 1, 5, 9)
    # run(44, 1, 5, 10)
    # run(44, 1, 5, 100)

    # run(i)
    break