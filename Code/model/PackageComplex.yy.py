
import numpy as np
import heapq
import random
from numpy import *
from common.file_util import fu_load_json
from common.file_util import fu_save_json
from common.file_util import fu_save_csv
from model.Param import *
from model.DataUtil import *
from model.CommonUtil import *
from model.Model import *
from model.ModelComplex import *



user_num = get_user_num()
transformModel = TransformModel()
transformModel = try_cuda(transformModel)
transform_optimiser = optim.SGD(params=transformModel.parameters(), lr=LEARN_RATE1)

complexModel = ModelComplex(user_num, transformModel)
complexModel = try_cuda(complexModel)
complex_optimiser = optim.SGD(params=complexModel.parameters(), lr=LEARN_RATE2)
complex_optimiser2 = optim.SGD(params=complexModel.parameters(), lr=LEARN_RATE3)



def create_train_transform_data(transform_model_data, feature_index):

    items = []
    loader = du_create_loader(len(transform_model_data), TRANSFORM_BATCH_SIZE)
    for step, (batch_x, batch_y) in enumerate(loader):

        data1, data2, data2_size, data3, data3_size, labels = du_get_transform_batch_data(batch_x, transform_model_data, feature_index)
        item = {"data1":data1, "data2":data2, "data2_size":data2_size, "data3":data3, "data3_size":data3_size, "labels":labels}
        items.append(item)

    return items


def train_transform(global_transform_data, net, optimiser):

    random.shuffle(global_transform_data)

    num = 0
    total_num = 0
    train_loss = create_0_value()

    for d in global_transform_data:

        data1 = d['data1']
        data2 = d['data2']
        data3 = d['data3']
        data2_size = d['data2_size']
        data3_size = d['data3_size']
        labels = d['labels']

        labels = create_float_value(labels)
        labels = labels.view(-1, 1)
        data1 = create_float_value(data1)
        data2 = create_float_value(data2)
        data3 = create_float_value(data3)

        total_num = total_num + len(labels)

        predict, loss = net.forward_and_loss(data1, data2, data2_size, data3, data3_size, labels)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        train_loss = train_loss + loss

        # print(labels, predict)
        # print(len(labels), len(predict))

        for i in range(0, len(predict)):
            p = predict[i]
            label = labels[i]
            if p >= 0.5 and label == 1:
                num = num + 1
            if p < 0.5 and label == 0:
                num = num + 1

    average_loss = train_loss / total_num
    print("train_transform loss is : " + str(average_loss))
    print("train_transform accuracy is : " + str(num / total_num))
    return average_loss, num / total_num



def test_transform(global_transform_data, net):

    num = 0
    nump = 0
    labelp = 0
    total_num = 0


    for d in global_transform_data:

        data1 = d['data1']
        data2 = d['data2']
        data3 = d['data3']
        data2_size = d['data2_size']
        data3_size = d['data3_size']
        labels = d['labels']

        labels = create_float_value(labels)
        labels = labels.view(-1, 1)
        data1 = create_float_value(data1)
        data2 = create_float_value(data2)
        data3 = create_float_value(data3)

        total_num = total_num + len(labels)

        predict = net.forward(data1, data2, data2_size, data3, data3_size)

        # print(labels, predict)
        # print(len(labels), len(predict))

        for i in range(0, len(predict)):
            p = predict[i]
            label = labels[i]
            if p >= 0.5 and label == 1:
                num = num + 1
            if p < 0.5 and label == 0:
                num = num + 1
            if p >= 0.5:
                nump = nump + 1
            if label == 1:
                labelp = labelp + 1

    print("test_transform accuracy is : " + str(num / total_num))
    print(nump, labelp, total_num)


def create_train_complex_data(complex_model_data, feature_index):


    items = []
    loader = du_create_loader(len(complex_model_data), COMPLEX_BATCH_SIZE)
    for step, (batch_x, batch_y) in enumerate(loader):


        batch_datas = [complex_model_data[int(i)] for i in batch_x]
        labels = []
        datas = []

        for d in batch_datas:

            if d['top10_include'] == 0:
                continue

            top10_probabilities = d['top10_probabilities']
            top10_categories = d['top10_categories']
            label = int(d['top10_label'])

            data1, data2, data2_size, data3, data3_size = du_get_complex_data(d, feature_index)

            item = {'ptopk':top10_probabilities, 'ctopk':top10_categories, 'data1':data1, 'data2':data2, 'data2_size':data2_size, 'data3':data3, 'data3_size':data3_size}
            datas.append(item)
            labels.append(label)

        if len(datas) == 0:
            continue

        targets = create_variable_by_Tensor(torch.LongTensor(labels))

        item = {"datas":datas, "targets":targets, "labels":labels}
        items.append(item)

    return items


def create_test_complex_data(complex_model_data, feature_index):

    items = []
    loader = du_create_loader(len(complex_model_data), COMPLEX_BATCH_SIZE)
    for step, (batch_x, batch_y) in enumerate(loader):

        batch_datas = [complex_model_data[int(i)] for i in batch_x]
        datas = []

        for d in batch_datas:

            top10_probabilities = d['top10_probabilities']
            top10_categories = d['top10_categories']
            rlabel = d['label']

            data1, data2, data2_size, data3, data3_size = du_get_complex_data(d, feature_index)

            item = {'ptopk':top10_probabilities, 'ctopk':top10_categories, 'data1':data1, 'data2':data2, 'data2_size':data2_size, 'data3':data3, 'data3_size':data3_size, 'rlabel':rlabel}
            datas.append(item)

        item = {"datas":datas}
        items.append(item)

    return items


def train_complex(global_complex_model_data, net, optimiser):

    random.shuffle(global_complex_model_data)

    num = 0
    total_num = 0
    train_loss = create_0_value()

    for d in global_complex_model_data:

        datas = d['datas']
        targets = d['targets']
        labels = d['labels']
        total_num = total_num + len(labels)

        predict, loss = net.forward_and_loss(datas, targets)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        train_loss = train_loss + loss

        for i in range(0, len(predict)):
            p = predict[i]
            p = p.cpu().detach().numpy()
            p = np.argmax(p)
            label = labels[i]
            if p == label:
                num = num + 1


    average_loss = train_loss / total_num
    print("train_complex loss is : " + str(average_loss))
    print("train_complex accuracy is : " + str(num / total_num))
    return average_loss, num / total_num


def test_complex(test_model_data, net):

    num = 0
    total_num = 0

    for d in test_model_data:

        datas = d['datas']
        total_num = total_num + len(datas)

        predict = net.forward(datas)

        for i in range(0, len(predict)):
            p = predict[i]
            p = p.cpu().detach().numpy()
            p = np.argmax(p)
            ctopk = datas[i]['ctopk']
            y = ctopk[p]
            rlabel = datas[i]['rlabel']
            if y == rlabel:
                num = num + 1

    print("test_complex accuracy is : " + str(num / total_num))
    return num / total_num


def get_train_fold_data(fold, feature_index):

    global_transform_model_data = []
    global_complex_model_data = []

    if fold < PRE_TRAIN_LENGTH:
        return [], []

    for i in range(fold + 1 - PRE_TRAIN_LENGTH, fold + 1):

        transform_model_data = get_transform_model_data(i)
        complex_model_data = get_complex_model_data(i, 5)

        print('fold : '+str(i), len(transform_model_data), len(complex_model_data))
        print(len(global_transform_model_data), len(global_complex_model_data))

        fold_transform_model_data = create_train_transform_data(transform_model_data, feature_index)
        global_transform_model_data.extend(fold_transform_model_data)

        fold_complex_model_data = create_train_complex_data(complex_model_data, feature_index)
        global_complex_model_data.extend(fold_complex_model_data)


    return global_transform_model_data, global_complex_model_data

def run_fold(fold):

    global transformModel
    global transform_optimiser

    global complexModel
    global complex_optimiser
    global complex_optimiser2

    feature_index = get_feature_index()

    global_transform_model_data, global_complex_model_data = get_train_fold_data(fold, feature_index)

    test_transform_model_data_next = get_transform_model_data(fold + 1)
    test_transform_model_data_next = create_train_transform_data(test_transform_model_data_next, feature_index)

    test_comple_model_data_now = get_complex_model_data(fold, 5)
    test_comple_model_data_now = create_test_complex_data(test_comple_model_data_now, feature_index)

    test_comple_model_data_next = get_complex_model_data(fold + 1, 5)
    test_comple_model_data_next = create_test_complex_data(test_comple_model_data_next, feature_index)
    print("global_transform_model_data "+str(len(global_transform_model_data))+" global_complex_model_data "+str(len(global_complex_model_data)))


    max_accuracy3 = 0
    #
    # for i in range(0, 10):
    #     train_transform(global_transform_model_data, transformModel, transform_optimiser)

    # for _ in range(0, 10):
    #     loss1, accuracy1 = train_transform(global_transform_model_data, transformModel, transform_optimiser)
    # test_transform(test_transform_model_data, transformModel)
    # test_complex(test_comple_model_data, complexModel)

    for i in range(POCH):

        print("FOLD "+str(fold)+" POCH "+str(i))

        # loss1, accuracy1 = train_transform(global_transform_model_data, transformModel,transform_optimiser)
        # test_transform(test_transform_model_data_next, transformModel)

        # loss2, accuracy2 = train_complex(global_complex_model_data, complexModel, complex_optimiser)
        for _ in range(0, 10):
            loss2, accuracy2 = train_complex(global_complex_model_data, complexModel, complex_optimiser2)

        # if np.isnan(loss1[0].cpu().detach().numpy()) or np.isnan(loss2[0].cpu().detach().numpy()):
        # if np.isnan(loss2[0].cpu().detach().numpy()):
        #
        #
        #     user_num = get_user_num()
        #     transformModel = TransformModel()
        #     transformModel = try_cuda(transformModel)
        #     transform_optimiser = optim.SGD(params=transformModel.parameters(), lr=LEARN_RATE1)
        #
        #     complexModel = ModelComplex(user_num, transformModel)
        #     complexModel = try_cuda(complexModel)
        #     complex_optimiser = optim.SGD(params=complexModel.parameters(), lr=LEARN_RATE2)
        #     complex_optimiser2 = optim.SGD(params=complexModel.parameters(), lr=LEARN_RATE3)

            # for ii in range(0, 10):
            #     train_transform(global_transform_model_data, transformModel, transform_optimiser)


        complexModel.eval()
        accuracy_now = test_complex(test_comple_model_data_now, complexModel)
        accuracy_next = test_complex(test_comple_model_data_next, complexModel)
        complexModel.train()
        if accuracy_now > max_accuracy3:
            max_accuracy3 = accuracy_now
            torch.save(complexModel, '../data_model/fold_complex_' + str(fold) + '.pth')

        if accuracy_next > 0.6:
            break

    # for d in complex_model_data:
    #
    #     bug_id = d['bug_id']
    #     original_predict = d['original_predict']
    #     original_category = d['original_category']
    #     label = d['label']




def run():

    for i in range(PRE_TRAIN_LENGTH, 100):
        run_fold(i)
    # break

run()