# 使用 attr one_hot  和 title——desc 计算准确率

import heapq

from common.file_util import fu_load_json
from common.file_util import fu_save_json
from common.file_util import fu_save_csv
from sklearn.naive_bayes import MultinomialNB
from numpy import *

last_data = fu_load_json('../data_disposal/ec_3.3_last_train_data.json')
meta_data = fu_load_json('../data_base/ec_meta.json')
folds_meta = meta_data['folds']

attr = [d['attr'] for d in last_data]
title = [d['title_words'] for d in last_data]
desc = [d['desc_words'] for d in last_data]
label = [d['label'] for d in last_data]

ids = [int(d['bug_id']) for d in last_data]

x = concatenate((attr, title, desc), axis=1)
y = [int(o) for o in label]

# print(y)

def calc_top_k(predict, test, train_index, id_test, fold_index, k):
    predict_data = []
    num = 0
    for i in range(0, len(predict)):
        p = predict[i]
        p = heapq.nlargest(k, range(len(p)), p.__getitem__)
        p = [train_index[o] for o in p]
        t = test[i]
        if t in p:
            num = num + 1

        original_predict = predict[i]
        original_category = train_index
        item = {'bug_id':id_test[i], 'label':test[i], 'predict':p, 'original_predict':original_predict.tolist(), 'original_category':original_category}
        #item = {'bug_id': id_test[i], 'label': test[i], 'predict': p}
        predict_data.append(item)

    path = '../data_temp/ec_ml_nv_fold_'+str(fold_index)+'_topk_'+str(k)+'.json'
    if k == 5:
        fu_save_json(predict_data, path)
    return num / len(predict)


def get_data_by_fold(fold_index):

    if fold_index == 0:
        min_id = 0
    else:
        min_id = folds_meta[fold_index]

    if fold_index == 100:
        max_id = 10000000
    else:
        max_id = folds_meta[fold_index + 1]

    xs = []
    ys = []
    bid = []


    for i in range(0, len(ids)):
        id = ids[i]
        if id > min_id and id <= max_id:
            xs.append(x[i])
            ys.append(y[i])
            bid.append(id)

    return xs, ys, bid


def get_train_data_by_fold(fold_index):

    if fold_index == 100:
        max_id = 10000000
    else:
        max_id = folds_meta[fold_index + 1]

    xs = []
    ys = []
    bid = []

    for i in range(0, len(ids)):
        id = ids[i]
        if id <= max_id:
            xs.append(x[i])
            ys.append(y[i])
            bid.append(id)

    return xs, ys, bid

rows = []
for i in range(1, 101):


    x_train, y_train, id_train = get_train_data_by_fold(i - 1)
    x_test, y_test, id_test = get_data_by_fold(i)


    # print(len(x_train), len(y_train))
    # print(y_train)

    clf = MultinomialNB().fit(x_train, y_train)
    y_predict = clf.predict_proba(x_test)
    y_train_index = list(set(y_train))

    # print(y_train_index)


    row_data = ['fold'+str(i)]

    for k in range(1, 6):
        pr = calc_top_k(y_predict, y_test, y_train_index, id_test, i, k)
        print('fold: '+str(i)+' top: '+str(k)+' accuracy: '+str(pr))
        row_data.append(pr)

    rows.append(row_data)

head = ['fold', 'top1', 'top2', 'top3', 'top4', 'top5']
fu_save_csv(head, rows, '../data_result/ec_ml_nv_data_1.csv')