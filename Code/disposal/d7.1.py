
import numpy as np
import heapq
from numpy import *
from common.file_util import fu_load_json
from common.file_util import fu_save_json
from common.file_util import fu_save_csv
from model.Param import *
from model.DataUtil import *
from model.CommonUtil import *


origin_data = fu_load_json('../data_base/ec_attr_num.json')
tossing_data = fu_load_json('../data_base/ec_tossing_path.json')
meta_data = fu_load_json('../data_base/ec_meta.json')



fold_index = meta_data['fold_ids']
fold_index = [int(d) for d in fold_index]
fold_start_time = []


tossing_index = {}
for d in tossing_data:
    #print(d)
    tossing_index[int(d['bug_id'])] = d

for fi in fold_index:
    st = tossing_index[fi]['open_time']
    fold_start_time.append(st)

origin_index = {}
for d in origin_data:
    origin_index[int(d['bug_id'])] = d

fid2tossing = {}
ids2tossing = {}
userLastAvilabel = {}


data_p2_index = {}
data_p3_index = {}


original_feature_index = fu_load_json('../data_disposal/ec_3.4_last_train_data.json')
original_feature_index[MISSING_FLAG] = [0 for i in range(0, FEATHER_SIZE)]
feature_index = {}

for key in original_feature_index:
    feature_index[int(key)] = original_feature_index[key]



def load_ml_data(fold, topk):
    path = '../data_temp/ec_ml_nv_fold_'+str(fold)+'_topk_'+str(topk)+'.json'
    return fu_load_json(path)



def create_fold_index(fold):

    # data_p1 = fu_load_json('../data_net/ec_p1_fold' + str(fold + 2) + '.json')
    data_p2 = fu_load_json('../data_net/ec_p2_fold' + str(fold + 2) + '.json')
    data_p3 = fu_load_json('../data_net/ec_p3_fold' + str(fold + 2) + '.json')

    for d in data_p2:
        dd = data_p2[d]
        if len(dd) > LOG_RETAIN_LENGTH:
            dd = dd[len(dd)-LOG_RETAIN_LENGTH:]
        data_p2_index[int(d)] = dd


    for d in data_p3:
        dd = data_p3[d]
        if len(dd) > LOG_RETAIN_LENGTH:
            dd = dd[len(dd)-LOG_RETAIN_LENGTH:]
        data_p3_index[int(d)] = dd

    data_p2_index[MISSING_FLAG] = [MISSING_FLAG]
    data_p3_index[MISSING_FLAG] = [MISSING_FLAG]




def calc_transform(net, bug_id, cs):


    data1 = [int(bug_id) for i in range(0, len(cs))]
    data1, data2, data2_size, data3, data3_size = du_get_predict_data(data1, cs, cs, data_p2_index, data_p3_index, feature_index)

    predict = net.forward(data1, data2, data2_size, data3, data3_size)
    # predict_exp = predict.exp()
    # partition = predict_exp.sum()
    #
    # return predict_exp / partition
    return predict


def adjust_predict(net, fold, bug_id, original_predict, original_category):

    user_num = meta_data['user_num']
    ps = []
    cs = []

    original_predict_order = heapq.nlargest(10, range(len(original_predict)), original_predict.__getitem__)
    for i in range(0, len(original_predict_order)):
        index = original_predict_order[i]
        ps.append(original_predict[index])
        cs.append(original_category[index])


    v = create_float_value([ps])
    # v = create_float_value([[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
    w = calc_transform(net, bug_id, cs)
    # print(w)
    # print(w.size())
    # print(w[0].sum())

    # for ss in w[0]:
    #     print(ss)

    z = v.mm(w)

    r = heapq.nlargest(10, range(len(z[0])), z[0].__getitem__)
    return r

def run(fold):


    data_p2_index.clear()
    data_p3_index.clear()
    create_fold_index(fold)

    ml_data = load_ml_data(fold, 5)

    net = torch.load('../data_model/fold_'+str(fold-1)+'.pth')
    net = try_cuda(net)

    for d in ml_data:

        bug_id = d['bug_id']
        original_predict = d['original_predict']
        original_category = d['original_category']

        a = adjust_predict(net, fold, bug_id, original_predict, original_category)
        d['adjust_predict'] = a

        # print(d['label'], d['predict'], d['adjust_predict'])


    prs = []
    for j in range(1, 6):
        num = 0
        for md in ml_data:
            ap = md['adjust_predict'][0: j]
            if md['label'] in ap:
                num = num + 1

        pr = num / len(ml_data)
        prs.append(pr)
    return prs


rows = []

for i in range(2, 3):

    row_data = ['fold' + str(i)]

    prs = run(i)
    print('fold: ' + str(i) + ' prs: ' + str(prs))

    for pr in prs:
        row_data.append(pr)
    rows.append(row_data)

head = ['fold', 'top1', 'top2', 'top3', 'top4', 'top5']
fu_save_csv(head, rows, '../data_result/ec_ml_nv_tg_data_1.csv')