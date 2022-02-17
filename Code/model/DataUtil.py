import torch
import torch.utils.data as Data
import heapq
from torch.autograd import Variable
from torch import optim
from model.Param import *
from common.file_util import *

def du_create_loader(length, batch_size):
    t = torch.linspace(0, length - 1, length)
    torch_dataset = Data.TensorDataset(t, t)
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    return loader



def du_constract_log_data(uids, feature_index, max_log_length):


    data_uid = uids

    data_size = [len(d) for d in data_uid]

    for i in range(0, len(data_uid)):
        d = data_uid[i]

        if len(d) > max_log_length:
            td = d[len(d)-max_log_length:]
            data_uid[i] = td

        elif len(d) < max_log_length:
            cha = max_log_length - len(d)
            chas = [MISSING_FLAG for _ in range(0, cha)]
            chas.extend(d)
            data_uid[i] = chas


    t_data_size = []
    for dz in data_size:
        if dz > max_log_length:
            t_data_size.append(max_log_length)
        else:
            t_data_size.append(dz)
    data_size = t_data_size


    # 倒序构造数据
    data = []
    for t in range(0, max_log_length):

        i = max_log_length - 1 - t

        ix = [d[i] for d in data_uid]
        fix = [feature_index[j] for j in ix]
        # fix = create_float_value(fix)
        data.append(fix)

    # data log_length * BATCH * COMPUTE_SIZE
    # data2_size 1 * BATCH
    if len(data) != LOG_RETAIN_LENGTH:
        print(data_size)

    return data, data_size



def du_get_transform_batch_data(batch_x, data, feature_index):

    bug_ids = [data[int(i)]['bug_id'] for i in batch_x]
    labels = [data[int(i)]['label'] for i in batch_x]
    fp_data = [data[int(i)]['fp_data'] for i in batch_x]  # BATCH * LOG_LENGTH
    tp_data = [data[int(i)]['tp_data'] for i in batch_x]  # BATCH * LOG_LENGTH
    weights = [data[int(i)]['weight'] for i in batch_x]

    data1 = [feature_index[i] for i in bug_ids]

    data2, data2_size = du_constract_log_data(fp_data, feature_index, LOG_RETAIN_LENGTH)
    data3, data3_size = du_constract_log_data(tp_data, feature_index, LOG_RETAIN_LENGTH)


    return data1, data2, data2_size, data3, data3_size, labels, weights


def du_get_complex_data(d, feature_index):

    # print('--------------------------------------')
    bug_id = d['bug_id']

    top10_data2 = d['top10_data2']
    top10_data3 = d['top10_data3']

    uid2 = []
    for uid in top10_data2:
        uids = [uid for _ in range(0, 10)]
        uid2.extend(uids)

    uid3 = []
    for _ in range(0, 10):
        for uid in top10_data3:
            uid3.append(uid)


    #  data1 100 COMPUTE_SIZE
    #  data2 LENGTH * 100 * COMPUTE_SIZE
    #  data2_size  1 * 100
    #  data3  LENGTH * 100 * COMPUTE_SIZE
    #  data3_size  1 * 100

    data1 = [feature_index[bug_id] for _ in range(0, 100)]
    data2, data2_size = du_constract_log_data(uid2, feature_index, LOG_RETAIN_LENGTH)
    data3, data3_size = du_constract_log_data(uid3, feature_index, LOG_RETAIN_LENGTH)


    return data1, data2, data2_size, data3, data3_size

def create_float_value(data):

    if torch.cuda.is_available():
        return Variable(torch.FloatTensor(data).cuda())
    else:
        return Variable(torch.FloatTensor(data))

def create_long_value(data):

    if torch.cuda.is_available():
        return Variable(torch.LongTensor(data).cuda())
    else:
        return Variable(torch.LongTensor(data))

def get_complex_model_data(fold, topk):
    path = '../data_temp/ec_ml_nv_rich_data_fold_'+str(fold)+'_topk_'+str(topk)+'.json'
    return fu_load_json(path)

def get_transform_model_data(fold):
    data_p1 = fu_load_json('../data_net/ec_p4_fold' + str(fold) + '.json')
    return data_p1



def get_user_ablity_index(fold):

    data_p2_index = {}
    data_p3_index = {}

    data_p2_index[MISSING_FLAG] = [MISSING_FLAG]
    data_p3_index[MISSING_FLAG] = [MISSING_FLAG]

    data_p2 = fu_load_json('../data_net/ec_p2_fold' + str(fold) + '.json')
    data_p3 = fu_load_json('../data_net/ec_p3_fold' + str(fold) + '.json')

    for d in data_p2:
        dd = data_p2[d]
        if len(dd) > LOG_RETAIN_LENGTH:
            dd = dd[len(dd) - LOG_RETAIN_LENGTH:]
        data_p2_index[int(d)] = dd

    for d in data_p3:
        dd = data_p3[d]
        if len(dd) > LOG_RETAIN_LENGTH:
            dd = dd[len(dd) - LOG_RETAIN_LENGTH:]
        data_p3_index[int(d)] = dd

    for i in range(0, get_user_num()+1):
        if i not in data_p2_index:
            data_p2_index[i] = [-1]
        if i not in data_p3_index:
            data_p3_index[i] = [-1]

    return data_p2_index, data_p3_index

def get_feature_index():

    original_feature_index = fu_load_json('../data_disposal/ec_3.4_last_train_data.json')
    original_feature_index[MISSING_FLAG] = [0 for i in range(0, FEATHER_SIZE)]
    feature_index = {}

    for key in original_feature_index:
        feature_index[int(key)] = original_feature_index[key]

    for key in range(100000, 150001):
        if key not in feature_index:
            feature_index[key] = [0 for i in range(0, FEATHER_SIZE)]

    # for key in original_feature_index:
    #     feature_index[int(key)] = [int(key), int(key)]

    return feature_index

def get_user_num():
    meta_data = fu_load_json('../data_base/ec_meta.json')
    return meta_data['user_num']