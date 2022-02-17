# 正负采样 构造训练集
import random
from common.file_util import fu_load_json
from common.file_util import fu_save_json


tossing_data = fu_load_json('../data_base/ec_tossing_path.json')

tossing_index = {}
for d in tossing_data:
    tossing_index[int(d['bug_id'])] = d

meta_data = fu_load_json('../data_base/ec_meta.json')
folds_meta = meta_data['folds']
user_num = meta_data['user_num']

for fold_index in range(1, 101):

    max_bug_id = folds_meta[fold_index]

    # 不能完成的用户
    p2_datas = {}
    # 最终修复的用户
    p3_datas = {}


    for d in tossing_data:

        if int(d['bug_id']) <= max_bug_id:

            bug_id = int(d['bug_id'])
            tossing_path = d['tossing_path']
            f_tossing_path = tossing_path[0: len(tossing_path) - 1]
            end_tossing = tossing_path[len(tossing_path) - 1]


            for u in f_tossing_path:
                u = int(u)
                if u not in p2_datas:
                    p2_datas[u] = [bug_id]
                else:
                    dd = p2_datas[u]
                    dd.append(bug_id)
                    p2_datas[u] = dd


            if end_tossing not in p3_datas:
                p3_datas[end_tossing] = [bug_id]
            else:
                dd = p3_datas[end_tossing]
                dd.append(bug_id)
                p3_datas[end_tossing] = dd

    fu_save_json(p2_datas, '../data_net/ec_p2_fold' + str(fold_index) + '.json')
    fu_save_json(p3_datas, '../data_net/ec_p3_fold' + str(fold_index) + '.json')