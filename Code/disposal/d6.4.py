#正负采样
import heapq
import random
from common.file_util import *
from model.DataUtil import *



tossing_data = fu_load_json('../data_base/ec_tossing_path.json')
tossing_index = {}
for d in tossing_data:
    tossing_index[int(d['bug_id'])] = d


def run_fold(fold):

    print(fold)


    data = fu_load_json('../data_temp/ec_ml_nv_rich_data_fold_'+str(fold)+'_topk_5.json')
    data_p2_index, data_p3_index = get_user_ablity_index(fold)
    sample_datas = []

    for d in data:


        bug_id = d['bug_id']
        top10_categories = d['top10_categories']
        top10_probabilities = d['top10_probabilities']
        label = d['label']



        f_categories = []
        f_probabilities = []
        for i in range(0, len(top10_categories)):
            c = top10_categories[i]
            p = top10_probabilities[i]
            if c != label:
                f_categories.append(c)
                f_probabilities.append(p)


        # 正采样
        for _ in range(0, 10):
            for i in range(0, len(f_categories)):
                ft = f_categories[i]
                pt = f_probabilities[i]
                item = {'bug_id': int(d['bug_id']), 'fp': ft, 'tp': label, 'label': 1, 'weight':[0, pt]}
                sample_datas.append(item)

        # 负采样1
        for i in range(0, len(f_categories)):
            for j in range(0, len(f_categories)):
                ft = f_categories[i]
                pt = f_probabilities[i]
                et = f_categories[j]
                item = {'bug_id': int(d['bug_id']), 'fp': ft, 'tp': et, 'label': 0, 'weight':[pt, 0]}
                sample_datas.append(item)


        # # 负采样2
        # for i in range(0, len(f_categories)):
        #     ft = f_categories[i]
        #     pt = f_probabilities[i]
        #     et = 0
        #     while True:
        #         rt = random.randint(0, USER_NUM)
        #         if rt not in f_categories:
        #             et = rt
        #             break
        #     item = {'bug_id': int(d['bug_id']), 'fp': ft, 'tp': et, 'label': 0, 'weight':[pt, 0]}
        #     sample_datas.append(item)


        # # 正采样
        # for _ in range(0, 10):
        #     for ft in candidates:
        #         item = {'bug_id': int(d['bug_id']), 'fp': ft, 'tp': label, 'label': 1}
        #         sample_datas.append(item)
        #
        # # 负采样1
        # for ft in candidates:
        #     for et in f_candidates:
        #         item = {'bug_id': int(d['bug_id']), 'fp': ft, 'tp': et, 'label': 0}
        #         sample_datas.append(item)
        #
        #
        # # 负采样2
        # for ft in candidates:
        #     et = 0
        #     while True:
        #         rt = random.randint(0, USER_NUM)
        #         if rt not in candidates:
        #             et = rt
        #             break
        #     item = {'bug_id': int(d['bug_id']), 'fp': ft, 'tp': et, 'label': 0}
        #     sample_datas.append(item)



        for s in sample_datas:
            fp = s['fp']
            tp = s['tp']
            s['fp_data'] = data_p2_index[fp]
            s['tp_data'] = data_p3_index[tp]
            # print(s)

    fu_save_json(sample_datas, '../data_net/ec_p4_fold' + str(fold) + '.json')


for i in range(1, 101):
    run_fold(i)

