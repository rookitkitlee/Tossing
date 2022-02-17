import heapq
from common.file_util import *
from model.DataUtil import *


def run_fold(fold):

    print(fold)

    data = fu_load_json('../data_temp/ec_ml_nv_fold_'+str(fold)+'_topk_5.json')

    data_p2_index, data_p3_index = get_user_ablity_index(fold)

    for d in data:
        # print(d)
        original_predict = d['original_predict']
        original_category = d['original_category']
        label = d['label']

        top10_probabilities = []
        top10_categories = []

        original_predict_order = heapq.nlargest(10, range(len(original_predict)), original_predict.__getitem__)
        original_predict_order_sort_by_id = sorted(original_predict_order)
        for i in range(0, len(original_predict_order_sort_by_id)):
            index = original_predict_order_sort_by_id[i]
            top10_probabilities.append(original_predict[index])
            top10_categories.append(original_category[index])

        d['top10_probabilities'] = top10_probabilities
        d['top10_categories'] = top10_categories

        if label in top10_categories:
            d['top10_include'] = 1
            d['top10_label'] = top10_categories.index(label)
        else:
            d['top10_include'] = 0

        d['top10_data2'] = [data_p2_index[j] for j in top10_categories]
        d['top10_data3'] = [data_p3_index[j] for j in top10_categories]


    fu_save_json(data, '../data_temp/ec_ml_nv_rich_data_fold_'+str(fold)+'_topk_5.json')

for i in range(1, 101):
    run_fold(i)
