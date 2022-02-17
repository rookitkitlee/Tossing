# 把 计算meta 数据

import pandas as pd
import numpy as np
import math

from common.word_util import wu_cut_text_and_clean_words
from common.file_util import fu_save_json
from common.num_util import nu_is_number
from common.file_util import fu_load_json

data = {}
def run():
    # datas = fu_load_json('../data_disposal/ec_3.3_last_train_data.json')
    # sample_length = len(datas)
    # fold_width = int(sample_length / 11) + 1
    # folds = []
    # fold_ids = []
    # for i in range(0, 10):
    #     f1 = fold_width * (i + 1)
    #     folds.append(f1)
    #     fold_ids.append(datas[f1]['bug_id'])
    # data['sample_length'] = sample_length
    # data['fold_width'] = fold_width
    # data['folds'] = folds
    # data['fold_ids'] = fold_ids

    folds = [100000]
    for i in range(100):
        step = 110000 + i * 400
        folds.append(step)

    data['folds'] = folds

    # 获取最大的 用户id


    max_user = 0
    tossing_data = fu_load_json('../data_base/ec_tossing_path.json')
    for d in tossing_data:
        for p in d['tossing_path']:
            if p > max_user:
                max_user = p

    data['user_num'] = max_user + 1
    fu_save_json(data, '../data_base/ec_meta.json')

if __name__ == '__main__':
    run()


