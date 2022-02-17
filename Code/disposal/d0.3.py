# 把 ec des_title 由 xls 转 json

import pandas as pd
import numpy as np
import math

from common.word_util import wu_cut_text_and_clean_words
from common.file_util import fu_save_json
from common.num_util import nu_is_number
from common.file_util import fu_load_json


def run_es():

    fixed_ids = fu_load_json('../data_base/ec_fixed_ids.json')
    es_title_desc = []

    df = pd.read_excel(io='../data_original/eclipse_desc_title_utf8.xls')  ##默认读取sheet = 0的
    for index, row in df.iterrows():

        if not nu_is_number(row['bug_id']):
            continue
        if math.isnan(float(row['bug_id'])):
            continue

        bug_id = int(row['bug_id'])
        if bug_id not in fixed_ids:
            continue


        print(row['bug_id'])
        bug_id = row['bug_id']
        title_words = wu_cut_text_and_clean_words(row['title'])
        desc_words = wu_cut_text_and_clean_words(row['desc'])

        d = {'bug_id':bug_id, 'title_words':title_words, 'desc_words':desc_words}
        es_title_desc.append(d)
        # print('-----------')


    fu_save_json(es_title_desc, '../data_base/ec_title_desc.json')


if __name__ == '__main__':
    run_es()


