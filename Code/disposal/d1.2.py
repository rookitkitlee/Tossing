# 从中 抽取 前 4000个样本 作为采样数据

from common.file_util import fu_load_json
from common.file_util import fu_save_json

def sample_data():
    data = fu_load_json('../data_base/ec_title_desc.json')
    sdata = []

    for d in data:
        bug_id = int(d['bug_id'])

        if bug_id <= 110000:
            sdata.append(d)

    print(len(sdata))
    fu_save_json(sdata, '../data_disposal/ec_1.2_title_desc_sample_10000.json')

sample_data()