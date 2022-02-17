
from common.file_util import *

attr_index = {}

attr_data = fu_load_csv('../data_original/ec_attr_vector.txt')
for d in attr_data:

    vector = []
    bug_id = 0
    for i in range(0, len(d)):
        item = d[i]
        if i == 0:
            bug_id = item[0:6]
            vector.append(float(item[8:]))
        elif i == len(d)-1:
            v = float(item[1: len(item)-1])
            vector.append(v)
        else:
            vector.append(float(item[1:]))

    attr_index[bug_id] = vector


title_index = {}

title_data = fu_load_csv('../data_original/ec_title_vector.txt')
for d in title_data:

    vector = []
    bug_id = 0
    for i in range(0, len(d)):
        item = d[i]
        if i == 0:
            bug_id = item[0:6]
            vector.append(float(item[8:]))
        elif i == len(d)-1:
            v = float(item[1: len(item)-1])
            vector.append(v)
        else:
            vector.append(float(item[1:]))

    title_index[bug_id] = vector


feather_index = {}

for d in attr_index:
    attr = attr_index[d]
    title = title_index[d]
    attr.extend(title)

    feather_index[int(d)] = attr
    # print(d)

for i in range(100000, 150001):
    if i not in feather_index:
        print(i)
        feather_index[i] = [0 for i in range(0, 62)]

fu_save_json(feather_index, '../data_disposal/ec_3.4_last_train_data.json')