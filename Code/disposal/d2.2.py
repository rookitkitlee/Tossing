# 计算 上一步  每个字符的熵  按照熵值 排列 取 title 前 100 个 desc 前 100
import math
from common.file_util import fu_load_csv
from common.file_util import fu_load_json
from common.file_util import fu_save_json

bug2label = {}
label2bug = {}
sample_ids = []

sample_data = fu_load_json('../data_disposal/ec_1.2_title_desc_sample_10000.json')
sample_ids = [d['bug_id'] for d in sample_data]

# bug_data = fu_load_csv('../data/ec_attr_num.csv')
# for d in bug_data[1:]:
#     if int(d[0]) in sample_ids:
#         bug2label[int(d[0])] = int(d[4])

bug_data = fu_load_json('../data_base/ec_attr_num.json')
for d in bug_data:
    if int(d['bug_id']) in sample_ids:
        bug2label[int(d['bug_id'])] = int(d['label'])


#print(bug2label)

for k in bug2label.items():
    id = k[0]
    label = k[1]
    if label not in label2bug:
        arr = []
        arr.append(id)
        label2bug[label] = arr
    else:
        arr = label2bug[label]
        arr.append(id)
        label2bug[label] = arr

#print(label2bug)
#print(len(label2bug))


# 建立id 与bug 的索引
id2bug = {}
sample_data = fu_load_json('../data_disposal/ec_1.2_title_desc_sample_10000.json')
for s in sample_data:
    id2bug[s['bug_id']] = s

# 计算熵的方法
def calc_entropy(data):
    total = sum(data)
    p_data = [d/total for d in data]
    entropy = 0
    for pd in p_data:
        if pd == 0.0:
            continue
        e = - pd * math.log2(pd)
        entropy = entropy + e
    return entropy

# 计算title 中字符的熵
title_word_entropy = {}
title_json = fu_load_json('../data_disposal/ec_2.1_title_words.json')
for word in title_json:

    c_nums = []
    for c in label2bug.items():
        bugs = c[1]
        c_num = 0
        for bug_id in bugs:
            title_words = id2bug[bug_id]['title_words']
            if word in title_words:
                c_num = c_num + 1
        c_nums.append(c_num)

    entropy = calc_entropy(c_nums)
    title_word_entropy[word] = entropy

title_entropy = sorted(title_word_entropy.items(), key=lambda x: x[1], reverse=False)
title_entrory_word = [twe[0] for twe in title_entropy[0:100]]
fu_save_json(title_entrory_word, '../data_disposal/ec_2.2_title_entrory_word.json')


# 计算desc 中字符的熵
desc_word_entropy = {}
desc_json = fu_load_json('../data_disposal/ec_2.1_desc_words.json')
for word in desc_json:

    c_nums = []
    for c in label2bug.items():
        bugs = c[1]
        c_num = 0
        for bug_id in bugs:
            desc_words = id2bug[bug_id]['desc_words']
            if word in desc_words:
                c_num = c_num + 1
        c_nums.append(c_num)

    entropy = calc_entropy(c_nums)
    desc_word_entropy[word] = entropy

desc_entropy = sorted(desc_word_entropy.items(), key=lambda x: x[1], reverse=False)
desc_entrory_word = [twe[0] for twe in desc_entropy[0:100]]
fu_save_json(desc_entrory_word, '../data_disposal/ec_2.2_desc_entrory_word.json')