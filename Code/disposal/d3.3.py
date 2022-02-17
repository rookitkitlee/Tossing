# 整合 attr title_desc 数据


from common.file_util import fu_load_json
from common.file_util import fu_save_json

title_desc_data = fu_load_json('../data_disposal/ec_3.1_tfidf_title_desc.json')
attr_data = fu_load_json('../data_disposal/ec_3.2_ec_onehot_attr.json')


id2td = {}

for d in title_desc_data:
    id2td[int(d['bug_id'])] = d

ec_last_train_data = []
for d in attr_data:
    td = id2td[int(d['bug_id'])]
    item = {'bug_id': d['bug_id'], 'attr': d['attr'], 'title_words':td['title_words'], 'desc_words':td['desc_words'], 'label': d['label']}
    ec_last_train_data.append(item)


print(len(ec_last_train_data))
fu_save_json(ec_last_train_data, '../data_disposal/ec_3.3_last_train_data.json')