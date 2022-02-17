# one hot 编码 attr

from sklearn.preprocessing import OneHotEncoder
from common.file_util import fu_load_csv
from common.file_util import fu_save_json
from common.file_util import fu_load_json

# origin_data = fu_load_csv('../data_base/ec_attr_num.csv')
# fixed_ids = fu_load_json('../data_base/ec_fixed_ids.json')

base_data = fu_load_json('../data_base/ec_attr_num.json')


pcdarr = []
bug_ids = []
bug_labels = []

# for d in origin_data[1:]:
#
#     arr = [d[1], d[2], d[3]]
#     #arr = [d[1], d[2]]
#     pcdarr.append(arr)
#     bug_ids.append(d[0])
#     bug_labels.append(d[4])

for d in base_data:

    arr = [d['product'], d['component'], d['hardware']]
    #arr = [d[1], d[2]]
    pcdarr.append(arr)
    bug_ids.append(d['bug_id'])
    bug_labels.append(d['label'])

enc = OneHotEncoder()
enc.fit(pcdarr)

encoder_pcd = enc.transform(pcdarr).toarray()

ec_onehot_attr = []
for i in range(0, len(bug_ids)):
    item = {'bug_id':bug_ids[i], 'attr':encoder_pcd[i].tolist(), 'label':bug_labels[i]}
    ec_onehot_attr.append(item)

fu_save_json(ec_onehot_attr, '../data_disposal/ec_3.2_ec_onehot_attr.json')