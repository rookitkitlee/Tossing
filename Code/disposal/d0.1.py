# 把 ec_attr_num csv数据转成 json数据


from common.file_util import fu_load_csv
from common.file_util import fu_load_json
from common.file_util import fu_save_json

data = fu_load_csv('../data_original/ec_attr_num.csv')
fixed_ids = fu_load_json('../data_base/ec_fixed_ids.json')


jd = []
for d in data[1:]:

    bug_id = int(d[0])
    if bug_id not in fixed_ids:
        continue

    item = {'bug_id':d[0], 'product':d[1], 'component':d[2], 'hardware':d[3], 'label':d[4]}
    jd.append(item)

print(len(jd))
fu_save_json(jd, '../data_base/ec_attr_num.json')