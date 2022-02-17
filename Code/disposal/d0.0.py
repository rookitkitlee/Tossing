# 把 ec_attr_num csv数据转成 json数据


from common.file_util import fu_load_csv
from common.file_util import fu_save_json

data = fu_load_csv('../data_original/eclipse_status.csv')

ids = []

for d in data[1:]:

    try:
        bug_id = int(d[0])
        status = d[1]
        if status.find('FIXED') != -1:
            ids.append(bug_id)
    except:
        continue

fu_save_json(ids, '../data_base/ec_fixed_ids.json')
print(len(ids))