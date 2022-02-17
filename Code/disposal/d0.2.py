# 处理 tossing path

from common.file_util import fu_load_csv
from common.file_util import fu_save_json
from common.file_util import fu_load_json


# tossing_data = fu_load_csv('../data/ec_tossing_path_num_new.csv')
# tossing_json = []
#
# for d in tossing_data[2:]:
#     bug_id = d[0]
#     tossing = d[3].split('*')
#     item = {'bug_id':bug_id, 'tossing_path':tossing}
#     tossing_json.append(item)
#
#
# fu_save_json(tossing_json, '../data/ec_tossing_path.json')

fixed_ids = fu_load_json('../data_base/ec_fixed_ids.json')
tossing_data = fu_load_csv('../data_original/ec_dev_time.csv')
tossing_json = []

def str2arr(str):

    dd = str[1: len(str)-1]
    ds = dd.split(',')
    ds = [d.strip() for d in ds]
    ds = [d[1:len(d)-1] for d in ds]
    ds = [int(d) for d in ds]
    return ds


for d in tossing_data[1:]:

    bug_id = int(d[0])
    if d[2] == "[]":
        continue

    bug_id = int(d[0])
    if bug_id not in fixed_ids:
        continue

    tossing_path = str2arr(d[1])
    tossing_time = str2arr(d[2])
    open_time = tossing_time[0]
    close_time = tossing_time[len(tossing_time) - 1]
    item = {'bug_id': bug_id, 'tossing_path': tossing_path, 'tossing_time': tossing_time, 'open_time': open_time, 'close_time': close_time}
    tossing_json.append(item)

fu_save_json(tossing_json, '../data_base/ec_tossing_path.json')