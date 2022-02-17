
from common.file_util import fu_load_json
from common.file_util import fu_save_json
from common.file_util import fu_save_csv

origin_data = fu_load_json('../data_base/ec_attr_num.json')
tossing_data = fu_load_json('../data_base/ec_tossing_path.json')
meta_data = fu_load_json('../data_base/ec_meta.json')

#fold_index = [104833, 110111, 114647, 118937, 123371, 127657, 132172, 136635, 141062, 145326]
fold_index = meta_data['fold_ids']
fold_index = [int(d) for d in fold_index]
fold_start_time = []


tossing_index = {}
for d in tossing_data:
    #print(d)
    tossing_index[int(d['bug_id'])] = d

for fi in fold_index:
    st = tossing_index[fi]['open_time']
    fold_start_time.append(st)

origin_index = {}
for d in origin_data:
    origin_index[int(d['bug_id'])] = d

fid2tossing = {}
ids2tossing = {}
userLastAvilabel = {}


class Tossing_User:
    def __init__(self, fid, tid, p, c, pc, num, pr, av):
        self.fid = fid
        self.tid = tid
        self.p = p
        self.c = c
        self.pc = pc
        self.num = num
        self.pr = pr
        self.av = av
    def show(self):
        print(self.fid, self.tid, self.p, self.c, self.pc, self.num, self.pr, self.av)

def create_tossing(target_fold):

    # fid2tossing = {}
    # ids2tossing = {}

    # print(fid2tossing)
    # print(ids2tossing)

    max_bug_id = fold_index[target_fold - 1]
    # min_bug_id = max_bug_id - len(origin_data) / 11

    max_close_time = fold_start_time[target_fold - 1]

    for d in origin_data:
        bug_id = int(d['bug_id'])
        p = int(d['product'])
        c = int(d['component'])

        # if bug_id < min_bug_id:
        #     continue
        if bug_id >= max_bug_id:
            break

        if bug_id not in tossing_index:
            continue

        if tossing_index[bug_id]['close_time'] >= max_close_time:
            continue

        if bug_id not in tossing_index:
            continue

        tossing_path = tossing_index[bug_id]['tossing_path']
        if len(tossing_path) <= 1:
            continue
        from_tossing = tossing_path[:len(tossing_path)-1]
        to_tossing = tossing_path[len(tossing_path)-1]


        tossing_time = tossing_index[bug_id]['tossing_time']
        for k in range(0, len(tossing_time)):
            uid = tossing_path[k]
            utime = tossing_time[k]

            if uid not in userLastAvilabel:
                userLastAvilabel[uid] = utime
            if userLastAvilabel[uid] < utime:
                userLastAvilabel[uid] = utime


        # print(tossing_path, from_tossing, to_tossing)
        for ft in from_tossing:
            key = str(ft)+"_"+str(to_tossing)
            # print(key)

            pc = str(p) + "_" + str(c)

            if key not in ids2tossing:
                tu = Tossing_User(ft, to_tossing, [p], [c], [pc], 1, 0, True)
                ids2tossing[key] = tu
            else:
                tu = ids2tossing[key]
                tu.num = tu.num + 1

                ps = tu.p
                if p not in ps:
                    ps.append(p)
                tu.p = ps

                cs = tu.c
                if c not in cs:
                    cs.append(c)
                tu.c = cs

                pcs = tu.pc
                if pc not in pcs:
                    pcs.append(pc)
                tu.pc = pcs

                ids2tossing[key] = tu

            tu = ids2tossing[key]
            if ft not in fid2tossing:
                fid2tossing[int(ft)] = [tu]
            else:
                arr = fid2tossing[int(ft)]
                # arr.append(tu)
                # fid2tossing[int(ft)] = arr
                flag = True
                for a in arr:
                    if int(a.tid) == int(to_tossing):
                        flag = False
                        break
                if flag:
                    arr.append(tu)
                    fid2tossing[int(ft)] = arr


def calc_tossing_pr():
    for key in fid2tossing:



        # print('------------------------')
        # print(key)
        total_num = 0
        if len(fid2tossing[key]) == 0:
            continue

        for tu in fid2tossing[key]:
            total_num = total_num + tu.num

            # tu.show()

        # print(total_num)

        for tu in fid2tossing[key]:

            tu.pr = tu.num / total_num
            if tu.pr < 0.15:
                tu.av = False
            if tu.num < 10:
                tu.av = False

            # print(key, total_num)
            # tu.show()
            # if tu.av:
            #     tu.show()
        # for tu in fid2tossing[key]:
        #     tu.show()


def load_ml_data(fold, topk):
    path = '../data_temp/ec_ml_nv_fold_'+str(fold)+'_topk_'+str(topk)+'.json'
    return fu_load_json(path)

def replace_score(bug_id, p, c, can):

    # if True:
    #     return -1

    # can.show()

    if not can.av:
        return -1

    # pc = str(p)+"_"+str(c)
    # if pc not in can.pc:
    #     return -1

    # if p not in can.p and c not in can.c:
    #     return -1

    score = can.pr

    # pc = str(p)+"_"+str(c)
    # if pc in can.pc:
    #     score = score + 2

    if p in can.p:
        score = score + 1
    if c in can.c:
        score = score + 1

    if int(bug_id) not in tossing_data:
        return score
    if int(can.tid) not in userLastAvilabel:
        return score

    bug_open_time = tossing_data[int(bug_id)]['open_time']
    user_availabel = userLastAvilabel[int(can.tid)]

    if bug_open_time - user_availabel < 30 * 24 * 60 * 60:
        score = score + 1

    return score

def replace_predict(bug_id, p, c, fid):

    if fid not in fid2tossing:
        return int(-1)

    cans = fid2tossing[fid]
    max_score = -1
    max_target = -1

    for can in cans:

        score = replace_score(bug_id, p, c, can)
        if score > max_score:
            max_score = score
            max_target = int(can.tid)
    return max_target

def adjust_predict(bug_id, data, topk):
    result = []
    for i in range(0, int(topk/2)):
        t = data[i]
        result.append(t)

        bug = origin_index[int(bug_id)]
        p = int(bug['product'])
        c = int(bug['component'])

        rp = replace_predict(bug_id, p, c, t)
        if int(rp) != -1 and rp not in result:
            result.append(rp)

    # for d in data:
    #     if len(result) >= len(data):
    #         break
    #     if d not in result:
    #         result.append(d)
    return result
    # if topk % 2 == 1:
    #     result.append(data[int(topk/2)])



def run(fold, topk):

    if topk <= 1:
        return

    fid2tossing.clear()
    ids2tossing.clear()
    userLastAvilabel.clear()

    create_tossing(fold)
    calc_tossing_pr()

    # print(fid2tossing)
    #


    ml_data = load_ml_data(fold, topk)

    num = 0
    for md in ml_data:
        ap = adjust_predict(int(md['bug_id']), md['predict'], topk)
        # print(md, ap)
        if md['label'] in ap:
            num = num + 1

    pr = num / len(ml_data)
    print('fold: ' + str(fold) + ' top: ' + str(topk) + ' accuracy: ' + str(pr))
    return pr



# run(9, 5)

rows = []

for i in range(1, 10):

    row_data = ['fold' + str(i)]
    for j in range(2, 6):
        pr = run(i, j)
        # print('fold: ' + str(i + 2) + ' top: ' + str(j) + ' accuracy: ' + str(pr))
        row_data.append(pr)

    break
    rows.append(row_data)

head = ['fold', 'top2', 'top3', 'top4', 'top5']
fu_save_csv(head, rows, '../data_result/ec_ml_nv_tossing_data_1.csv')