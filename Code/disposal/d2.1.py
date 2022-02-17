# 从采样的数据当中 查出都有哪些字符 而且每个字符出现次数不能低于 title 1%  desc 1%

from common.file_util import fu_save_json
from common.file_util import fu_load_json
from common.word_util import wu_get_non_repeat_list


def extract_ec_title_words():

    ec_title_words = {}
    ec_filter_title_words = []
    data = fu_load_json('../data_disposal/ec_1.2_title_desc_sample_10000.json')

    filter_length = len(data) / 100
    for d in data:
        for w in wu_get_non_repeat_list(d['title_words']):
            if w not in ec_title_words:
                ec_title_words[w] = 1
            else:
                ec_title_words[w] = ec_title_words[w] + 1

    ec_title_words = sorted(ec_title_words.items(), key=lambda x: x[1], reverse=True)
    #print(ec_title_words)
    print(len(ec_title_words))

    for k in ec_title_words:
        if k[1] >= filter_length:
            ec_filter_title_words.append(k[0])

    print(ec_filter_title_words)
    print(len(ec_filter_title_words))
    fu_save_json(ec_filter_title_words, '../data_disposal/ec_2.1_title_words.json')

def extract_ec_desc_words():

    ec_desc_words = {}
    ec_filter_desc_words = []
    data = fu_load_json('../data_disposal/ec_1.2_title_desc_sample_10000.json')

    filter_length = len(data) / 100
    for d in data:
        for w in wu_get_non_repeat_list(d['desc_words']):
            if w not in ec_desc_words:
                ec_desc_words[w] = 1
            else:
                ec_desc_words[w] = ec_desc_words[w] + 1

    ec_desc_words = sorted(ec_desc_words.items(), key=lambda x: x[1], reverse=True)
    #print(ec_desc_words)
    print(len(ec_desc_words))

    for k in ec_desc_words:
        if k[1] >= filter_length:
            ec_filter_desc_words.append(k[0])

    print(ec_filter_desc_words)
    print(len(ec_filter_desc_words))
    fu_save_json(ec_filter_desc_words, '../data_disposal/ec_2.1_desc_words.json')

extract_ec_title_words()
extract_ec_desc_words()