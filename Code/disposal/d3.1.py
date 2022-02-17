# 根据上一步的词的熵  整理全部数据的 title desc

from common.file_util import fu_load_json
from common.file_util import fu_save_json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


title_entropy_word = fu_load_json('../data_disposal/ec_2.2_title_entrory_word.json')
desc_entropy_word = fu_load_json('../data_disposal/ec_2.2_desc_entrory_word.json')
sample_data = fu_load_json('../data_disposal/ec_1.2_title_desc_sample_10000.json')

def arr2str(data):
    result = ''
    for d in data:
        if result == '':
            result = result + d
        else:
            result = result + ' ' + d
    return result

sample_entroy_data = []
for d in sample_data:
    bug_id = d['bug_id']
    title_words = d['title_words']
    desc_words = d['desc_words']
    bug_entropy_title_words = []
    bug_entropy_desc_words = []
    for w in title_words:
        if w in title_entropy_word:
            bug_entropy_title_words.append(w)
    for w in desc_words:
        if w in desc_entropy_word:
            bug_entropy_desc_words.append(w)
    item = {'bug_id':bug_id, 'title_words':bug_entropy_title_words, 'desc_words':bug_entropy_desc_words}
    sample_entroy_data.append(item)

# for d in sample_entroy_data:
#     print(d)

title_train_data = [arr2str(d['title_words']) for d in sample_entroy_data]
title_count_worker = CountVectorizer()
title_count_matrix = title_count_worker.fit_transform(title_train_data)
title_tfidf_worker = TfidfTransformer(use_idf=False)
title_tfidf_matrix = title_tfidf_worker.fit_transform(title_count_matrix)
# print(title_vector_matrix.toarray()[20])


desc_train_data = [arr2str(d['desc_words']) for d in sample_entroy_data]
desc_count_worker = CountVectorizer()
desc_count_matrix = desc_count_worker.fit_transform(desc_train_data)
desc_tfidf_worker = TfidfTransformer(use_idf=False)
desc_tfidf_matrix = desc_tfidf_worker.fit_transform(desc_count_matrix)
# print(desc_vector_matrix.toarray()[20])


#　把所有样本的　title 和 desc 编程 tfidf
all_data = fu_load_json('../data_base/ec_title_desc.json')
all_entropy_data = []
for d in all_data:
    bug_id = d['bug_id']
    title_words = d['title_words']
    desc_words = d['desc_words']
    bug_entropy_title_words = []
    bug_entropy_desc_words = []
    for w in title_words:
        if w in title_entropy_word:
            bug_entropy_title_words.append(w)
    for w in desc_words:
        if w in desc_entropy_word:
            bug_entropy_desc_words.append(w)
    item = {'bug_id':bug_id, 'title_words':bug_entropy_title_words, 'desc_words':bug_entropy_desc_words}
    all_entropy_data.append(item)



# title_all_data = [arr2str(d['title_words']) for d in all_entropy_data]
# print(title_all_data)


all_title_data = [arr2str(d['title_words']) for d in all_entropy_data]
all_title_count_matrix = title_count_worker.transform(all_title_data)
all_title_tfidf_matrix = title_tfidf_worker.transform(all_title_count_matrix)
all_title_tfidf_matrix = all_title_tfidf_matrix.toarray()

all_desc_data = [arr2str(d['desc_words']) for d in all_entropy_data]
all_desc_count_matrix = desc_count_worker.transform(all_desc_data)
all_desc_tfidf_matrix = desc_tfidf_worker.transform(all_desc_count_matrix)
all_desc_tfidf_matrix = all_desc_tfidf_matrix.toarray()

all_id_data = [d['bug_id'] for d in all_entropy_data]

all_tfidf_data = []
for i in range(0, len(all_id_data)):
    item = {'bug_id':all_id_data[i], 'title_words':all_title_tfidf_matrix[i].tolist(), 'desc_words':all_desc_tfidf_matrix[i].tolist()}
    print(item)
    all_tfidf_data.append(item)

fu_save_json(all_tfidf_data, '../data_disposal/ec_3.1_tfidf_title_desc.json')