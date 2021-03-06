import jieba

stop_words = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours',
            'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their',
            'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once',
            'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you',
            'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will',
            'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be',
            'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself',
            'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both',
            'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn',
            'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about',
            'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn',
            'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']

stop_symbol = ['', ' ', '\n', '~', '!', '@', '#', '$', '%', '%', '^', '&', '*', '(', ')', '-', '+', '_', '=', '{', '}',
               '[', ']', '|', '\\', ':', ';', '"', '\'', '<', '>', ',', '.', '?', '/']

stop_num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def wu_cut_text(text):
    return jieba.cut(text, cut_all=True)

def wu_clean_words(words):
    result = []
    for w in words:
        if w != '\t' and w not in stop_words and w not in stop_symbol and w not in stop_num:
            result.append(w.lower())
    return result

def wu_cut_text_and_clean_words(text):
    text = str(text)
    words = wu_cut_text(text)
    return wu_clean_words(words)

def wu_get_non_repeat_list(data):
    return list(set(data))