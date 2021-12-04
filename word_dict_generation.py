import jieba
import json

data = []
num_folds = 5

for i in range(num_folds):
    data += json.load(open('train_{}.json'.format(i),'r'))
def tokenizer(x):
    res = [w for w in jieba.cut(x)]
    return res
def dict_generation(data):
    stop_words = []
    with open('stopwords.txt', 'r') as f:
        for line in f:
            stop_words.append(line.strip())
    word_dict = {'pad':0,'unk':1}
    cnt = 2
    max_len = 0            
    for line in data:
        query = line[2]
        tokens = tokenizer(query)
        if len(tokens) > max_len:
            max_len = len(tokens)
        for token in tokens:
            if token not in word_dict and token not in stop_words:
                word_dict[token] = cnt
                cnt += 1
    print(max_len)
    return word_dict
word_dict = dict_generation(data)
json.dump(word_dict, open('word_dict.json','w'))
