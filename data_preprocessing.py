from numpy.core.fromnumeric import transpose
import pandas as pd 
import random
import json
num_folds = 5
train_df = pd.read_csv('train.csv', delimiter=',')
data = []
labels = set()
for i, row in train_df.iterrows():
    labels.add(row['label'])
    data.append([i, row['label'],row['query']])
labels = list(labels)
id_to_labels = {k:v for k,v in enumerate(labels)}
json.dump(id_to_labels, open('id_label_dict.json','w'))
label_to_ids = {k:int(v) for v, k in id_to_labels.items()}
for i, row in enumerate(data):
    data[i] = [row[0], label_to_ids[row[1]], row[2]]
random.shuffle(data)
for i in range(num_folds-1):
    json.dump(data[i*(len(data)//5):(i+1)*(len(data)//5)] ,open('train_{}.json'.format(i),'w'))
json.dump(data[(num_folds-1)*(len(data)//5):] ,open('train_4.json','w'))

test_df = pd.read_csv('test.csv', delimiter=',')
data = []
for i, row in test_df.iterrows():
    data.append([i, 0, row['query']])
json.dump(data, open('test.json','w'))

