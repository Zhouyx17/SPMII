import pandas as pd
import json
import numpy as np
id_label_dict = json.load(open('id_label_dict.json','r'))
label_id_dict = {v:k for k,v in id_label_dict.items()}
def get_labels(file_name):
    labels = []
    with open(file_name,'r') as f:
        for i,line in enumerate(f):
            if i > 0:
                entry = line.strip().split(',')
                label = label_id_dict[entry[1]]
                labels.append(label)
    return labels
all_labels = []
for i in range(5):
    all_labels.append(get_labels('output_{}.csv'.format(i)))
final_label = []
ttl = 0
all_agree = 0
for l1,l2,l3,l4,l5 in zip(all_labels[0],all_labels[1],all_labels[2],all_labels[3],all_labels[4]):
    cnt = np.zeros((11,1))
    cnt[int(l1)] += 1
    cnt[int(l2)] += 1
    cnt[int(l3)] += 1
    cnt[int(l4)] += 1
    cnt[int(l5)] += 1
    res = np.argmax(cnt)
    if np.max(cnt) == 5:
        all_agree += 1
    ttl += 1
    final_label.append(str(res))
df = pd.read_csv('output_0.csv')
final_label_name = [id_label_dict[ids] for ids in final_label]
df['label'] = final_label_name
df.to_csv('output_ensembled.csv',index=False)
print(all_agree/ttl)