import pandas as pd
import json
import numpy as np
import pickle
data = []
id_label_dict = json.load(open('id_label_dict.json','r'))
label_id_dict = {v:k for k,v in id_label_dict.items()}
for i in range(5):
    data.append(pickle.load(open('output_{}.pkl'.format(i),'rb')))
final_label = []
for i in range(data[0].shape[0]):
    tmp = 0
    for j,one in enumerate(data):
        if j == 0:
            tmp = one[i,:]
        else:
            tmp += one[i,:]
    tmp /= len(data)
    final_label.append(str(int(np.argmax(tmp))))
df = pd.read_csv('output_0.csv')
final_label_name = [id_label_dict[ids] for ids in final_label]
df['label'] = final_label_name
df.to_csv('output_soft_ensembled.csv',index=False)
