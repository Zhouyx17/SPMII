# SPMII
The code repo of SPMII project
## requirements
See requirements.txt. Some packages are not necessary.
## Data Preprocessing
Run data_preprocessing.py
## Models
### Chinese BERT base
#### Train
Run run_bert_chinese.py for train and evaluate model by k-folds.
More efficient method is needed for leveraging the full train set.
#### Test
Run run_bert_chinese_test.py
### TextCNN
#### Train
Run word_dict_generation.py first to generate the token to id dictionary, then run run_textcnn.py to train the textCNN model. Acc ~70%
