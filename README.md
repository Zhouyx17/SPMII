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
## Ensembling
Two ensembling methods are provided: voting and averaging.
### Voting
Run vote_ensembling.py (make sure that you get all the output csv files that you want to ensemble.)
### Averaging
Run soft_ensembling.py (make sure that you get all the output pkl files that you want to ensemble.)
*** the pkl file saves the predicted probability of each examples in the test set. The run_bert_chinese_test.py has been modified to generate this file. 
