import argparse
import json
import logging
import os

import numpy as np
import torch
from torch._C import dtype
from torch.optim import lr_scheduler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
import jieba
from models import textCNN

logger = logging.getLogger(__name__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # del
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def tokenizer(x):
    res = [w for w in jieba.cut(x)]
    return res

def tokenize_and_convert_to_ids(x, word_dict):
    tokens = tokenizer(x)
    token_ids = []
    for token in tokens:
        if token not in word_dict:
            token_ids.append(word_dict['unk'])
        else:
            token_ids.append(word_dict[token])
    return token_ids

def process_data(data, word_dict, max_seq=50, dataset='train',batch_size=32):
    input_ids_list, label_list = [],[]
    
    for line in data:
        tokens_a = tokenize_and_convert_to_ids(line[2], word_dict)
        if len(tokens_a) > max_seq:
            tokens_a = tokens_a[:max_seq]
        elif len(tokens_a) < max_seq:
            tokens_a += [word_dict['pad'] for i in range(max_seq-len(tokens_a))]
        input_ids_list.append(tokens_a)
        label_list.append(line[1])
    all_input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    all_label_ids = torch.tensor(label_list, dtype=torch.long)

    all_data = TensorDataset(all_input_ids, all_label_ids)
    sampler = RandomSampler(all_data) if dataset == "train" else SequentialSampler(all_data)
    dataloader = DataLoader(all_data, sampler=sampler, batch_size=batch_size)
    return dataloader
def train(device, train_data, dev_data, model,word_dict, batchsize=32, max_seq=50, num_epoch=20, test_id=0):
    
    train_dataloader = process_data(train_data, word_dict, max_seq=max_seq, dataset='train',batch_size=batchsize)
    
    model.train()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    global_step = 0
    best_acc = 0.0
    for ep in trange(num_epoch, desc="Training"):
        for step, batch in tqdm(enumerate(train_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, label_ids = batch
            logits = model(input_ids)
            loss = loss_fct(logits, label_ids)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        model.eval()
        dev_acc = eval(device, dev_data, model, word_dict, batchsize, max_seq)
        logger.info("Epoch {} acc {} step {}".format(ep, dev_acc, global_step))
        if dev_acc > best_acc:
            best_acc = dev_acc
            save_model(model, output_dir='saved_model_{}'.format(test_id))  # save model
        model.train()
    model.eval()
            
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)            
            
def save_model(model_to_save, output_dir):
    save_model_dir = os.path.join(output_dir, 'saved_model')
    mkdir(save_model_dir)
    output_model_file = os.path.join(save_model_dir, 'pytorch_model.bin')

    torch.save(model_to_save.state_dict(), output_model_file, _use_new_zipfile_serialization=False)
         
    
def eval_1(preds, labels):
    acc = np.sum((preds == labels).astype(np.int32))/len(preds)
    return acc

def eval(device, dev_data, model, word_dict, batchsize=32, max_seq=50):
    test_dataloader = process_data(dev_data, word_dict, max_seq=max_seq, dataset='train',batch_size=batchsize)

    model.eval()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')

    eval_loss = 0
    num_steps = 0
    preds = []
    labels = []
    for step, batch in enumerate(tqdm(test_dataloader, desc='test')):
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            input_ids, label_ids = batch
            logits = model(input_ids)
            loss = loss_fct(logits, label_ids)
            eval_loss += loss.mean().item()
            num_steps += 1
            preds += torch.argmax(logits, dim=1).detach().tolist()
            labels += label_ids.detach().tolist()
    acc = eval_1(np.asarray(preds), np.asarray(labels))
    return acc

device = 'cuda'
test_id = 0
num_folds = 5
batch_size = 100
np.random.seed(24)
torch.manual_seed(24)
max_seq = 50
num_epoch = 100
warmup = 0.3
saved_file = 'run_chinese_bert_split_{}'.format(test_id)
mkdir('saved_model_{}_textCNN'.format(test_id))
train_data = []
dev_data = []
logging.basicConfig(format='%(message)s', datefmt='%m/%d/%Y %H:%M', level=logging.INFO,
                        handlers=[logging.FileHandler("{0}/{1}.log".format('saved_model_{}'.format(test_id), 'output')),
                                  logging.StreamHandler()])
for i in range(num_folds):
    if i == test_id:
        dev_data = json.load(open('train_{}.json'.format(i),'r'))
    else:
        train_data += json.load(open('train_{}.json'.format(i),'r'))
word_dict = json.load(open('word_dict.json','r'))
model = textCNN(len(word_dict), embed_dim=300, class_num=11, kernel_num=10, kernel_size=[3,4,5,6,7]).to(device)
train(device, train_data, dev_data, model, word_dict, batchsize=batch_size, max_seq=max_seq, num_epoch=num_epoch, test_id=test_id)

