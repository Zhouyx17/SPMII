import argparse
import json
import logging
import os

import numpy as np
import torch
from pytorch_pretrained_bert.file_utils import CONFIG_NAME, WEIGHTS_NAME
from torch._C import dtype
from torch.optim import lr_scheduler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
from transformers import BertForSequenceClassification, BertTokenizer
import transformers
from transformers import AdamW

logger = logging.getLogger(__name__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # del
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def process_data(data, tokenizer, max_seq=50, dataset='train',batch_size=32):
    input_mask_list, input_ids_list, label_list = [],[],[]
    for line in data:
        tokens_a = tokenizer.tokenize(line[2])

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        len_input = len(input_ids)
        input_ids += [0] * (max_seq - len_input)
        input_mask += [0] * (max_seq - len_input)
        input_ids_list.append(input_ids)
        input_mask_list.append(input_mask)
        label_list.append(line[1])
    all_input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    all_input_mask = torch.tensor(input_mask_list, dtype=torch.long)
    all_label_ids = torch.tensor(label_list, dtype=torch.long)

    all_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    sampler = RandomSampler(all_data) if dataset == "train" else SequentialSampler(all_data)
    dataloader = DataLoader(all_data, sampler=sampler, batch_size=batch_size)
    return dataloader
def train(device, train_data, dev_data, model, tokenizer, batchsize=32, max_seq=50, num_epoch=20, test_id=0,warmup=0.3):
    
    train_dataloader = process_data(train_data, tokenizer, max_seq=max_seq, dataset='train',batch_size=batchsize)
    model.train()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = AdamW(model.parameters(), lr=2e-5)
    lr_scheduler = transformers.get_scheduler("linear",optimizer, int(warmup*len(train_dataloader)*num_epoch), len(train_dataloader)*num_epoch)
    global_step = 0
    best_acc = 0.0
    for ep in trange(num_epoch, desc="Training"):
        for step, batch in tqdm(enumerate(train_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids = batch
            output = model(input_ids, input_mask)
            logits = output.logits
            loss = loss_fct(logits, label_ids)
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1
        model.eval()
        dev_acc = eval(device, dev_data, model, tokenizer, batchsize, max_seq)
        logger.info("Epoch {} acc {} step {}".format(ep, dev_acc, global_step))
        if dev_acc > best_acc:
            best_acc = dev_acc
            save_model(model, tokenizer, output_dir='saved_model_{}'.format(test_id))  # save model
        model.train()
    model.eval()
            
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)            
            
def save_model(model_to_save, tokenizer, output_dir):
    save_model_dir = os.path.join(output_dir, 'saved_model')
    mkdir(save_model_dir)
    output_model_file = os.path.join(save_model_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(save_model_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file, _use_new_zipfile_serialization=False)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(save_model_dir)            
    
def eval_1(preds, labels):
    acc = np.sum((preds == labels).astype(np.int32))/len(preds)
    return acc

def eval(device, dev_data, model, tokenizer, batchsize=32, max_seq=50):
    test_dataloader = process_data(dev_data, tokenizer, max_seq=max_seq, dataset='test',batch_size=batchsize)

    model.eval()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')

    eval_loss = 0
    num_steps = 0
    preds = []
    labels = []
    for step, batch in enumerate(tqdm(test_dataloader, desc='test')):
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids = batch
            output = model(input_ids, input_mask)
            logits = output.logits
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
batch_size = 32
np.random.seed(24)
torch.manual_seed(24)
max_seq = 50
num_epoch = 20
warmup = 0.3
saved_file = 'run_chinese_bert_split_{}'.format(test_id)
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

model = BertForSequenceClassification.from_pretrained('bert-chinese-base', num_labels=11).to(device)
tokenizer = BertTokenizer.from_pretrained('bert-chinese-base')
train(device, train_data, dev_data, model, tokenizer, batchsize=batch_size, max_seq=max_seq, num_epoch=num_epoch, test_id=test_id, warmup=warmup)

