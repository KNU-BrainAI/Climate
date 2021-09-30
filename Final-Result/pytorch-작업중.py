#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09/27
ref
#1)https://www.kaggle.com/atulanandjha/bert-testing-on-imdb-dataset-extensive-tutorial
#2)https://dacon.io/competitions/official/235744/codeshare/3100?page=1&dtype=recent
SKT/ KoBERT in PyTorch
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import os
import tqdm

from konlpy.tag import Okt

import sklearn
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import log_loss, accuracy_score,f1_score
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


import torch 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler 
from torch.nn import functional as F
from transformers import BertModel, AutoTokenizer, TFBertModel
import torch.nn as nn 
#random seed 고정
tf.random.set_seed(12345)
np.random.seed(12345)
BATCH_SIZE = 12
NUM_EPOCHS = 10
VALID_SPLIT = 0.2

MAX_LEN=96


train= pd.read_csv('train_preproc.csv')
test=pd.read_csv('test_preproc.csv')
sample_submission=pd.read_csv('data/dacon-data/sample_submission.csv')

tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1', sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})
bert_model = TFBertModel.from_pretrained('skt/kobert-base-v1', from_pt=True)

def bert_tokenizer(text, MAX_LEN):
    
    encoding = tokenizer.encode_plus(text, 
                                     add_special_tokens = True,    
                                     truncation = True, 
                                     #padding = "max_length",
                                     max_length=MAX_LEN, 
                                     pad_to_max_length=True,  
                                     #return_tensors = "pt",
                                     return_attention_mask = True
                                     ) #pt = Pytorch tensors
    
    #input_id = encoding["input_ids"]   
    #attention_mask = encoding["attention_mask"]
    #token_type_id = encoding['token_type_ids']
    
    input_id = encoding.get('input_ids')   
    attention_mask = encoding.get('attention_mask')
    token_type_id = encoding.get('token_type_ids')
    
    input_id = torch.tensor(input_id)
    attention_mask = torch.tensor(attention_mask)
    token_type_id = torch.tensor(token_type_id)
    
    return input_id, attention_mask, token_type_id


input_ids =[]
attention_masks =[]
token_type_ids =[]
train_data_labels = []

for train_sent, train_label in zip(train['data'], train['label']):
    try:
        train_label = torch.tensor(train_label)
        input_id, attention_mask, token_type_id = bert_tokenizer(train_sent, MAX_LEN=MAX_LEN)
        
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        #########################################
        train_data_labels.append(train_label)
        
    except Exception as e:
        print(e)
        print(train_sent)
        pass




#PT

class Bert_PyTorch(nn.Module):
    def __init__(self, dropout=0.1):
        super(Bert_PyTorch, self).__init__()
        
        self.bert = bert_model 
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.linear = nn.Linear(768, 46)
        self.sigmoid = nn.Sigmoid() 
        
    def forward(self, tokens, attention_mask=None):
        _, pooled_output = self.bert(inputs, attention_mask=attention_mask, output_all_encoded_layers=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        logits = self.sigmoid(linear_output)
        return logits

classification_model = Bert_PyTorch()


optimizer = Adam(classification_model.parameters(), lr=3e-5)
loss = nn.CrossEntropyLoss() 

'''
for ep in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) 
        optimizer.step()
        scheduler.step() 
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(ep+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(ep+1, train_acc / (batch_id+1)))

    
    model.eval()
    
    for test_batch_id, (test_token_ids, test_valid_length, test_segment_ids, test_label) in enumerate(tqdm_notebook(test_dataloader)):
        test_token_ids = test_token_ids.long().to(device)
        test_segment_ids = test_segment_ids.long().to(device)
        test_valid_length= test_valid_length
        test_label = test_label.long().to(device)
        test_out = model(token_ids, valid_length, segment_ids)
        test_loss = loss_fn(out, label)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (test_batch_id+1)))

    if test_acc > highest_acc:
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss,
            }, '/content/drive/MyDrive/natural/torchckpt/model.pt')
        patience = 0
    else:
        print("test acc did not improved. best:{} current:{}".format(highest_acc, test_acc))
        patience += 1
        if patience > 5:
            break
    print('current patience: {}'.format(patience))
    print("************************************************************************************")



os_model_name = "Sept-KoBERT-Climate"
#earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001,patience=5)
earlystop_callback = EarlyStopping(monitor='val_f1', min_delta=0.0001,patience=5)
checkpoint_path = os.path.join(os_model_name, 'weights.h5')
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create path if exists
if os.path.exists(checkpoint_dir):
    print("{} -- Folder already exists \n".format(checkpoint_dir))
else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("{} -- Folder create complete \n".format(checkpoint_dir))
    
cp_callback = ModelCheckpoint(
    checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)
# 학습과 eval 시작
history = classification_model.fit(train_inputs, train_labels, epochs=6, batch_size=8,
                    validation_split = VALID_SPLIT, callbacks=[earlystop_callback, cp_callback])
#훈련 후 Test data 검증 
input_ids =[]
attention_masks =[]
token_type_ids =[]
train_data_labels = []
for test_sent in test['data']:
    try:
        #input_id, attention_mask, token_type_id = bert_tokenizer(clean_text(test_sent), MAX_LEN=40)
        input_id, attention_mask, token_type_id = bert_tokenizer(test_sent, MAX_LEN=MAX_LEN)
        
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        #########################################
       
    except Exception as e:
        print(e)
        print(test_sent)
        pass
      
test_input_ids=np.array(input_ids, dtype=int)
test_attention_masks=np.array(attention_masks, dtype=int)
test_token_type_ids=np.array(token_type_ids, dtype=int)
###########################################################
test_input_ids = test_input_ids[:,0,:]
test_attention_masks = test_attention_masks[:,0,:]
test_token_type_ids = test_token_type_ids[:,0,:]
test_inputs=(test_input_ids, test_attention_masks, test_token_type_ids)
results = classification_model.predict(test_inputs)
results=tf.argmax(results, axis=1)
sample_submission['label']=results
sample_submission.to_csv('0927-SKT-KOBERT-Submission.csv', index=False)

'''
