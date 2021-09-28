#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

KoELECTRA in TensorFLOW 



refs 

https://github.com/monologg/KoELECTRA
https://github.com/puzzlecollector/Natural-Language-Based-Climate-Technology-Classification-AI-Contest/blob/e4a0ec5965ff64fa2b18e52c318f9b83738a39e2/ensemble_KoELECTRA_XLM_RoBERTA.ipynb
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
from transformers import TFElectraModel, ElectraTokenizer
from transformers import TFElectraForSequenceClassification
#model = TFElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
model = TFElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator", num_labels=46)
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")



tf.random.set_seed(12345)
np.random.seed(12345)
train= pd.read_csv('train_preproc.csv')
test=pd.read_csv('test_preproc.csv')
sample_submission=pd.read_csv('sample_submission.csv')



BATCH_SIZE = 12
NUM_EPOCHS = 10
VALID_SPLIT = 0.2 
MAX_LEN = 96 # max token size for BERT, ELECTRA

def electra_tokenizer(sent, MAX_LEN):  
    encoded_dict = tokenizer.encode_plus(
        text = sent, 
        add_special_tokens = True, # add [CLS] and [SEP]
        pad_to_max_length = False, 
        return_attention_mask = True # constructing attention_masks 
    )  
    
    input_id = encoded_dict['input_ids'] 
    attention_mask = encoded_dict['attention_mask'] # differentiate padding from non padding 
    token_type_id = encoded_dict['token_type_ids'] # differentiate two sentences, not "really" necessary for now    
    
    if len(input_id) > 512: # head + tail methodology 
        input_id = input_id[:129] + input_id[-383:] 
        attention_mask = attention_mask[:129] + attention_mask[-383:]  
        token_type_id = token_type_id[:129] + token_type_id[-383:]    
        print("Long Text!! Using Head+Tail Truncation")
    elif len(input_id) <= 512: 
        input_id = input_id + [0]*(512 - len(input_id)) 
        attention_mask = attention_mask + [0]*(512 - len(attention_mask))
        token_type_id = token_type_id + [0]*(512 - len(token_type_id))  
        
    return input_id, attention_mask, token_type_id




input_ids =[]
attention_masks =[]
token_type_ids =[]
train_data_labels = []

for train_sent, train_label in zip(train['data'], train['label']):
    try:
        input_id, attention_mask, token_type_id = electra_tokenizer(train_sent, MAX_LEN=MAX_LEN)
        
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        #########################################
        train_data_labels.append(train_label)
        
    except Exception as e:
        print(e)
        print(train_sent)
        pass




train_input_ids=np.array(input_ids, dtype=int)
train_attention_masks=np.array(attention_masks, dtype=int)
train_token_type_ids=np.array(token_type_ids, dtype=int)
###########################################################

train_input_ids = train_input_ids[:,0,:]
train_attention_masks = train_attention_masks[:,0,:]
train_token_type_ids = train_token_type_ids[:,0,:]


train_inputs=(train_input_ids, train_attention_masks, train_token_type_ids)
train_labels=np.asarray(train_data_labels, dtype=np.int32)



###




class Electra_Model(tf.keras.Model):
    def __init__(self, num_category):
        super(Electra_Model, self).__init__()

        self.bert = bert_model
        self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(num_category, 
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range), 
                                                name="classifier")
        
    def call(self, inputs, attention_mask=None, token_type_ids=None, training=False):
        
        #outputs 값: # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1] 
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)

        return logits



#from sklearn.model_selection import KFold


classification_model = Electra_Model(num_category=46)

os_model_name = "RoBERTa-Climate-0815"



#with mirrored_strategy.scope():    
# 학습 준비하기
optimizer = tf.keras.optimizers.Adam(3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
classification_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])


# overfitting을 막기 위한 ealrystop 추가
## min_delta: the threshold that triggers the termination (acc should at least improve 0.0001)
## patience: no improvment epochs (patience = 1, 1번 이상 상승이 없으면 종료)\

    



os_model_name = "BERT-Climate2"

#earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001,patience=5)
earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001,patience=5)


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
history = classification_model.fit(train_inputs, train_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
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

sample_submission.to_csv('M11-KoELECTRA-base-Submission_TEST.csv', index=False)
