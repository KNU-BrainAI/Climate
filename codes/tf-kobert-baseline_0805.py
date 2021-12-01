# -*- coding: utf-8 -*-
"""

2021-08-04-수 오후 8:26 ~

@Title: KoBERT 직접 구현 수목 뿌시기 
@author: Eunchan Lee 


@References [내가 참고한 사이트들]

1) transforemrs BertModel(pyTorch) 설명 
https://towardsdatascience.com/how-to-use-bert-from-the-hugging-face-transformer-library-d373a22b0209


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

from torch.nn import functional as F
from tokenization_kobert import KoBertTokenizer
from transformers import BertModel, DistilBertModel, AutoTokenizer, TFBertModel
#tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert') # monologg/distilkobert도 동일

#random seed 고정
tf.random.set_seed(1234)
np.random.seed(1234)
BATCH_SIZE = 8
NUM_EPOCHS = 30
VALID_SPLIT = 0.2

#MAX_LEN=96
MAX_LEN=96*3




train= pd.read_csv('train_preproc_with연구목표_0804.csv')
#train= pd.read_csv('train_preproc_with연구목표_0804_light.csv')
test=pd.read_csv('test_preproc_with연구목표_0804.csv')
sample_submission=pd.read_csv('data/dacon-data/sample_submission.csv')





tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert') # monologg/distilkobert도 동일
bert_model = TFBertModel.from_pretrained('monologg/kobert', from_pt=True)
#distilbert_model = DistilBertModel.from_pretrained('monologg/distilkobert')


'''

#Example 
print(tokenizer.encode("보는내내 그대로 들어맞는 예측 카리스마 없는 악역"))

text = "보는내내 그대로 들어맞는 예측 카리스마 없는 악역"

encoding = tokenizer.encode_plus(text, 
                                 add_special_tokens = True,    
                                 truncation = True, 
                                 #padding = "max_length",
                                 max_length=MAX_LEN, 
                                 pad_to_max_length=True, 
                                 return_attention_mask = True, 
                                 return_tensors = "tf") #pt = Pytorch tensors


#

'''


'''
#if you want to retrieve the specific parts of the encoding, 
#inputs = encoding["input_ids"][0]
#attention_mask = encoding["attention_mask"][0]
#token_type_id = encoding['token_type_ids'][0]


#tf.Tensor 

inputs = encoding["input_ids"]   
attention_mask = encoding["attention_mask"]
token_type_id = encoding['token_type_ids']


#Additionally, because the tokenizer returns a dictionary of different values, 
#instead of finding those values as shown above and individually passing these into the model, 
#we can just pass in the entire encoding like this: output = bert_model(**encoding) 

output = bert_model(**encoding) 

'''




def bert_tokenizer(text, MAX_LEN):
    
    encoding = tokenizer.encode_plus(text, 
                                     add_special_tokens = True,    
                                     truncation = True, 
                                     #padding = "max_length",
                                     max_length=MAX_LEN, 
                                     pad_to_max_length=True, 
                                     return_attention_mask = True, 
                                     return_tensors = "tf") #pt = Pytorch tensors
    
    input_id = encoding["input_ids"]   
    attention_mask = encoding["attention_mask"]
    token_type_id = encoding['token_type_ids']
    
    return input_id, attention_mask, token_type_id











# Train 데이터를 Tokenizing 시키고 준비된 list에 넣는 코드 
## try-except 거슬리니까 나중에 빼도 되는지 검토하기 

input_ids =[]
attention_masks =[]
token_type_ids =[]
train_data_labels = []

for train_sent, train_label in zip(train['data'], train['label']):
    try:
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












'''
class Bert_Model(nn.Module):
   def __init__(self, class):
       super(Bert_Model, self).__init__()
       self.bert = bert_model
       self.out = nn.Linear(self.bert.config.hidden_size, classes)
   def forward(self, input):
       _, output = self.bert(**input)
       out = self.out(output)
       return out
   
    
'''   



class BertModel_byBrainAILAB(tf.keras.Model):
    def __init__(self, num_category):
        super(BertModel_byBrainAILAB, self).__init__()

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




classification_model = BertModel_byBrainAILAB(num_category=46)





# 학습 준비하기
optimizer = tf.keras.optimizers.Adam(3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
classification_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])


# overfitting을 막기 위한 ealrystop 추가
## min_delta: the threshold that triggers the termination (acc should at least improve 0.0001)
## patience: no improvment epochs (patience = 1, 1번 이상 상승이 없으면 종료)\

    



os_model_name = "KoBERT-Climate"

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

sample_submission.to_csv('KOBERT-Submission_0805_6epoch.csv', index=False)

