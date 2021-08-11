# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:27:16 2021

@author: PC
"""


import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import os

train_data = pd.read_csv("train.csv")
train_data.index = range(len(train_data))

train_data=train_data[['과제명', '요약문_연구내용','요약문_연구목표','요약문_한글키워드','label']]

train_data['요약문_연구내용'].fillna('NAN', inplace=True)
train_data['요약문_연구목표'].fillna('NAN', inplace=True)
train_data['요약문_한글키워드'].fillna('NAN', inplace=True)

train_data['data']=train_data['과제명']+train_data['요약문_연구내용']+train_data['요약문_연구목표']+train_data['요약문_한글키워드']


def clean_text(texts):
    corpus = []
    for i in range(0, len(texts)):
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '',str(texts[i])) #remove punctuation
        review = re.sub(r'\d+','', str(texts[i]))# remove number
        review = review.lower() #lower case
        review = re.sub(r'\s+', ' ', review) #remove extra space
        review = re.sub(r'<[^>]+>','',review) #remove Html tags
        review = re.sub(r'\s+', ' ', review) #remove spaces
        review = re.sub(r"^\s+", '', review) #remove space from start
        review = re.sub(r'\s+$', '', review) #remove space from the end
        corpus.append(review)
    return corpus

train_data.data = clean_text(train_data.data)

train_data_text = list(train_data['data'])

train_clear_text = []

for i in tqdm(range(len(train_data_text))):
  train_clear_text.append(str(train_data_text[i]).replace('\\n', ''))
train_data['clear_text'] = train_clear_text
train_data.head()

train_clear_text = list(train_data['clear_text'])

train_clear_text2 = []

for text in train_clear_text:
  temp = re.sub('[-=+,#:;//●<>▲\?:^$.☆!◦【】★()□▶■○◆ ❏Ⅰ◎①@*\"※~>`\'…》]', ' ', text)
  train_clear_text2.append(temp)
train_data['data'] = train_clear_text2

print(train_data.head())

labeling = list(train_data['label'])

################# 0인지 확인 
X_train = []

for i in tqdm(range(len(labeling))):
    if labeling[i] == 0:
        X_train.append(0)
    else:
        X_train.append(1)
       
train_data['label2'] = X_train

################ 중분류
X2_train = []

for i in tqdm(range(len(labeling))):
    if labeling[i] == 0:
        X2_train.append(0)
    elif 1 <= labeling[i]  <= 22:
        X2_train.append(1)
    elif 23 <= labeling[i]  <= 40:
        X2_train.append(2)
    elif 41 <= labeling[i]  <= 45:
        X2_train.append(3)

train_data['label3'] = X2_train        
  
  
# 0이면 0 - 1부터 22까지 1 / 23부터 40까지 2 / 41부터 45까지 3 
train_data2 = train_data[['data','label','label2','label3']]

        

train_data2.to_csv("binary_+++_labeling_train.csv", encoding="utf-8-sig")