# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 15:21:40 2021

@author: PC
"""

import pandas as pd
import random 

train= pd.read_csv('data/dacon-data/train.csv')
test=pd.read_csv('data/dacon-data/test.csv')
sample_submission=pd.read_csv('data/dacon-data/sample_submission.csv')


james = train['label']

james_value_counts = james.value_counts() 




train=train[['과제명', '요약문_연구목표', 'label']]
test=test[['과제명', '요약문_연구목표']]
train['요약문_연구목표'].fillna('NAN', inplace=True)
test['요약문_연구목표'].fillna('NAN', inplace=True)

train['data']=train['과제명']+train['요약문_연구목표']
test['data']=test['과제명']+test['요약문_연구목표']


train=train[['data','label']]
test=test['data']


# label 0 5천개 

sample_0 = [] 
cnt = 0
for i in range(len(train)):
    if train.iloc[i].label == 0:
        sample_0.append(train.iloc[i].tolist())
        cnt = cnt + 1     
    if cnt >= 5000: break 


sample = []

#0~45

for i in range(46):
    sample.append([])


for i in range(len(train)):
    sample[train.iloc[i].label].append(train.iloc[i].tolist())
    

#각 label별 곱셈 가중치 -> notion / research/ DACON / Action item/ 데이터 분석 
multiple_weight=[1,3,13,25,60,2,40,30,10,34,5,17,13,6,2,20,6,25,2,1,3,7,25,2,1,4,11,6,8,3,14,4,13,6,6,12,3,13,13,17,8,50,70,10,35,3]


for i in range(len(multiple_weight)):
    sample[i] = sample[i] * multiple_weight[i]
    


train_over_sample = []  
train_over_sample = train_over_sample + sample_0

for i in range(45):
    train_over_sample = train_over_sample + sample[i+1]


tris = train_over_sample

random.shuffle(train_over_sample)

apeach = pd.DataFrame(tris)
ryan = pd.DataFrame(train_over_sample)
ryan.columns = ['data', 'label']
apeach.columns = ['data', 'label']


ryan.to_csv("balance_dataset_shuffle.csv",encoding="utf-8-sig")

apeach.to_csv("balance_dataset.csv",encoding="utf-8-sig")


#train_over_sample




