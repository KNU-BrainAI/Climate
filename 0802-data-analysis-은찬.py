# -*- coding: utf-8 -*-
"""
Data Analysis 0802 은찬 
"""

import pandas as pd
import random 

train= pd.read_csv('data/dacon-data/train.csv')
test=pd.read_csv('data/dacon-data/test.csv')
sample_submission=pd.read_csv('data/dacon-data/sample_submission.csv')



james = train['label']

james_value_counts = james.value_counts() 


#####
# 어떤 데이터 사용할지 정하는 PART 
train=train[['과제명', '요약문_연구목표', '요약문_연구내용','요약문_기대효과','요약문_한글키워드', 'label']]
test=test[['과제명', '요약문_연구목표', '요약문_연구내용','요약문_기대효과','요약문_한글키워드']]
train['요약문_연구내용'].fillna('NAN', inplace=True)
test['요약문_연구내용'].fillna('NAN', inplace=True)
train['요약문_연구목표'].fillna('NAN', inplace=True)
test['요약문_연구목표'].fillna('NAN', inplace=True)
train['요약문_기대효과'].fillna('NAN', inplace=True)
test['요약문_기대효과'].fillna('NAN', inplace=True)
train['요약문_한글키워드'].fillna('NAN', inplace=True)
test['요약문_한글키워드'].fillna('NAN', inplace=True)


train['data']=train['과제명']+train['요약문_연구내용']+train['요약문_연구목표']+train['요약문_기대효과']+train['요약문_한글키워드']
test['data']=test['과제명']+test['요약문_연구내용']+test['요약문_연구목표']+test['요약문_기대효과']+test['요약문_한글키워드']




#####
#TRAIN DATA를 SIZE 500이하로 맞추는 코드 
lens = list() 

for i in range(len(train)):
    #train.iloc[i]['len'] = len(train.iloc[i].data)
    lens.append(len(train.iloc[i].data))
    

marek = pd.DataFrame(lens)    
train['before-len'] = marek 


lee_sin = list() 
for i in range(len(train)):
    #train.iloc[i]['data'] = train.iloc[i]['data'][:510]
    lee_sin.append(train.iloc[i]['data'][:500])

    

mareks = pd.DataFrame(lee_sin)    
train['afdata'] = mareks


lenss = list() 

for i in range(len(train)):
    #train.iloc[i]['len'] = len(train.iloc[i].data)
    lenss.append(len(train.iloc[i].afdata))
    

mareks = pd.DataFrame(lenss)    
train['len'] = mareks 


##### 


#TEST DATA를 SIZE 500이하로 맞추는 코드 
jayce = list() 

for i in range(len(test)):
    #train.iloc[i]['data'] = train.iloc[i]['data'][:510]
    jayce.append(test.iloc[i]['data'][:500])


marekss = pd.DataFrame(jayce)
test['afdata'] = marekss

#####

train=train[['data','afdata','label','before-len','len']]
test=test[['data','afdata']]


ryan = train[['afdata','label']]
ryan.columns = ['data', 'label']


apeach = test[['afdata']]
apeach.columns = ['data']



ryan.to_csv("train_maxlen_500.csv",encoding="utf-8-sig")
apeach.to_csv("test_maxlen_500.csv",encoding="utf-8-sig")







