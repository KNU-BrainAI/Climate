# -*- coding: utf-8 -*-
"""


전처리 코드 짜기

re-Organize at 09/07

"""

import pandas as pd
import random 

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import os



from konlpy.tag import Mecab
mecab = Mecab('C:/mecab/mecab-ko-dic')

train= pd.read_csv('data/dacon-data/train.csv')
test=pd.read_csv('data/dacon-data/test.csv')
sample_submission=pd.read_csv('data/dacon-data/sample_submission.csv')



train=train[['과제명', '요약문_한글키워드', '요약문_연구목표', 'label']]
test=test[['과제명', '요약문_한글키워드', '요약문_연구목표']]
train['과제명']=train['과제명'].fillna('')
test['과제명']=test['과제명'].fillna('')
train['요약문_한글키워드']=train['요약문_한글키워드'].fillna('')
test['요약문_한글키워드']=test['요약문_한글키워드'].fillna('')
train['요약문_연구목표']=train['요약문_연구목표'].fillna('')
test['요약문_연구목표']=test['요약문_연구목표'].fillna('')


train['data']=train['과제명']+train['요약문_한글키워드']+train['요약문_연구목표']
test['data']=test['과제명']+test['요약문_한글키워드']+test['요약문_연구목표']


train=train[['data','label']]
test=test['data']




import math 

#train = train[:100]
train_data = []
for x in range(len(train)):

    target_text = train.iloc[x]['data']
    
    target_nouns = mecab.nouns(target_text)
    target_x = ""
    
    for i in range(len(target_nouns)):
        target_x = target_x + " " + target_nouns[i]
    
    
   
    train_data.append(target_x)
        
        
train['data'] = train_data





test_data = []
for x in range(len(test)):

    target_text = test.iloc[x]
    
    target_nouns = mecab.nouns(target_text)
    target_x = ""
    
    for i in range(len(target_nouns)):
        target_x = target_x + " " + target_nouns[i]
        
        
    
    test_data.append(target_x)
    
test = pd.DataFrame(test_data)





    
test.columns = ['data']
train.to_csv('train_preproc.csv',index=False,encoding="utf-8-sig")
test.to_csv('test_preproc.csv',index=False,encoding="utf-8-sig")

