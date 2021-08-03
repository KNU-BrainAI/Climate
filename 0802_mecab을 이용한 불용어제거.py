# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import os

train_data = pd.read_csv("../train.csv")
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
  temp = re.sub('[-=+,#:;//●<>▲\?:^$.☆!★()Ⅰ@*\"※~>`\'…》]', ' ', text)
  train_clear_text2.append(temp)
train_data['clear_text'] = train_clear_text2

print(train_data.head())



from konlpy.tag import Mecab
mecab = Mecab('C:/mecab/mecab-ko-dic')

stop_df = pd.read_csv('한국어불용어100.txt',sep = '\t', header = None, names = ['형태','품사','비율'])

stop_df.loc[100] = '가'
stop_df.loc[101] = '합니다'


stop_words = list(stop_df.형태)


X_train = []

text = list(train_data['clear_text'])

for i in tqdm(range(len(text))):
  temp_X = []
  temp_X = mecab.nouns(text[i]) # 토큰화
  temp_X = [word for word in temp_X if not word in stop_words] # 불용어 제거
  temp_X = [word for word in temp_X if len(word) > 1]
  X_train.append(temp_X)


for i in range(len(X_train)):
    X_train[i] = ' '.join(X_train[i])

train_data['data'] = X_train
train_data2 = train_data[['data','label']]

train_data2.to_csv("mecab0803_alltrain.csv", encoding="utf-8-sig")
