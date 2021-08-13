# -*- coding: utf-8 -*-
"""



0/1 분류 csv + 1 중에서 분류 

make Not Zero test data 

@author: PC
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import os
import tqdm

from konlpy.tag import Okt



submission = pd.read_csv('KOBERT-binary_Submission_0812_5epoch.csv')

#submission_2 = pd.read_csv('notzero-Submission_0812_1epoch.csv')

#submission_ans = pd.read_csv('KOBERT-binary_Submission_0812_5epoch.csv')

test=pd.read_csv('test_preproc_0809.csv')

#sample_submission=pd.read_csv('data/dacon-data/sample_submission.csv')

notzero_label = [] 


for i in range(len(submission)):
    
    if submission.iloc[i]['label'] == 1: 
        notzero_label.append(submission.iloc[i]['index']-174304)


make_test = [] 
submission_notzero = []
#j = 0
for i in range(len(test)):
    if test.iloc[i][0] in notzero_label: 
        make_test.append(test.iloc[i].tolist())
        submission_notzero.append(submission.iloc[i].tolist())

make_test_df = pd.DataFrame(make_test)
sub_notzero_df = pd.DataFrame(submission_notzero)

make_test_df.columns = ['index','data']
sub_notzero_df.columns = ['index','label']


make_test_df.to_csv("notzero-test-7196.csv",encoding="utf-8-sig")
sub_notzero_df.to_csv("notzero-submission-7196.csv",encoding="utf-8-sig")










