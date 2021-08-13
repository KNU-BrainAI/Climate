# -*- coding: utf-8 -*-
"""



0/1 분류 csv + 1 중에서 분류 

최종 합치는 코드

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

submission_2 = pd.read_csv('notzero-Submission_0812_1epoch.csv')

submission_ans = pd.read_csv('KOBERT-binary_Submission_0812_5epoch.csv')



sub2_label = [] 


for i in range(len(submission_2)):
    sub2_label.append(submission_2.iloc[i]['index'])


j = 0
for i in range(len(submission_ans)):
    if submission_ans.iloc[i]['index'] in sub2_label: 
        submission_ans.iloc[i]['label'] = submission_2.iloc[j]['label']
        j = j+1





submission_ans.to_csv("submission_2중분류_ans_0812은찬1.csv",encoding="utf-8-sig",index=False)










