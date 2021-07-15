# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 16:24:04 2021

@author: PC
"""

from __future__ import unicode_literals, print_function, division
from io import open 
import glob
import os 

def findFiles(path): return glob.glob(path)

print(findFiles('../../data/names/*.txt'))

import unicodedata 
import string 

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

#유니코드 문자열을 ASCII로 변환, https://stackoverflow.com/a/518232/2809427

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
        )

print(unicodeToAscii('Ślusàrski'))

#각 언어의 이름 목록인 category_lines 사전 생성 
category_lines = {} 
all_categories = [] 

# 파일을 읽고 줄 단위로 분리 
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('../../data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines
    
n_categories = len(all_categories)

print(category_lines['Italian'][:5])




import torch


# all-letters 로 문자의 주소 찾기, 예시 "a" = 0 
def letterToIndex(letter):
    return all_letters.find(letter)

#검증을 위해서 한개의 문자를 <1 x n_letters> Tensor로 변환 
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


#한 줄(이름)을 <line_length x 1 x n_letters>,
#또는 One-hot 문자 벡터의 Array로 변경 
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor 

print(letterToTensor('J'))

print(lineToTensor('Jones').size())


import torch.nn as nn 

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
        
    def forward(self, input, hidden):
        combined = torch.cat((input,hidden),1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)




input = letterToTensor('A')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input, hidden)


input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)


#네트워크 출력(각 카테고리의 우도)으로 가장 확률이 높은 카테고리 이름(언어)과 카테고리 번호 반환
def categoryFromOutput(output):
    top_n, top_i = output.topk(1) #텐서의 가장 큰 값 및 주소 
    category_i = top_i[0].item() #텐서에서 정수 값으로 변경
    return all_categories[category_i], category_i

print(categoryFromOutput(output))   

#학습 예시(하나의 이름과 그 언어)를 얻는 빠른 방법도 필요합니다.
import random 

def randomChoice(l):
    return l[random.randint(0, len(l) -1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor 

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)
    

#네트워크 학습 





