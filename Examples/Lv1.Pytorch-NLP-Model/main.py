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

#softmax와 적합한 loss function
criterion = nn.NLLLoss()


learning_rate = 0.005 # 너무 높으면 발산할 수 있고, 너무 낮으면 학습이 안 될 수 있습니다 

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden() 
    
    rnn.zero_grad() 
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
        
    loss = criterion(output, category_tensor)
    loss.backward()
    
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
        
    return output, loss.item() 

#예시 데이터를 사용하여 실행 
import time 
import math 

n_iters = 100000 
print_every = 5000 
plot_every = 1000 



#도식화를 위한 손실 추적 
current_loss = 0 
all_losses = [] 

def timeSince(since):
    now = time.time() 
    s = now - since 
    m = math.floor(s / 60)
    s -= m * 60 
    return '%dm %ds' % (m,s)

start = time.time() 


for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss 
    
    
    #iter 숫자, 손실, 이름, 추측 화면 출력 
    if iter % print_every ==0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter/n_iters *100, timeSince(start), loss, line, guess, correct))
    
    if iter % plot_every == 0: 
        all_losses.append(current_loss / plot_every)
        current_loss = 0 
        

#결과 도식화 
## all_losses 를 이용한 손실 도식화는 네트워크 학습을 보여준다 

import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker 

plt.figure()
plt.plot(all_losses)


#결과 평가 
## Confusion Matrix 

confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000 

#주어진 라인의 출력 반환 
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
        
    return output 

# 예시들 중 어떤 것이 정확하게 예측되었는지 기록
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1
    
# 모든 행을 합계로 나누어 정규화
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()


#도식 설정 
fig = plt.figure() 
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)    

#축 설정 
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)


# 모든 tick에서 레이블 지정
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()


# Application 
# 사용자 입력으로 실행 


def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))
        
        #Get top N categories 
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = [] 
        
        for i in range(n_predictions):
            value = topv[0][i].item() 
            category_index = topi[0][i].item() 
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])


predict('Dovesky')
predict('Jackson')
predict('Aaron')
predict('Lee')
predict('Yamamoto')
predict('Zhang')







