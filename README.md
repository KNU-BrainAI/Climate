# Climate  



### 소개 

[Climate Technology 데이콘 경진대회](https://www.notion.so/sangtaeahn/fa65fed2d3994a1c9cb4b7596838790d#cccd2125f49748e5adcb82cc75b8d198)


### 최종 결과 

To Be Updated .. 


### Members 

|이름|Github|pictures|
|:---|:---|:---:|
|이창현|[@2changhyeon](https://github.com/2changhyeon)|/home/brainai/사진/ROBERT-Inputid.png|
|이은찬|[@purang2](https://github.com/purang2)|중앙정렬|

이창현  
이은찬  


### Works 

**Model**   
1. BERT     
2. KoBERT  
3. KoElectra  
4. RoBERTa-Large  


**전처리**   
1. Mecab 
2. BERT-Tokenizer
3. KoBERT-Tokenizer
4. RoBERTa-Tokenizer 


**전략**
1. Imbalanced Data → Oversampling  
2. Imbalanced Data → Focal Loss  
3. 한국어 데이터 Preprocessing → Mecab + re.sub()  
4. 2단 분류 (0과 0이 아닌것 분류 +0이 아닌것들 재분류) 
5. 다양한 NLP 고성능 모델 적용  
6. Multi-GPU 사용 (tensorflow.distribute.MirroredStrategy)  
7. To Be Updated..





### Requirements (Packages)

```python
# it's Our version!  
cudatoolkit==11.3.1  
cudnn==8.2.1
python==3.8.0
pytorch==1.9.0 
tensorflow-gpu==2.5.0
koNLPy==0.5.2
tqdm
transformers==4.8.2
spyder==5.0.5  
```













### Examples (공부했던 예제 기록) 

1. [First Name('성')을 분석하여 어느 나라 출신인지 예측하는 모델](https://tutorials.pytorch.kr/intermediate/char_rnn_classification_tutorial.html)  
2. [뉴스 기사 문장을 [1. 세계, 2.스포츠 3. 경제 4. 과학] 분류하기 , Pytorch TorchText 사용](https://tutorials.pytorch.kr/beginner/text_sentiment_ngrams_tutorial.html)  
3. [Kaggle 악플 분석 챌린지](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/code)  




