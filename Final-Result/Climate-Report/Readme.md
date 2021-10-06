# Climate-report
Climate Description! 😁🤞



-- 원래(Lab github Climate) 내용 -- 



# 연구 Journal Source !!! (은찬 작성중입니당.)

포맷 제공☆: 교수님 




## Target 

ICT Express https://www.journals.elsevier.com/ict-express

## 포맷 참고 논문(은찬, In ICT EXPRESS)
- [A comparative study of LPWAN technologies for large-scale IoT deployment](https://www.sciencedirect.com/science/article/pii/S2405959517302953)


# Title (tentative)

## Classifying Climate Technology in Research Proposals using Pretrained Language Models: A comparative Study  

---

## 1. INTRODUCTION (연구 소개)  
<!--
**1.1 Deep Learning**  




딥러닝은 많은 머신러닝 알고리즘 중에 인공 신경망(Artificial Neural Network)를 기반으로 한 방법들을 통칭하는 것을 의미한다.


머신러닝(Machine Learning)은 규칙을 일일히 프로그래밍하지 않아도 자동으로 데이터에서 규칙을 학습하는 알고리즘을 연구하는 분야입니다. 또한 이는 통계학과 깊은 관련이 있습니다. 통계학에서 유래된 머신러닝 알고리즘이 많으며 통계학과 컴퓨터 과학 분야가 상호 작용하면서 발전하고 있습니다. 최근 머신러닝의 발전은 통계나 수학 이론보다 경험을 바탕으로 발전하는 경우가 많습니다. 컴퓨터 과학 분야가 이런 발전을 주도하고 있으며, 컴퓨터 과학 분야의 대표적인 머신러닝 라이브러리는 사이킷런(Scikit-learn)입니다. 종종 사람들은 인공 신경망과 딥러닝을 크게 구분하지 않고 사용하니다.[1]


인공 신경망(Artificial Neural Network)은 뇌에 있는 생명학적 뉴런의 네트워크에서 영감을 받은 머신러닝 모델입니다. 이 인공 신경망은 딥러닝의 핵심입니다. 인공 신경망은 다재다능하고 강력하고 확장성이 좋아서 수백만 개의 이미지를 분류하거나, 음성 인식 서비스의 성능을 높이거나, 매일 수억 명에 이르는 사용자에게 가장 좋은 비디오를 추천하거나, 바둑 세계 챔피언을 이기기 위해 수백만 개의 기보를 익히고 자기자신과 게임하면서 학습하는(딥마인드의 알파고) 등 아주 복잡한 대규모 머신러닝 문제를 다루는 데 적합합니다.[2]


두 번째 AI 겨울 기간에도 여전히 인공지능에 대해 연구한 사람이 있었고 점차 이들의 연구가 차츰 빛을 보면서 인공지능 기술이 다시 주목받기 시작했습니다. 1998년 얀 르쿤이 최초의 합성공 신경망 모델을 만들어 손글씨 숫자를 인식하는 데 성공했습니다. 그 이후 2012년에 제프리 힌턴(Geoffrey Hinton)의 팀이 이미지 분류 대회인 ImageNet에서 기존의 머신러닝 방법을 누르고 압도적인 성능으로 우승했습니다. 힌턴이 사용한 모델은 AlexNet이며 역시 합성곱 신경망입니다. 이때부터 이미지 분류 작업 등에 합성곱 신경망 딥러닝 기법이 널리 사용되기 시작했습니다. [1]


[1]. [textbook 혼자 공부하는 머신러닝+딥러닝 ,박해선]  
[2]. [핸즈 온 머신러닝, 오렐리앙 제롬]  


Natural Language Processing (NLP) is ~.



**1.2. NLP**  

NLP is 인기가 많아지기 시작한 기술이다. 

Natural Language를 타겟으로 한다. 

Word Embedding 기법을 통해서 자연어의 연관관계를 벡터로 나타내고 이를 이용하여 딥러닝 네트워크는 다양한 자연어 처리 태스크에서 좋은 성능을 보일 수 있습니다. ~~~ 쏼라쏼라(더 적읍시다) ~~~~

**1.3. Classification** 

전통적으로 행해져 왔던 머신러닝의 큰 분야인 Classification은 ~~~~ 입니다. 

이는 딥러닝에서도 마찬가지로 매우 보편적인 태스크이며 널리 쓰이고 있습니다

자연어 처리에서도 스팸 분류 영화 리뷰 긍정 부정 분류 등에서 딥러닝 네트워크를 통한 자연어 처리를 가능하게 할 수 있습니다.

(더 적기)




**1.4. Language Models**  

- Language Model의 정의:~~

- 어텐션과 트랜스포머에 대한 인트로

- BERT 


- RoBERTa


- ELECTRA 
- 

-->
**⭐ Objective (연구 목적)**  

BERT와 같은 프리트레인 모델은 NLP Tasks의 성능을 끌어올리는 데에 공헌을 하였





## 2. METHODS


**2.1. Dataset : Fixed Datasets** 

- feedback in 랩미팅) Method - Dataset 포함시키기(대표적인 예 몇개만) 


**Pre-Processing** 

- 12 Train columns 중 4가지 '과제명', '요약문_한글키워드', '요약문_연구목표', 'label' 사용 
- **Mecab Tokenizer**를 통해 형태소 단위 분절 후 **명사형태만 추출하는 방식**을 최종 적


**2.2. 4 different LMs (BERT, KoBERT, RoBERTa, ELECTRA)**
- BERT 


- KoBERT 


- RoBERTa


- ELECTRA



**2.3. Computation methods + hyperparameters + Evaluation Method, Public and Private Score**


- Fixed-Parameter

```(python)
NUM_EPOCHS = 10
VALID_SPLIT = 0.2
MAX_LEN=96

optimizer = tf.keras.optimizers.Adam(3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
```






## 3. RESULTS
- feedback in 랩미팅) 모델간의 차이를 구분하려는거라서 나중에 public만 보여줘도 상관 없을듯

**3.1. BAR GRAPH**



**3.2. TABLE**  

|Filename|Model|Details|Public(F1)|Private(F1)|
|------|------|---|---|---|
|**Model #1**|BERT-base-Multilingual-cased|Fixed-Parameter|0.65340|0.64159|
|**Model #2**|BERT-base-Multilingual-uncased|Fixed-Parameter|0.70660|0.68062|
|**Model #3**|KLUE-BERT-base|Fixed-Parameter|0.74360|0.72007|
|**Model #4**|KoBERT/monologg|Fixed-Parameter|0.67256|0.66650|
|**Model #5**|RoBERTa-base|Fixed-Parameter|0.59196|0.56519|
|**Model #6**|KLUE-RoBERTa-base|Fixed-Parameter|0.71893|0.69808|
|**Model #7**|KLUE-RoBERTa-large|Fixed-Parameter|---|---|
|**Model #8**|XLM-RoBERTa-large|Fixed-Parameter|0.63074|0.60900|
|**Model #9**|KoELECTRA-small-v3-discriminator|Fixed-Parameter|---|---|  
|**Model #10**|KoELECTRA-base-v3-discriminator|Fixed-Parameter|---|---|  


- ***uncased**: it does not make a difference between english and English.  

**3.3. Performance plot as a function of the number of epochs**


- 은찬) 기존 Result가 10에폭이니까 10 20 30 등 descrete 하게 하기 (한 에폭씩 꺾은선 그래프로 보기는 무리가 있을듯)  
- **to be late**



## 4. CONCLUSIONS AND DISCUSSION

(Professor: 이게 길어야 좋은 논문이다..)
ㅠㅠㅠ


