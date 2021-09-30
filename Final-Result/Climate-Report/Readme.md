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

**1.1 Deep Learning**  

**1.2. NLP**  

**1.3. Classification**   

**1.4. Language Models**  

**1.5. Objective (연구 목적)**  




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



**2.3. Computation methods + hyperparameters**


- Fixed-Parameter

```(python)
NUM_EPOCHS = 10
VALID_SPLIT = 0.2
MAX_LEN=96

optimizer = tf.keras.optimizers.Adam(3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
```


**2.4. Evaluation Method, Public and Private Score** 



## 3. RESULTS
- feedback in 랩미팅) 모델간의 차이를 구분하려는거라서 나중에 public만 보여줘도 상관 없을듯

**3.1. BAR GRAPH**



**3.2. TABLE**  

|Filename|Model|Details|Public(F1)|Private(F1)|
|------|------|---|---|---|
|**Model #1**|BERT-base-Multilingual-cased|Fixed-Parameter|0.65340|0.64159|
|**Model #2**|BERT-base-Multilingual-uncased|Fixed-Parameter|0.70660|0.68062|
|**Model #3**|KLUE-BERT-base|Fixed-Parameter|0.74360|0.72007|
|**Model #4**|SKT-KoBERT(cased)|Fixed-Parameter|---|---|
|**Model #5**|KoBERT/monologg|Fixed-Parameter|0.67256|0.66650|
|**Model #6**|RoBERTa-base|Fixed-Parameter|0.59196|0.56519|
|**Model #7**|KLUE-RoBERTa-base|Fixed-Parameter|0.71893|0.69808|
|**Model #8**|KLUE-RoBERTa-large|Fixed-Parameter|---|---|
|**Model #9**|XLM-RoBERTa-large|Fixed-Parameter|0.63074|0.60900|
|**Model #10**|KoELECTRA-small-v3-discriminator|Fixed-Parameter|---|---|  
|**Model #11**|KoELECTRA-base-v3-discriminator|Fixed-Parameter|---|---|  


- ***uncased**: it does not make a difference between english and English.  

**3.3. Performance plot as a function of the number of epochs**


- 은찬) 기존 Result가 10에폭이니까 10 20 30 등 descrete 하게 하기 (한 에폭씩 꺾은선 그래프로 보기는 무리가 있을듯)  
- **to be late**



## 4. CONCLUSIONS AND DISCUSSION

(Professor: 이게 길어야 좋은 논문이다..)
ㅠㅠㅠ


