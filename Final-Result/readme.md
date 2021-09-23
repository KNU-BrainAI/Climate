## Final-Result 

--- 



## Pre-Processing 

- 12 Train columns 중 4가지 '과제명', '요약문_한글키워드', '요약문_연구목표', 'label' 사용 
- **Mecab Tokenizer**를 통해 형태소 단위 분절 후 **명사형태만 추출하는 방식**을 최종 적용

<img src="images/after_preproc.PNG">



## Models 

**Single Models**  
|Filename|Model|Details|Macro F1|
|------|------|---|---|
|**Model1**|BERT-base-Multilingual-cased|Fixed-Parameter|0.64159|
|**Model2**|BERT-base-Multilingual-uncased|Fixed-Parameter|---|
|**Model3**|KoBERT/monologg|Fixed-Parameter|0.66650|
|**Model4**|KLUE-RoBERTa-large|Fixed-Parameter|---|
|**Model5**|KLUE-RoBERTa-base|Fixed-Parameter|---|
|**Model6**|KLUE-BERT-base|Fixed-Parameter|---|
|**Model7**|KoELECTRA-base-v3-discriminator|Fixed-Parameter|---|  
|**Model9**|SKT-KoBERT(cased)|Fixed-Parameter|---|


***uncased**: it does not make a difference between english and English.  

**Ensemble Models**  
|Filename|Model|설명|F1 스코어|
|------|------|---|---|
|**앙상블 Model1**|?|Fixed-Parameter|---|
|**앙상블 Model2**|?|Fixed-Parameter|---|


## Fixed-Parameter

```(python)
NUM_EPOCHS = 10
VALID_SPLIT = 0.2
MAX_LEN=96

optimizer = tf.keras.optimizers.Adam(3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
```




