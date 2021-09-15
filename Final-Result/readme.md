## Final-Result 

--- 



## Pre-Processing 

- 12 Train columns 중 4가지 '과제명', '요약문_한글키워드', '요약문_연구목표', 'label' 사용 
- **Mecab Tokenizer**를 통해 형태소 단위 분절 후 **명사형태만 추출하는 방식**을 최종 적용

<img src="images/after_preproc.PNG">



## Models 

|Filename|Model|설명|F1 스코어|
|------|---|---|---|
|**Model1**|BERT|Fixed-Parameter|0.64159|
|**Model2**|KoBERT|Fixed-Parameter|0.66650|
|**Model3**|RoBERTa-base|Fixed-Parameter|---|
|**Model4**|XLM-RoBERTa-base|Fixed-Parameter|---|
|**Model5**|KLUE-RoBERTa|Fixed-Parameter|---|
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




