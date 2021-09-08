## Final-Result 

--- 
## Models 

|Filename|Model|설명|F1 스코어|
|------|---|---|---|
|**Model1**|BERT|Fixed-Parameter|---|
|**Model2**|KoBERT|Fixed-Parameter|---|
|**Model3**|RoBERTa-base|Fixed-Parameter|---|
|**Model4**|XLM-RoBERTa-base|Fixed-Parameter|---|
|**Model5**|?|---|


## Fixed-Parameter

```(python)
NUM_EPOCHS = 10
VALID_SPLIT = 0.2
MAX_LEN=96

optimizer = tf.keras.optimizers.Adam(3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
```




