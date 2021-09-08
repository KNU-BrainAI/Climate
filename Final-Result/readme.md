## Final-Result 

--- 
## Models 

|Filename|Model|설명|F1 스코어|
|------|---|---|---|
|**Model1**|BERT|10 Epochs|---|
|**Model2**|KoBERT|10 Epochs|---|
|**Model3**|RoBERTa-base|10 Epochs|---|
|**Model4**|XLM-RoBERTa-base|10 Epochs|---|
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




