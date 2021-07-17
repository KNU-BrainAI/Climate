# -*- coding: utf-8 -*-
"""

Toxic Comment Classification by Bert
Eunchan 0716, Kaggle 
"""


import numpy as np
import pandas as pd
import re                                  
import string  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from transformers import BertModel, BertConfig



## Device setting 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)



## load csv files
submission_df=pd.read_csv('dataset/sample_submission.csv')
train_df=pd.read_csv("dataset/train.csv")
test_df = pd.read_csv('dataset/test.csv')



## Preprocessing 
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

train_df['comment_text'] = train_df['comment_text'].apply(lambda x: clean_text(x))
X = list(train_df['comment_text'])
y = np.array(train_df.loc[:,'toxic':])

test_df['comment_text'] = test_df['comment_text'].apply(lambda x: clean_text(x))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)



def preprocessing_for_bert(data):
    input_ids = []
    attention_masks = []
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text = sent,
            add_special_tokens = True,
            max_length = MAX_LEN,
            pad_to_max_length = True,
            return_attention_mask = True
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
      
    return input_ids, attention_masks


all_text = np.concatenate([train_df.comment_text.values, test_df.comment_text.values])
len_sent = [len(sent) for sent in all_text]
avg_len = np.mean(len_sent)

MAX_LEN = 150
token_ids = list(preprocessing_for_bert([X[0]])[0].squeeze().numpy())
print(X[0])
print(token_ids)

train_inputs, train_masks = preprocessing_for_bert(X_train)
val_inputs, val_masks = preprocessing_for_bert(X_val)




from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_val)
batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


import torch
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 6
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
                        nn.Linear(D_in, H),
                        nn.ReLU(),
                        nn.Linear(H, D_out))
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:,0,:]
        logits = self.classifier(last_hidden_state_cls)
        
        return logits


from transformers import AdamW, get_linear_schedule_with_warmup

def initialize_model(epochs=4):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=False)
    
    bert_classifier.to(device)
    
    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                     lr=5e-5, #Default learning rate
                     eps=1e-8 #Default epsilon value
                     )
    
    # Total number of training steps
    total_steps = len(train_dataloader) * epochs
    
    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps=0, # Default value
                                              num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler


import random
import time

loss_fn = nn.BCEWithLogitsLoss()

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    print("Start training...\n")
    for epoch_i in range(epochs):
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)
        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            logits = model(b_input_ids, b_attn_mask)
            loss = loss_fn(logits, b_labels.float())
            batch_loss += loss.item()
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
            if (step % 10 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
    avg_train_loss = total_loss / len(train_dataloader)
    
    print("-"*70)
    
    if evaluation == True:
        val_loss, val_accuracy = evaluate(model, val_dataloader)
        time_elapsed = time.time() - t0_epoch
        print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
        print("-"*70)
    print("\n")
    
print("Training complete!")



def evaluate(model, val_dataloader):
    model.eval()
    val_accuracy = []
    val_loss = []
    for batch in val_dataloader:
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        loss = loss_fn(logits, b_labels.float())
        val_loss.append(loss.item())
        accuracy = accuracy_thresh(logits.view(-1,6), b_labels.view(-1,6))
        val_accuracy.append(accuracy)
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    
    return val_loss, val_accuracy

def accuracy_thresh(y_pred, y_true, thresh:float=0.5, sigmoid:bool=True):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    return ((y_pred>thresh)==y_true.byte()).float().mean().item()


set_seed(42)
bert_classifier, optimizer, scheduler = initialize_model(epochs=1)
train(bert_classifier, train_dataloader, val_dataloader, epochs=1, evaluation=True)





import torch.nn.functional as F

def bert_predict(model, test_dataloader):
    model.eval()
    all_logits = []
    for batch in test_dataloader:
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    all_logits = torch.cat(all_logits, dim=0)
    probs = all_logits.sigmoid().cpu().numpy()
    
    return probs


probs = bert_predict(bert_classifier, val_dataloader)

test_inputs, test_masks = preprocessing_for_bert(test_df.comment_text)
test_dataset = TensorDataset(test_inputs, test_masks)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=32)

probs = bert_predict(bert_classifier, test_dataloader)

submission = pd.DataFrame(probs,columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate'])
test_df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]=submission
final_sub = test_df[['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']]

final_sub.to_csv('submission.csv', index=False)


