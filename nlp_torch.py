#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install transformers')
get_ipython().system('git clone https://github.com/indobenchmark/indonlu')


# In[6]:


import random
import numpy as np
import pandas as pd
import torch
import os
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from nltk.tokenize import TweetTokenizer
from indonlu.utils.forward_fn import forward_sequence_classification
from indonlu.utils.metrics import document_sentiment_metrics_fn
from indonlu.utils.data_utils import DocumentSentimentDataset, DocumentSentimentDataLoader


# In[3]:


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def count_param(module, trainable=False):
    if trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
  
def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('{}:{:.2f}'.format(key, value))
    return ' '.join(string_list)


# In[4]:


set_seed(19072021)


# In[5]:


tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
config = BertConfig.from_pretrained('indobenchmark/indobert-base-p1')
config.num_labels = DocumentSentimentDataset.NUM_LABELS
  
model = BertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p1', config=config)
model


# In[6]:


count_param(model)


# In[7]:


get_ipython().system('pwd')


# In[8]:


lokasi = os.path.join("/content/indonlu/dataset/smsa_doc-sentiment-prosa")


# In[9]:


train_dataset_path = f'{lokasi}/train_preprocess.tsv'
valid_dataset_path = f'{lokasi}/valid_preprocess.tsv'
test_dataset_path = f'{lokasi}/test_preprocess_masked_label.tsv'


# In[10]:


train_dataset = DocumentSentimentDataset(train_dataset_path, tokenizer, lowercase=True)
valid_dataset = DocumentSentimentDataset(valid_dataset_path, tokenizer, lowercase=True)
test_dataset = DocumentSentimentDataset(test_dataset_path, tokenizer, lowercase=True)

train_loader = DocumentSentimentDataLoader(dataset=train_dataset, max_seq_len=512, batch_size=32, num_workers=16, shuffle=True)  
valid_loader = DocumentSentimentDataLoader(dataset=valid_dataset, max_seq_len=512, batch_size=32, num_workers=16, shuffle=False)  
test_loader = DocumentSentimentDataLoader(dataset=test_dataset, max_seq_len=512, batch_size=32, num_workers=16, shuffle=False)


# In[11]:


w2i, i2w = DocumentSentimentDataset.LABEL2INDEX, DocumentSentimentDataset.INDEX2LABEL


# In[12]:


text = 'Aku ganteng, kamu jelek'
subwords = tokenizer.encode(text)
subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)
  
logits = model(subwords)[0]
label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
  
print(f'Text: {text} | Label : {i2w[label]} ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)')


# In[13]:


optimizer = optim.Adam(model.parameters(), lr=3e-6)
model = model.cuda()


# In[14]:


n_epochs = 5
for epoch in range(n_epochs):
  model.train()
  torch.set_grad_enabled(True)

  total_train_loss = 0
  list_hyp, list_label = [], []

  train_pbar = tqdm(train_loader, leave=True, total=len(train_loader))
  for i, batch_data in enumerate(train_pbar):
      loss, batch_hyp, batch_label = forward_sequence_classification(model, batch_data[:-1], i2w=i2w, device='cuda')

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      tr_loss = loss.item()
      total_train_loss = total_train_loss + tr_loss

      list_hyp += batch_hyp
      list_label += batch_label

      train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
          total_train_loss/(i+1), get_lr(optimizer)))

  metrics = document_sentiment_metrics_fn(list_hyp, list_label)
  print("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch+1),
      total_train_loss/(i+1), metrics_to_string(metrics), get_lr(optimizer)))

  model.eval()
  torch.set_grad_enabled(False)

  total_loss, total_correct, total_labels = 0, 0, 0
  list_hyp, list_label = [], []

  pbar = tqdm(valid_loader, leave=True, total=len(valid_loader))
  for i, batch_data in enumerate(pbar):
      batch_seq = batch_data[-1]        
      loss, batch_hyp, batch_label = forward_sequence_classification(model, batch_data[:-1], i2w=i2w, device='cuda')
      
      valid_loss = loss.item()
      total_loss = total_loss + valid_loss

      list_hyp += batch_hyp
      list_label += batch_label
      metrics = document_sentiment_metrics_fn(list_hyp, list_label)

      pbar.set_description("VALID LOSS:{:.4f} {}".format(total_loss/(i+1), metrics_to_string(metrics)))
      
  metrics = document_sentiment_metrics_fn(list_hyp, list_label)
  print("(Epoch {}) VALID LOSS:{:.4f} {}".format((epoch+1),
      total_loss/(i+1), metrics_to_string(metrics)))


# In[16]:


text = 'jelek'
subwords = tokenizer.encode(text)
subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)
  
logits = model(subwords)[0]
label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
  
print(f'Text: {text} | Label : {i2w[label]} ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)')


# In[26]:


torch.save(model, "/content/drive/MyDrive/Colab Notebooks/otak.pt")


# In[11]:


device = torch.device("cpu") # Jika diload menggunakan CPU pakai perintah ini
model = torch.load("otak.pt", map_location=device)
model.eval()


# In[15]:


tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
w2i, i2w = DocumentSentimentDataset.LABEL2INDEX, DocumentSentimentDataset.INDEX2LABEL


# In[27]:


text = "tertarik, saya ingin membelinya"
subwords = tokenizer.encode(text)
subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)
  
logits = model(subwords)[0]
label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
  
print(f'Text: {text} | Label : {i2w[label]} ({F.softmax(logits, dim=-1).squeeze()[label] * 100:.3f}%)')

