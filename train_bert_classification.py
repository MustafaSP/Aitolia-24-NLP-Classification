import json
from collections import defaultdict
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import random
from utils import *

json_datas=[]
with open("All_Labeled.json", 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line.strip())
        json_datas.append(json_obj)

grouped_datas = defaultdict(list)
grouped_datas[1]=[]
grouped_datas[2]=[]
grouped_datas[3]=[]
grouped_datas[4]=[]
grouped_datas[5]=[]

for item in json_datas:
    impression = None
    for label in item['label']:
        if label[2] == "IMPRESSION":
            impression = item['text'][label[0]:label[1]]
            break
    
    for grouped_data in grouped_datas:
        if impression.find(str(grouped_data)):
            grouped_datas[grouped_data].append(item['text'])


# JSON verilerini yükleme
json_datas = []
with open("All_Labeled.json", 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            json_datas.append(json_obj)


# Verileri gruplama
grouped_datas = defaultdict(list)
grouped_datas["1"]=[]
grouped_datas["2"]=[]
grouped_datas["3"]=[]
grouped_datas["4"]=[]
grouped_datas["5"]=[]

for item in json_datas:
    impression = None
    for label in item['label']:
        if label[2] == "IMPRESSION":
            text = item['text']
            impression = item['text'][label[0]:label[1]]
            item['text']=clean_text(item['text'][:label[0]-7])
            break
    
    for grouped_data in grouped_datas:
        if impression.find(grouped_data) != -1:
            grouped_datas[grouped_data].append(item['text'])

# BERT tokenizer ve model yükleme
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
model = BertForSequenceClassification.from_pretrained('dbmdz/bert-base-turkish-cased', num_labels=len(grouped_datas))

# Dataset sınıfı
class MamografiDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)  # dtype torch.long olarak bırakılmalı
        }

# Etiketleri sayısal değerlere çevirme
impression_to_label = {impression: idx for idx, impression in enumerate(grouped_datas.keys())}
texts = []
labels = []
for impression, text_list in grouped_datas.items():
    for text in text_list:
        texts.append(text)
        labels.append(impression_to_label[impression])

# Listeyi karıştırma
combined = list(zip(texts, labels))
random.shuffle(combined)
texts, labels = zip(*combined)

# Eğitim ve doğrulama veri seti oluşturma
dataset = MamografiDataset(texts=texts, labels=labels, tokenizer=tokenizer, max_length=512)

# Eğitim parametreleri ayarlama
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=32,
    per_device_train_batch_size=4,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1000,  # Daha az sıklıkta loglama
    save_steps=2000,     # Daha az sıklıkta model kaydetme
    save_total_limit=3,  # En fazla 3 checkpoint sakla
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Eğitimi başlatma
trainer.train()

trainer.train()

