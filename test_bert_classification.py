from transformers import BertTokenizer, BertForSequenceClassification
import torch
from utils import *

tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
model = BertForSequenceClassification.from_pretrained('./results/best')

model.eval()

def predict(text, model, tokenizer, max_length=512):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    with torch.no_grad():
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    predicted_class = torch.argmax(logits, dim=1).item()
    
    return predicted_class

text = "UNİLATERAL DİGİTAL MAMOGRAFİ İNCELEMESİ\nSol meme mamografik olarak tip B meme paternindedir.\nSol meme alt iç kadranda lobüle yapıda noktasal kalsifkasyonlar içeren anterior konturu meme parankimi ile örtülü dens kitle izlendi.\nPatolojik mikrokalsifikasyon kümesi saptanmamıştır.\nAksiller alanda patolojik boyutta büyümüş lenf nodu veya kitle saptanmamıştır."
predicted_class = predict(clean_text(text), model, tokenizer)

print(f"Tahmin edilen sınıf: {predicted_class}")