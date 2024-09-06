import re

def clean_text(text):
    text = text.lower()
    text = text.strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text