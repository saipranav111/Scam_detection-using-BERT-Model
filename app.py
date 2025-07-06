# ============================================
# app.py : Scam/Clickbait Detection with BERT
# ============================================

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# 1️⃣ Load & Clean Data
# ---------------------------
print("📥 Loading dataset...")
df = pd.read_csv('clickbait_data.csv')  # Make sure headache.csv is in same folder

print("\n✅ Head of dataset:")
print(df.head())

print("\n✅ Data Info:")
print(df.info())

# Drop rows with missing values
df = df.dropna(subset=['text', 'label']).reset_index(drop=True)

# Map labels if they are text
if df['label'].dtype == 'object':
    df['label'] = df['label'].str.lower().map({'scam':1, 'clickbait':1, 'non-clickbait':0, 'ham':0, 'spam':1})
    df['label'] = df['label'].fillna(0).astype(int)

print("\n✅ Label counts:")
print(df['label'].value_counts())

# ---------------------------
# 2️⃣ Split into Train/Test
# ---------------------------
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'],
    df['label'],
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

print(f"\n✅ Train size: {len(train_texts)}, Test size: {len(test_texts)}")

# ---------------------------
# 3️⃣ Tokenize
# ---------------------------
print("\n🔑 Tokenizing...")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=64)

# ---------------------------
# 4️⃣ Create Dataset Class
# ---------------------------
class ScamClickbaitDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ScamClickbaitDataset(train_encodings, train_labels)
test_dataset = ScamClickbaitDataset(test_encodings, test_labels)

# ---------------------------
# 5️⃣ Load Model & Trainer
# ---------------------------
print("\n🚀 Loading BERT for sequence classification...")

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy='epoch',
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# ---------------------------
# 6️⃣ Train Model
# ---------------------------
print("\n🎓 Training model...")
trainer.train()

# ---------------------------
# 7️⃣ Evaluate
# ---------------------------
print("\n📊 Evaluating...")

predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)

acc = accuracy_score(test_labels, preds)
print(f"\n✅ Test Accuracy: {acc:.2%}\n")

print(classification_report(test_labels, preds))

cm = confusion_matrix(test_labels, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Safe', 'Clickbait/Scam'],
            yticklabels=['Safe', 'Clickbait/Scam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ---------------------------
# 8️⃣ Predict New Text
# ---------------------------
def predict_new_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred = torch.argmax(probs, axis=1).item()
        confidence = probs[0][pred].item()
    label = "🚨 Clickbait / Scam" if pred == 1 else "✅ Safe / Non-Clickbait"
    return label, confidence

print("\n🔍 Testing prediction on sample text...")
sample_text = "Congratulations! You won a free trip. Click here now!"
label, conf = predict_new_text(sample_text)
print(f"Text: {sample_text}")
print(f"Prediction: {label} ({conf:.2%} confidence)")

# Optional: Interactive prediction
while True:
    user_input = input("\nEnter a headline/text to test (or type 'exit'): ")
    if user_input.lower() == 'exit':
        break
    label, conf = predict_new_text(user_input)
    print(f"Prediction: {label} ({conf:.2%} confidence)")
