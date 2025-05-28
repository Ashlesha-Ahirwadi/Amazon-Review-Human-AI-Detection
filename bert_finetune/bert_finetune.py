import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('/Users/ashleshaahirwadi/Desktop/NLP_Project/data/fake reviews dataset.csv')
df = df[['text_', 'label']].dropna()
df = df[df['label'].isin(['CG', 'OR'])]
df['label'] = (df['label'] == 'CG').astype(int)

# Train/test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text_'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42, stratify=df['label'])

# Tokenization
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_texts, train_labels, tokenizer)
test_dataset = ReviewDataset(test_texts, test_labels, tokenizer)

# Model
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    roc_auc = roc_auc_score(labels, pred.predictions[:,1])
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc_auc}

training_args = TrainingArguments(
    output_dir='./results_bert',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluate
preds = trainer.predict(test_dataset)
y_true = np.array(test_labels)
y_pred = np.argmax(preds.predictions, axis=1)
y_prob = preds.predictions[:,1]

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_prob)
cm = confusion_matrix(y_true, y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Save confusion matrix
os.makedirs('results', exist_ok=True)
np.save('results/bert_confusion_matrix.npy', cm)

# Plot and save confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Original', 'CG'], yticklabels=['Original', 'CG'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (BERT)')
plt.tight_layout()
plt.savefig('results/bert_confusion_matrix.png')
plt.close() 