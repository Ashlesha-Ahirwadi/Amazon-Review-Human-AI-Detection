import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('/Users/ashleshaahirwadi/Desktop/NLP_Project/data/fake reviews dataset.csv')

# Basic preprocessing: drop NAs, keep only needed columns
df = df[['text_', 'label']].dropna()

# Feature engineering: lexical features
def type_token_ratio(text):
    tokens = text.split()
    return len(set(tokens)) / (len(tokens) + 1e-6)

def avg_sentence_length(text):
    sentences = text.split('.')
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return 0
    return np.mean([len(s.split()) for s in sentences])

def punctuation_ratio(text):
    punct = sum([1 for c in text if c in '.,;:!?'])
    return punct / (len(text) + 1e-6)

df['type_token_ratio'] = df['text_'].apply(type_token_ratio)
df['avg_sentence_length'] = df['text_'].apply(avg_sentence_length)
df['punctuation_ratio'] = df['text_'].apply(punctuation_ratio)

# TF-IDF features
vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=5000)
tfidf = vectorizer.fit_transform(df['text_'])

# Combine features
lexical_features = df[['type_token_ratio', 'avg_sentence_length', 'punctuation_ratio']].values
from scipy.sparse import hstack
X = hstack([tfidf, lexical_features])
y = (df['label'] == 'CG').astype(int)  # 1 = Computer-generated, 0 = Original

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train logistic regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Save confusion matrix
os.makedirs('results', exist_ok=True)
np.save('results/baseline_confusion_matrix.npy', cm)

# Plot and save confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Original', 'CG'], yticklabels=['Original', 'CG'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Baseline)')
plt.tight_layout()
plt.savefig('results/baseline_confusion_matrix.png')
plt.close() 