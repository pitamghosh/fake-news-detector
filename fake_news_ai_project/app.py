import pandas as pd
import numpy as np
import string
import nltk
import pickle

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ---------------- CLEAN FUNCTION ----------------
def clean_text(text):
    text = str(text).lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ---------------- LOAD DATA ----------------
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake["label"] = 1
real["label"] = 0

data = pd.concat([fake, real])
data = data.sample(frac=1).reset_index(drop=True)

data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["label"]

# ---------------- TF-IDF (UPGRADED) ----------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X_vectorized = vectorizer.fit_transform(X)

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------------- SAVE ----------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
