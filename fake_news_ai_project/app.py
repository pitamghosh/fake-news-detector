import streamlit as st
import pickle
import string
import nltk
import pandas as pd
import numpy as np
import os
from nltk.corpus import stopwords

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# ---------------- STOPWORDS ----------------
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# ---------------- CLEAN FUNCTION ----------------
def clean_text(text):
    text = str(text).lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ---------------- UI ----------------
st.title("📰 Fake News Detector")

user_input = st.text_area("Enter News Text")

if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        probability = model.predict_proba(vectorized)[0]

        real_prob = probability[0]
        fake_prob = probability[1]

        # Decision
        if real_prob > fake_prob:
            final_verdict = "Real"
            confidence = real_prob * 100
            st.success("Real News ✅")
        else:
            final_verdict = "Fake"
            confidence = fake_prob * 100
            st.error("Fake News ❌")

        st.write(f"Confidence: {confidence:.2f}%")

       
