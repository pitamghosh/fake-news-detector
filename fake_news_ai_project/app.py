import streamlit as st
import pickle
import string
import nltk
import os
import requests
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

# ---------------- NEWS API ----------------
API_KEY = "YOUR_NEW_API_KEY_HERE"  # এখানে নতুন key বসাবি

def web_verification(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={API_KEY}"
    response = requests.get(url).json()
    
    if response.get("status") == "ok":
        return response.get("totalResults", 0)
    return 0

# ---------------- UI ----------------
st.title("📰 AI-Based Fake News Detector")

user_input = st.text_area("Enter News Text")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # ---- ML Prediction ----
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        probability = model.predict_proba(vectorized)[0]
        
        real_prob = probability[0]
        fake_prob = probability[1]

        if real_prob > fake_prob:
            ml_result = "Real"
            confidence = real_prob * 100
        else:
            ml_result = "Fake"
            confidence = fake_prob * 100

        # ---- Web Verification ----
        results_found = web_verification(user_input)

        # ---- Final Decision Logic ----
        if ml_result == "Fake" and results_found < 3:
            final_result = "Fake News ❌"
        elif ml_result == "Real" and results_found > 3:
            final_result = "Real News ✅"
        else:
            final_result = "Uncertain ⚠️ Needs Manual Verification"

        # ---- Display ----
        st.subheader("🔍 ML Prediction:")
        st.write(f"{ml_result} (Confidence: {confidence:.2f}%)")

        st.subheader("🌐 Web Verification:")
        st.write(f"Related Articles Found: {results_found}")

        st.subheader("🎯 Final Decision:")
        st.write(final_result)
