import streamlit as st
import pickle
import string
import nltk
import time
import pandas as pd
import numpy as np
import os
from nltk.corpus import stopwords

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Fake News Detector Pro",
    page_icon="📰",
    layout="centered"
)

# -------------------- SAFE MODEL PATH --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# -------------------- STOPWORDS --------------------
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# -------------------- DARK MODE --------------------
theme = st.sidebar.toggle("🌙 Dark Mode")

if theme:
    bg_color = "#1e1e1e"
    text_color = "white"
    card_color = "#2c2c2c"
else:
    bg_color = "#f4f6f9"
    text_color = "#2c3e50"
    card_color = "white"

# -------------------- CUSTOM CSS --------------------
st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
}}
.card {{
    background-color: {card_color};
    padding: 40px;
    border-radius: 15px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.15);
}}
.title {{
    text-align: center;
    font-size: 34px;
    font-weight: bold;
    color: {text_color};
}}
.subtitle {{
    text-align: center;
    color: gray;
    margin-bottom: 25px;
}}
.result {{
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    margin-top: 20px;
}}
.footer {{
    text-align: center;
    margin-top: 30px;
    font-size: 14px;
    color: gray;
}}
</style>
""", unsafe_allow_html=True)

# -------------------- CLEAN FUNCTION --------------------
def clean_text(text):
    text = str(text).lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -------------------- AI EXPLANATION --------------------
def get_top_features(vectorized_text, model, vectorizer, top_n=5):
    feature_names = np.array(vectorizer.get_feature_names_out())
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        sorted_idx = np.argsort(coefs)
        prediction = model.predict(vectorized_text)[0]

        if prediction == 0:
            top_features = feature_names[sorted_idx[:top_n]]
        else:
            top_features = feature_names[sorted_idx[-top_n:]]

        return top_features
    return []

# -------------------- SESSION HISTORY --------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- UI CARD --------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='title'>📰 Fake News Detector Pro</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI Powered News Classification System</div>", unsafe_allow_html=True)

user_input = st.text_area("Enter News Text Below:", height=180)

if st.button("Analyze News"):
    if user_input.strip() != "":
        with st.spinner("Analyzing..."):
            time.sleep(1)
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            probability = model.predict_proba(vectorized)[0]
            confidence = max(probability) * 100

        label = "Real News" if prediction == 0 else "Fake News"

        st.session_state.history.append({
            "Text": user_input[:50] + "...",
            "Prediction": label,
            "Confidence (%)": round(confidence, 2)
        })

        st.markdown("---")

        if prediction == 0:
            st.markdown(
                f"<div class='result' style='color:green;'>✅ {label}<br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result' style='color:red;'>❌ {label}<br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )

        st.progress(int(confidence))

        chart_data = pd.DataFrame({
            "Label": ["Real", "Fake"],
            "Probability": probability
        })
        st.bar_chart(chart_data.set_index("Label"))

        st.markdown("### 🤖 Why this prediction?")
        top_words = get_top_features(vectorized, model, vectorizer)

        if len(top_words) > 0:
            for word in top_words:
                st.write(f"• {word}")
        else:
            st.write("Explanation not available for this model type.")

    else:
        st.warning("⚠ Please enter some text.")

st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.history:
    st.markdown("### 📜 Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history))

st.markdown("<div class='footer'>Developed with Machine Learning & Streamlit</div>", unsafe_allow_html=True)
