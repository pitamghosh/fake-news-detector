import streamlit as st
import pickle
import string
import nltk
import time
import pandas as pd
import numpy as np
from nltk.corpus import stopwords

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Fake News Detector Pro",
    page_icon="📰",
    layout="centered"
)

# -------------------- LOAD MODEL --------------------
import os

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

# -------------------- EXPLANATION FUNCTION --------------------
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
    else:
        return []

# -------------------- SESSION HISTORY --------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- UI --------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='title'>📰 Fake News Detector Pro</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI Powered News Classification System</div>", unsafe_allow_html=True)

user_input = st.text_area("Enter News Text Below:", height=180)

if st.button("Analyze News"):

    if user_input.strip() == "":
        st.warning("⚠ Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(1)

            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])

            probability = model.predict_proba(vectorized)[0]

            fake_prob = probability[0]
            real_prob = probability[1]

            # Confidence threshold logic
            if real_prob > 0.65:
                label = "Real News"
                confidence = real_prob * 100
            elif fake_prob > 0.65:
                label = "Fake News"
                confidence = fake_prob * 100
            else:
                label = "Uncertain"
                confidence = max(real_prob, fake_prob) * 100

        # Save history
        st.session_state.history.append({
            "Text": user_input[:50] + "...",
            "Prediction": label,
            "Confidence (%)": round(confidence, 2)
        })

        st.markdown("---")

        # RESULT
        if label == "Real News":
            st.markdown(
                f"<div class='result' style='color:green;'>✅ {label}<br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )
        elif label == "Fake News":
            st.markdown(
                f"<div class='result' style='color:red;'>❌ {label}<br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result' style='color:orange;'>⚠ {label}<br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )

        st.progress(int(confidence))

        # Probability Chart
        chart_data = pd.DataFrame({
            "Label": ["Fake", "Real"],
            "Probability": [fake_prob, real_prob]
        })
        st.bar_chart(chart_data.set_index("Label"))

        # Explanation
        st.markdown("### 🤖 Why this prediction?")
        top_words = get_top_features(vectorized, model, vectorizer)

        if len(top_words) > 0:
            st.write("Top Influential Words:")
            for word in top_words:
                st.write(f"• {word}")
        else:
            st.write("Explanation not available for this model type.")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------- HISTORY DISPLAY --------------------
if st.session_state.history:
    st.markdown("### 📜 Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history))

# -------------------- FOOTER --------------------
st.markdown("<div class='footer'>Developed with Machine Learning & Streamlit</div>", unsafe_allow_html=True)
