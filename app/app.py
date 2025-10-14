import streamlit as st
import joblib
import re

# -----------------------------
# 1️⃣ Load models & vectorizers
# -----------------------------
# Make sure all 4 files are in the same folder as app.py
model_a = joblib.load("Model_A.pkl")  # Balanced model
model_b = joblib.load("Model_B.pkl")  # Imbalanced model

tfidf_a = joblib.load("balanced_tfidf_vectorizer.pkl")
tfidf_b = joblib.load("imbalanced_tfidf_vectorizer.pkl")

# -----------------------------
# 2️⃣ Helper: text cleaning
# -----------------------------
def clean_text(text):
    text = text.strip().lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# -----------------------------
# 3️⃣ Streamlit UI
# -----------------------------
st.set_page_config(page_title="Automated Review Rating System", layout="centered")

st.title("Automated Review Rating Predictor")

# Input box
user_input = st.text_area("✍️ Enter your review :", height=150)

if st.button("SUBMIT"):
    if user_input.strip() == "":
        st.warning("Please enter a review before prediction.")
    else:
        # Clean and transform input for both models
        cleaned_review = clean_text(user_input)

        input_vec_a = tfidf_a.transform([cleaned_review])
        input_vec_b = tfidf_b.transform([cleaned_review])

        # Predictions
        pred_a = model_a.predict(input_vec_a)[0]
        pred_b = model_b.predict(input_vec_b)[0]

        # Display results
        st.subheader("📊 Prediction Results")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Model A (Balanced)")
            st.metric(label="Predicted Rating", value=str(pred_a))

        with col2:
            st.markdown("### Model B (Imbalanced)")
            st.metric(label="Predicted Rating", value=str(pred_b))

        st.success("Prediction completed successfully!")