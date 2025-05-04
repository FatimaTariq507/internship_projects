import streamlit as st
import pickle
from preprocess import clean_text

# Load model and vectorizer
model, vectorizer = pickle.load(open("model.pkl", "rb"))

st.title("🎬 Movie Review Sentiment Analyzer")
review = st.text_area("Enter your movie review:")

if st.button("Analyze Sentiment"):
    cleaned = clean_text(review)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    st.write("Sentiment:", "Positive 😊" if prediction == 1 else "Negative 😞")
