import streamlit as st
from transformers import pipeline

# Load model only once
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

summarizer = load_model()

def summarize_text(text):

    if not text or len(text.strip()) == 0:
        return "No text found to summarize."

    chunk_size = 700
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    summaries = []

    for chunk in chunks[:5]:   # limit chunks for speed
        result = summarizer(
            chunk,
            max_length=120,
            min_length=30,
            do_sample=False,
            truncation=True
        )

        summaries.append(result[0]["summary_text"])

    return " ".join(summaries)