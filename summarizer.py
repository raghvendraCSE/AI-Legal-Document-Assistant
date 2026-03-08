import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

summarizer = load_model()

def summarize_text(text):

    text = text[:2000]

    result = summarizer(
        text,
        max_length=120,
        min_length=30,
        do_sample=False
    )

    return result[0]["summary_text"]
