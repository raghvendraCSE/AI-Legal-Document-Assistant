import streamlit as st
from transformers import pipeline

# Load model only once
@st.cache_resource
def load_qa_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )

qa_model = load_qa_model()


def answer_question(context, question):

    prompt = f"""
    Answer the question based on the legal document.

    Context:
    {context}

    Question:
    {question}
    """

    result = qa_model (
        prompt,
        max_length=200,
        truncation=True
    )

    return result[0]["generated_text"]