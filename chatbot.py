import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_qa_model():
    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


tokenizer, model = load_qa_model()


def answer_question(context, question):

    prompt = f"""
    Answer the question based on the legal document.

    Context:
    {context}

    Question:
    {question}
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        inputs["input_ids"],
        max_length=200
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
