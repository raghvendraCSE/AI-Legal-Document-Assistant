import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_model():
    model_name = "sshleifer/distilbart-cnn-12-6"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


tokenizer, model = load_model()


def summarize_text(text):

    if not text or len(text.strip()) == 0:
        return "No text found."

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary
