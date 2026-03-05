import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

@st.cache_resource
def load_simplifier():
    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


tokenizer, model = load_simplifier()


def simplify_document(text):

    if not text or len(text.strip()) == 0:
        return "No text found."

    prompt = "simplify: " + text

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        inputs["input_ids"],
        max_length=120
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return result
