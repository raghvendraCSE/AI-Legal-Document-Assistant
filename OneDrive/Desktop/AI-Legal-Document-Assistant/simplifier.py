from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_simplifier():
    model_name = "google/flan-t5-small"
    
    return pipeline(
        task="text2text-generation",
        model=model_name,
        tokenizer=model_name
    )

simplifier = load_simplifier()

def simplify_document(text):

    if not text or len(text.strip()) == 0:
        return "No text found."

    prompt = "simplify: " + text

    result = simplifier(
        prompt,
        max_length=120,
        do_sample=False,
        truncation=True
    )

    return result[0]["generated_text"]