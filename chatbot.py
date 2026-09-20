# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# @st.cache_resource
# def load_qa_model():
#     model_name = "google/flan-t5-small"

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     return tokenizer, model


# tokenizer, model = load_qa_model()


# def answer_question(context, question):

#     prompt = f"""
#     Answer the question based on the legal document.

#     Context:
#     {context}

#     Question:
#     {question}
#     """

#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

#     outputs = model.generate(
#         inputs["input_ids"],
#         max_length=200
#     )

#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_qa_model():
    model_name = "google/flan-t5-base"   # better than small; use flan-t5-small if memory is low
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_qa_model()

def answer_question(context, question):
    # Question goes FIRST so it is never cut off by truncation
    prompt = (
        f"Question: {question}\n\n"
        f"Context: {context}\n\n"
        "Answer the question using only the context above. "
        "If the answer is not in the context, say 'I could not find this in the document.'"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=150, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
