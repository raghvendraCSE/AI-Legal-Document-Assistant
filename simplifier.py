from transformers import pipeline

# Load simplification model
simplifier = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

def simplify_document(text):

    # Safety check
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
