import streamlit as st
from extractor import extract_text
from simplifier import simplify_document
from summarizer import summarize_text
from chunking import split_text
from vector_store import create_vector_store, search_document
from chatbot import answer_question


st.title("📄 AI Legal Document Assistant")


# Chat history storage
if "messages" not in st.session_state:
    st.session_state.messages = []


uploaded_file = st.file_uploader(
    "Upload Legal Document",
    type=["pdf","txt"]
)


if uploaded_file is not None:

    # Extract text
    if uploaded_file.type == "application/pdf":
        text = extract_text(uploaded_file)
    else:
        text = uploaded_file.read().decode()

    # Document statistics
    st.header("📊 Document Statistics")

    word_count = len(text.split())
    sentence_count = len(text.split("."))

    st.write("Total Words:", word_count)
    st.write("Total Sentences:", sentence_count)

    # Show preview
    st.header("📑 Document Preview")
    st.write(text[:800])

    # Split document into chunks
    chunks = split_text(text)

    # Create vector database
    index, embeddings = create_vector_store(chunks)


    # Simplify document
    if st.button("Simplify Document"):

        with st.spinner("Simplifying legal text..."):

            results = []

            for chunk in chunks[:5]:
                  results.append(simplify_document(chunk))

            simplified_text = " ".join(results)

        st.header("🧠 Simplified Text")
        st.write(simplified_text)


    # Generate summary
    if st.button("Generate Summary"):

        with st.spinner("Generating summary..."):

            summary = summarize_text(text)

        st.header("📌 Document Summary")
        st.write(summary)


    # -----------------------
    # CHATBOT SECTION
    # -----------------------

    st.header("💬 Chat With Document")

    # Display chat history
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Chat input
    query = st.chat_input("Ask a question about the document")


    if query:

        # Show user message
        st.chat_message("user").markdown(query)

        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })


        with st.spinner("Searching document..."):

            relevant_chunks = search_document(query, index, chunks)

            context = " ".join(relevant_chunks)

            answer = answer_question(context, query)


        # Show AI message
        with st.chat_message("assistant"):
            st.markdown(answer)


        # Save AI response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []