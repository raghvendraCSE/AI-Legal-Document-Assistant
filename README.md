# AI-Legal-Document-Assistant
# 📄 Legal Document AI Assistant

An AI-powered application that helps users **analyze, summarize, and interact with legal documents** using **Natural Language Processing (NLP)** and **Retrieval Augmented Generation (RAG)**.

The system allows users to upload legal documents and automatically generates summaries while enabling an interactive chatbot that answers questions based on the document content.

---

# 🚀 Features

* 📑 **Document Upload** – Upload legal documents in text format
* ✂️ **Automatic Summarization** – Generate concise summaries of long legal documents
* 🤖 **AI Chat with Documents** – Ask questions and get answers from the document
* 🔍 **Semantic Search** – Retrieve relevant sections using vector search
* ⚡ **Fast Processing** – Optimized RAG pipeline for accurate results

---

# 🧠 Technologies Used

* **Python**
* **Streamlit** – Web interface
* **Transformers (Hugging Face)** – NLP models
* **Sentence Transformers** – Text embeddings
* **FAISS** – Vector similarity search
* **PyTorch** – Deep learning framework

---

# 🏗️ Project Architecture

User Uploads Document
↓
Text Preprocessing
↓
Embedding Generation
↓
Vector Database (FAISS)
↓
User Query
↓
Retriever + Language Model
↓
Answer / Summary Displayed in Streamlit

---

# 📂 Project Structure

```
legal-document-ai/
│
├── app.py                # Main Streamlit application
├── summarizer.py         # Document summarization module
├── rag.py                # Retrieval Augmented Generation pipeline
├── utils.py              # Helper functions
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

# ⚙️ Installation

Clone the repository

```
git clone https://github.com/your-username/legal-document-ai.git
```

Move into the project folder

```
cd legal-document-ai
```

Install dependencies

```
pip install -r requirements.txt
```

Run the application

```
streamlit run app.py
```

---

# 🖥️ Application Interface

The application provides a simple interface where users can:

* Upload legal documents
* Generate summaries
* Ask questions about the document
* Receive AI-generated responses

---

# 📊 Use Cases

* Legal document analysis
* Contract review assistance
* Legal research support
* Educational purposes

---

# 🔮 Future Improvements

* Support for **PDF and DOCX documents**
* Improved **legal-specific language models**
* Multi-document comparison
* Cloud deployment

---

# 👨‍💻 Author

**Raghavendra Nath Chaturvedi**
B.Tech Computer Science & Engineering
Babu Banarasi Das Northern India Institute of Technology, Lucknow

---

# ⭐ Contributing

Contributions are welcome. Feel free to fork the repository and submit pull requests.

---

# 📜 License

This project is licensed under the **MIT License**.
