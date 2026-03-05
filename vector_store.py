from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_vector_store(chunks):

    embeddings = model.encode(chunks[:50])

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index, embeddings


def search_document(query, index, chunks):

    query_embedding = model.encode([query])

    D, I = index.search(np.array(query_embedding), k=3)

    results = [chunks[i] for i in I[0]]

    return results