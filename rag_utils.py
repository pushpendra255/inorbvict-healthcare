import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts):
    return model.encode(texts, convert_to_numpy=True)

def build_faiss_index(chunks):
    embeddings = embed_texts(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def search_index(index, query, chunks, top_k=3):
    q_emb = embed_texts([query])
    D, I = index.search(q_emb, top_k)
    return [(chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]
