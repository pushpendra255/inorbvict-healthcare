import os
import streamlit as st
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGChatbot:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = []
        self.index = None
        self.metadata = []

    def load_pdfs(self, uploaded_files):
        self.documents = []
        self.metadata = []
        
        for file in uploaded_files:
            reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    self.documents.append(text)
                    self.metadata.append({
                        "filename": file.name,
                        "page": i+1
                    })

        # Build FAISS index
        embeddings = self.embedder.encode(self.documents, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def answer(self, query, top_k=3):
        if not self.index:
            return "⚠️ Pehle PDF files upload karo."

        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_vec, top_k)

        results = []
        for idx in I[0]:
            if idx < len(self.documents):
                snippet = self.documents[idx][:500]  # snippet 500 chars max
                meta = self.metadata[idx]
                results.append(f"📄 **{meta['filename']} (Page {meta['page']})** → {snippet}")

        final_answer = "\n\n".join(results)
        return final_answer
