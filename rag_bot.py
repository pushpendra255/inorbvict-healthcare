import streamlit as st
import PyPDF2
from rag_utils import build_faiss_index, search_index

def run_rag_bot():
    st.header("Part B — RAG Bot")

    uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    query = st.text_input("Ask a question")

    if uploaded_files and query:
        chunks = []
        for file in uploaded_files:
            if file.name.endswith(".txt"):
                text = file.read().decode("utf-8")
            else:
                reader = PyPDF2.PdfReader(file)
                text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
            chunks.extend(text.split("."))

        index, _ = build_faiss_index(chunks)
        results = search_index(index, query, chunks)

        st.subheader("Top Results")
        for res, score in results:
            st.write(f"• {res} (score: {score:.2f})")
