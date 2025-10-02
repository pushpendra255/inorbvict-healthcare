import streamlit as st
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# Part A – Flow-Based Chatbot
# -----------------------------
class FlowChatbot:
    def __init__(self):
        self.questions = [
            "👤 What is your name?",
            "🎓 What is your highest qualification?",
            "💼 How many years of work experience do you have?",
            "🤖 Why do you want to join AI/ML at INORBVICT?"
        ]
        self.answers = {}
        self.index = 0

    def current_question(self):
        if self.index < len(self.questions):
            return self.questions[self.index]
        return None

    def submit(self, answer):
        if not answer.strip():
            return False, "⚠️ Please provide an answer."
        self.answers[self.questions[self.index]] = answer
        self.index += 1
        return True, "✅ Answer recorded!"

    def summary(self):
        return "\n".join([f"{q}: {a}" for q, a in self.answers.items()])


# -----------------------------
# Part B – RAG Chatbot
# -----------------------------
class RAGChatbot:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = []
        self.metadata = []
        self.index = None

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
                        "page": i + 1
                    })

        if self.documents:
            embeddings = self.embedder.encode(self.documents, convert_to_numpy=True)
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings)

    def answer(self, query, top_k=3):
        if not self.index:
            return "⚠️ Please upload PDF files first."

        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_vec, top_k)

        results = []
        for idx in I[0]:
            if idx < len(self.documents):
                snippet = self.documents[idx][:500]
                meta = self.metadata[idx]
                results.append(f"📄 **{meta['filename']} (Page {meta['page']})** → {snippet}")

        final_answer = "\n\n".join(results) if results else "❌ No relevant content found."
        return final_answer


# -----------------------------
# Part C – Streamlit UI
# -----------------------------
st.set_page_config(page_title="INORBVICT AIML Assignment", layout="wide")

st.title("🤖 INORBVICT – AIML Assignment Chatbot")
st.write("Select a mode from the dropdown below to test different parts of the assignment.")

mode = st.selectbox("Choose Mode", ["Part A – Flow Chatbot", "Part B – RAG Chatbot"])

# -----------------------------
# UI for Part A
# -----------------------------
if mode == "Part A – Flow Chatbot":
    st.header("🗂️ Flow-Based Chatbot (Guided Q&A)")

    if "flow" not in st.session_state:
        st.session_state.flow = FlowChatbot()

    q = st.session_state.flow.current_question()
    if q:
        st.info(q)
        ans = st.text_input("Your Answer", key=f"flow_q_{st.session_state.flow.index}")
        if st.button("Submit Answer"):
            ok, msg = st.session_state.flow.submit(ans)
            st.success(msg if ok else msg)
    else:
        st.success("🎉 All questions answered!")
        st.markdown("### 📋 Summary")
        st.text(st.session_state.flow.summary())

# -----------------------------
# UI for Part B
# -----------------------------
elif mode == "Part B – RAG Chatbot":
    st.header("📑 Retrieval-Augmented Generation (RAG) Chatbot")

    if "rag" not in st.session_state:
        st.session_state.rag = RAGChatbot()

    uploaded_files = st.file_uploader(
        "Upload up to 20 PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if len(uploaded_files) > 20:
            st.error("⚠️ You can upload maximum 20 files only.")
        else:
            st.session_state.rag.load_pdfs(uploaded_files)
            st.success(f"Loaded {len(uploaded_files)} file(s) successfully!")

    query = st.text_input("Ask a question based on the uploaded PDFs")
    if st.button("Get Answer"):
        if query.strip():
            with st.spinner("🔍 Searching in uploaded documents..."):
                ans = st.session_state.rag.answer(query)
                st.markdown("### ✅ Answer")
                st.write(ans)
        else:
            st.warning("Please enter a question.")
