import streamlit as st
import os
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# PART A - Flow Based Chatbot
# --------------------------
def flow_based_chat():
    st.subheader("🗣️ Flow-Based Chatbot (Part A)")
    st.caption("Fill in your details step by step, and get a professional profile summary.")

    if "step" not in st.session_state:
        st.session_state.step = 1
    if "responses" not in st.session_state:
        st.session_state.responses = {}

    if st.session_state.step == 1:
        name = st.text_input("👉 Full Name")
        if st.button("Next ➡️"):
            if name.strip():
                st.session_state.responses["Name"] = name
                st.session_state.step = 2
                st.rerun()
            else:
                st.warning("Please enter your name.")

    elif st.session_state.step == 2:
        age = st.number_input("🎂 Age", min_value=10, max_value=100, step=1)
        if st.button("Next ➡️"):
            st.session_state.responses["Age"] = age
            st.session_state.step = 3
            st.rerun()

    elif st.session_state.step == 3:
        email = st.text_input("📧 Email Address")
        if st.button("Next ➡️"):
            if "@" in email and "." in email:
                st.session_state.responses["Email"] = email
                st.session_state.step = 4
                st.rerun()
            else:
                st.warning("Enter a valid email.")

    elif st.session_state.step == 4:
        skills = st.text_area("💡 Key Skills (comma separated)")
        if st.button("Next ➡️"):
            st.session_state.responses["Skills"] = skills
            st.session_state.step = 5
            st.rerun()

    elif st.session_state.step == 5:
        exp = st.radio("📌 Experience", ["Fresher", "1-2 years", "3-5 years", "5+ years"])
        if st.button("Finish ✅"):
            st.session_state.responses["Experience"] = exp
            st.session_state.step = 6
            st.rerun()

    elif st.session_state.step == 6:
        st.success("✅ Profile Completed")
        st.markdown(f"""
        ### 🎉 Candidate Profile
        - **👤 Name:** {st.session_state.responses.get("Name")}
        - **🎂 Age:** {st.session_state.responses.get("Age")}
        - **📧 Email:** {st.session_state.responses.get("Email")}
        - **💡 Skills:** {st.session_state.responses.get("Skills")}
        - **📌 Experience:** {st.session_state.responses.get("Experience")}
        """)
        st.balloons()
        if st.button("🔄 Restart Form"):
            st.session_state.step = 1
            st.session_state.responses = {}
            st.rerun()

# --------------------------
# PART B - RAG Chatbot
# --------------------------
def extract_text_from_pdfs(uploaded_files):
    all_texts = []
    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            try:
                text += page.extract_text() + " "
            except:
                continue
        all_texts.append(text.strip())
    return all_texts

def build_vectorstore(texts):
    vectorizer = TfidfVectorizer(stop_words="english")
    embeddings = vectorizer.fit_transform(texts)
    return vectorizer, embeddings

def rag_chatbot():
    st.subheader("📚 RAG Chatbot (Part B)")
    st.caption("Upload documents and ask focused questions. Answers are based only on your uploaded content.")

    if "vectorizer" not in st.session_state:
        st.session_state.vectorizer = None
        st.session_state.embeddings = None
        st.session_state.docs = None

    uploaded_files = st.file_uploader("📂 Upload up to 20 PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if len(uploaded_files) > 20:
            st.error("⚠️ You can upload maximum 20 PDFs.")
            return
        texts = extract_text_from_pdfs(uploaded_files)
        vectorizer, embeddings = build_vectorstore(texts)
        st.session_state.vectorizer = vectorizer
        st.session_state.embeddings = embeddings
        st.session_state.docs = texts
        st.success("✅ Documents processed. You may now ask a question.")

    query = st.text_input("💭 Your Question")
    if st.button("Submit Question"):
        if query and st.session_state.vectorizer is not None:
            q_vec = st.session_state.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, st.session_state.embeddings).flatten()
            idx = sims.argmax()

            if sims[idx] > 0.2:  # threshold for relevance
                answer = st.session_state.docs[idx][:600] + "..."
                st.markdown("### 📝 Answer")
                st.info(answer)
            else:
                st.warning("⚠️ No relevant answer found in your uploaded documents.")
        elif query and st.session_state.vectorizer is None:
            st.warning("⚠️ Upload PDF(s) first.")

# --------------------------
# PART C - Free Chat Interface
# --------------------------
def free_chat():
    st.subheader("💬 Free Chatbot (Part C)")
    st.caption("Chat casually with the bot. It remembers your conversation like a real chat.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("📝 Type your message and press Send")

    if st.button("Send"):
        if user_input.strip():
            # Add user message
            st.session_state.chat_history.append(("user", user_input))

            # Bot response
            msg = user_input.lower()
            if "hello" in msg:
                response = "Hi 👋! Nice to meet you."
            elif "how are you" in msg:
                response = "I'm doing great, thanks for asking! 😊"
            elif "bye" in msg:
                response = "Goodbye! Take care 👋"
            else:
                response = f"I understood: **{user_input}**"

            st.session_state.chat_history.append(("bot", response))

    # Display chat in reverse (latest at bottom like real chat)
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(
                f"<div style='text-align:right;background:#DCF8C6;padding:10px;border-radius:10px;margin:5px 0;'><b>You:</b> {msg}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='text-align:left;background:#E8EAF6;padding:10px;border-radius:10px;margin:5px 0;'><b>Bot:</b> {msg}</div>",
                unsafe_allow_html=True
            )

# --------------------------
# MAIN APP
# --------------------------
def main():
    st.set_page_config(page_title="INORBVICT AIML Assignment", layout="wide", page_icon="🚀")
    st.title("🚀 INORBVICT – AIML Assignment Chatbot")
    st.markdown("---")

    option = st.sidebar.radio("🔽 Select Mode", 
                              ["Flow Mode (Part A)", "RAG Mode (Part B)", "Chat Interface (Part C)"])

    if option == "Flow Mode (Part A)":
        flow_based_chat()
    elif option == "RAG Mode (Part B)":
        rag_chatbot()
    elif option == "Chat Interface (Part C)":
        free_chat()

if __name__ == "__main__":
    main()
