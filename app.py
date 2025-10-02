import streamlit as st
import os
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# PART A - Flow Based Chatbot
# --------------------------
def flow_based_chat():
    st.header("🗣️ Flow-Based Chatbot (Part A)")

    if "step" not in st.session_state:
        st.session_state.step = 1
    if "responses" not in st.session_state:
        st.session_state.responses = {}

    # Step 1: Name
    if st.session_state.step == 1:
        name = st.text_input("👉 What is your full name?")
        if st.button("Submit Name"):
            if name.strip() != "":
                st.session_state.responses["Name"] = name
                st.session_state.step = 2
                st.rerun()
            else:
                st.warning("Please enter your name.")

    # Step 2: Age
    elif st.session_state.step == 2:
        age = st.number_input("🎂 How old are you?", min_value=10, max_value=100, step=1)
        if st.button("Submit Age"):
            st.session_state.responses["Age"] = age
            st.session_state.step = 3
            st.rerun()

    # Step 3: Email
    elif st.session_state.step == 3:
        email = st.text_input("📧 What is your email address?")
        if st.button("Submit Email"):
            if "@" in email and "." in email:
                st.session_state.responses["Email"] = email
                st.session_state.step = 4
                st.rerun()
            else:
                st.warning("Please enter a valid email.")

    # Step 4: Skills
    elif st.session_state.step == 4:
        skills = st.text_area("💡 What are your key skills? (comma separated)")
        if st.button("Submit Skills"):
            st.session_state.responses["Skills"] = skills
            st.session_state.step = 5
            st.rerun()

    # Step 5: Experience
    elif st.session_state.step == 5:
        exp = st.radio("📌 How much experience do you have?",
                       ["Fresher", "1-2 years", "3-5 years", "5+ years"])
        if st.button("Submit Experience"):
            st.session_state.responses["Experience"] = exp
            st.session_state.step = 6
            st.rerun()

    # Final Summary
    elif st.session_state.step == 6:
        st.success("✅ Thank you! Here’s your profile summary:")
        st.markdown(f"""
        ### 🎉 Profile Summary
        - **👤 Name:** {st.session_state.responses.get("Name")}
        - **🎂 Age:** {st.session_state.responses.get("Age")}
        - **📧 Email:** {st.session_state.responses.get("Email")}
        - **💡 Skills:** {st.session_state.responses.get("Skills")}
        - **📌 Experience:** {st.session_state.responses.get("Experience")}
        """)
        st.balloons()

        if st.button("🔄 Restart"):
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
    st.header("📚 RAG Chatbot (Part B)")
    st.markdown("Upload up to **20 PDFs**, then ask any question. The bot will answer only from uploaded docs.")

    if "vectorizer" not in st.session_state:
        st.session_state.vectorizer = None
        st.session_state.embeddings = None
        st.session_state.docs = None

    uploaded_files = st.file_uploader("📂 Upload PDF documents", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if len(uploaded_files) > 20:
            st.error("⚠️ You can upload maximum 20 PDFs.")
            return

        texts = extract_text_from_pdfs(uploaded_files)
        vectorizer, embeddings = build_vectorstore(texts)
        st.session_state.vectorizer = vectorizer
        st.session_state.embeddings = embeddings
        st.session_state.docs = texts
        st.success("✅ Documents processed successfully! Now ask your question below.")

    query = st.text_input("💭 Ask a question about your uploaded documents:")
    if query and st.session_state.vectorizer is not None:
        q_vec = st.session_state.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, st.session_state.embeddings).flatten()
        idx = sims.argmax()
        answer = st.session_state.docs[idx][:600] + "..."  # snippet
        st.markdown(f"### 📝 Answer\n{answer}")
    elif query and st.session_state.vectorizer is None:
        st.warning("⚠️ Please upload PDF(s) first.")

# --------------------------
# PART C - Free Chat Interface
# --------------------------
def free_chat():
    st.header("💬 Chat Interface (Part C)")
    st.write("This is a simple free-form chatbot where you can talk casually.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("📝 Type your message:")
    if st.button("Send"):
        if user_input.strip():
            st.session_state.chat_history.append(("You", user_input))
            # simple echo bot (can be replaced with LLM integration)
            response = f"I understood your message: **{user_input}**"
            st.session_state.chat_history.append(("Bot", response))

    if st.session_state.chat_history:
        for role, msg in st.session_state.chat_history:
            if role == "You":
                st.markdown(f"**🧑‍💻 You:** {msg}")
            else:
                st.markdown(f"**🤖 Bot:** {msg}")

# --------------------------
# MAIN APP - Dropdown
# --------------------------
def main():
    st.set_page_config(page_title="INORBVICT AIML Assignment", layout="wide")
    st.title("🚀 INORBVICT – AIML Assignment Chatbot")

    option = st.sidebar.selectbox("🔽 Select Mode", 
                                  ["Flow Mode (Part A)", "RAG Mode (Part B)", "Chat Interface (Part C)"])

    if option == "Flow Mode (Part A)":
        flow_based_chat()
    elif option == "RAG Mode (Part B)":
        rag_chatbot()
    elif option == "Chat Interface (Part C)":
        free_chat()

if __name__ == "__main__":
    main()
