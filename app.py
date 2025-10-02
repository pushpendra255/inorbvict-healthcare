import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
import numpy as np

# -------------------- API CONFIG --------------------
groq_api_key = "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"
groq_url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {groq_api_key}",
    "Content-Type": "application/json"
}

# -------------------- EMBEDDING MODEL --------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------- Dummy PDF Data (Health Domain) --------------------
pdf_data = {
    "doc1": "A balanced diet includes fruits, vegetables, whole grains, and lean proteins. Drinking enough water is also crucial for overall health.",
    "doc2": "Regular physical activity such as walking, jogging, or yoga improves cardiovascular health, strengthens muscles, and reduces stress.",
    "doc3": "Mental health is equally important as physical health. Practices like meditation, adequate sleep, and social connection improve well-being.",
    "doc4": "Diabetes management requires regular monitoring of blood sugar, following a low-sugar diet, exercising, and taking prescribed medications.",
    "doc5": "Good hygiene practices such as handwashing, safe food handling, and vaccination help prevent infectious diseases."
}

# -------------------- HELPER FUNCTIONS --------------------
def semantic_search(query, pdf_data, threshold=0.6):
    query_embedding = model.encode([query])[0]
    best_doc, best_score = None, -1
    for _, text in pdf_data.items():
        text_embedding = model.encode([text])[0]
        score = np.dot(query_embedding, text_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding)
        )
        if score > best_score:
            best_doc, best_score = text, score
    if best_score >= threshold:
        return best_doc
    return None

def ask_groq(query):
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7
    }
    response = requests.post(groq_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return "⚠️ Groq API error. Please try again."

# -------------------- STREAMLIT APP --------------------
st.set_page_config(page_title="Health Research & Chatbot", layout="wide")
st.title("🏥 Health Document Research & Chatbot System")

# Sidebar navigation
mode = st.sidebar.radio("Choose Mode:", [
    "Part A - Single Q&A",
    "Part B - Multi Q&A",
    "Part C - Chatbot"
])

# -------------------- PART A --------------------
if mode == "Part A - Single Q&A":
    st.subheader("🔹 Part A: Single Question Answering")
    query = st.text_input("Enter your health-related question:")

    if query:
        pdf_answer = semantic_search(query, pdf_data)
        if pdf_answer:
            st.success(f"📄 Answer from PDF: {pdf_answer}")
        else:
            st.info("❌ No relevant answer found in PDF. Using AI...")
            st.write(ask_groq(query))

# -------------------- PART B --------------------
elif mode == "Part B - Multi Q&A":
    st.subheader("🔹 Part B: Multi Question Answering")

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    query = st.text_input("Ask your health question here:")

    if st.button("Get Answer"):
        if query:
            pdf_answer = semantic_search(query, pdf_data)
            if pdf_answer:
                answer = f"📄 From PDF: {pdf_answer}"
            else:
                answer = f"❌ No relevant answer found in PDF.\n\n🤖 AI Suggestion: {ask_groq(query)}"
            st.session_state.qa_history.append((query, answer))

    # Show all asked Q&A
    for q, a in st.session_state.qa_history:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")

# -------------------- PART C --------------------
elif mode == "Part C - Chatbot":
    st.subheader("🔹 Part C: ChatGPT-like Health Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.chat_input("Ask me anything about health...")

    if user_query:
        # First try PDF search
        pdf_answer = semantic_search(user_query, pdf_data)
        if pdf_answer:
            bot_response = f"📄 From PDF: {pdf_answer}"
        else:
            bot_response = f"🤖 {ask_groq(user_query)}"

        # Save chat
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

    # Show chat history like real chatbot
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])
