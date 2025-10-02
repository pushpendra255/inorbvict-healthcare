# 🚀 INORBVICT – AIML Assignment Chatbot

This repository contains the submission for the **AIML Assignment** given by **INORBVICT Healthcare India Pvt. Ltd.**  

It demonstrates a **multi-mode chatbot system** built with **Streamlit** and **Groq LLaMA-3 API**, covering all three assignment parts:

---

## 🧩 Features

### 🔹 Part A – Flow-Based Chatbot
- Guided, step-by-step questionnaire.  
- Collects user details (Name, Age, Email, Skills, Experience).  
- Validates inputs (e.g., Age must be numeric, Email must be valid).  
- Generates a professional **Candidate Profile Summary** at the end.  

**Example Output:**

✅ Candidate Profile

Name: Pushpendra Singh bhadauriya 

Age: 25

Email: Pushpendra@test.com

Skills: Python, Machine Learning

Experience: 1–2 years


---

### 🔹 Part B – RAG Chatbot
- Implements **Retrieval-Augmented Generation (RAG)** for answering document-based queries.  
- Workflow:
  1. Upload up to 20 PDF files.  
  2. Extracts text with PyPDF2.  
  3. Builds TF-IDF vector store.  
  4. Retrieves most relevant content.  
  5. Sends context + query to **Groq API (LLaMA-3.1)**.  
  6. Falls back to Groq API if no relevant match is found.  

**Example Question:**  
“What are the key challenges modern primary health care faces?”

---

### 🔹 Part C – Free Chat Interface
- A free-form chatbot interface.  
- Works like a **normal conversational bot**.  
- Maintains conversation history.  
- Uses **Groq API** for all responses.  

**Example Question:**  
“What is machine learning in simple words?”

---

## 🛠️ Tech Stack
- **Python 3.10+**  
- **Streamlit** → UI framework  
- **PyPDF2** → PDF text extraction  
- **scikit-learn** → TF-IDF vectorization + cosine similarity  
- **Groq API (LLaMA-3.1)** → LLM responses  
- **NumPy, Pandas** → Data handling  

---

## 📂 Project Structure

├── app.py             # Main entry point with all modes ├── flow_bot.py        # Flow-based chatbot logic (Part A) ├── rag_bot.py         # RAG chatbot logic (Part B) ├── rag_utils.py       # Helper functions for RAG ├── requirements.txt   # Dependencies └── README.md          # Documentation

---

## ⚙️ Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pushpendra255/inorbvict-healthcare.git
   cd inorbvict-healthcare

2. Install dependencies:

pip install -r requirements.txt


3. Add your Groq API Key in app.py:

groq_api_key = "YOUR_API_KEY"


4. Run the app:

streamlit run app.py




---

🔍 Usage

Part A – Flow Mode

Fill in details step by step.

Get final Candidate Profile Summary.


Part B – RAG Mode

Upload PDFs.

Ask domain-specific questions.

Bot answers from your documents, with Groq fallback.


Part C – Free Chat

Chat casually with the bot.

Example:

You: “Tell me a fun fact about AI.”

Bot: “The term Artificial Intelligence was coined in 1956 at the Dartmouth Conference.”




---

📊 Example Test Questions

Flow Mode:

Name = Pushpendra Singh bhadauriya, Age = 25, Email = Pushpendra@test.com, Skills = Python, ML, Experience = 1–2 years → Generates profile summary.


RAG Mode (with healthcare PDFs):

“What are the key challenges that modern primary health care faces?”

“Why is integrating public health into primary care important?”


Free Chat Mode:

“What is machine learning in simple words?”

“Tell me a fun fact about AI.”



---

🌐 Deployment Links

Live App: https://inorbvict-healthcare.streamlit.app/

GitHub Repo: https://github.com/pushpendra255/inorbvict-healthcare

Output video - https://drive.google.com/file/d/1ub5MDho9uKNvnVxirWiK7Z7U6z0gOinN/view?usp=drivesdk


---

📌 Notes

RAG accuracy depends on the quality of uploaded documents.

Free Chat mode responses depend on Groq API quality.

Flow mode is limited to basic candidate profile fields (can be extended).



---

👤 Author

Pushpendra Singh
AIML Developer Candidate

