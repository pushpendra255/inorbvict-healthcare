# AIML Assignment — INORBVICT HEALTHCARE

This project implements the 3-part assignment:
- Part A — Flow-based guided chatbot with validation & final summary.
- Part B — RAG (Retrieval-Augmented Generation): upload PDF/TXT, chunk & index, semantic retrieval + generation with citations.
- Part C — Single Streamlit interface to choose modes.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment notes
- On **Streamlit Cloud**, select this repo and `app.py` as entrypoint.
- First run may take time due to model downloads.
