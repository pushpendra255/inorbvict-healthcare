# Main Streamlit interface
import streamlit as st
from flow_bot import run_flow_bot
from rag_bot import run_rag_bot

st.set_page_config(page_title="INORBVICT AIML Assignment", layout="wide")

st.title("INORBVICT Healthcare — AIML Assignment")

mode = st.selectbox("Select Assignment Part", ["Part A - Flow Bot", "Part B - RAG Bot", "Part C - Combined"])

if mode == "Part A - Flow Bot":
    run_flow_bot()

elif mode == "Part B - RAG Bot":
    run_rag_bot()

elif mode == "Part C - Combined":
    st.write("Choose which bot to interact with:")
    sub_mode = st.radio("Bot Type", ["Flow Bot", "RAG Bot"])
    if sub_mode == "Flow Bot":
        run_flow_bot()
    else:
        run_rag_bot()
