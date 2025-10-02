import streamlit as st

def run_flow_bot():
    st.header("Part A — Flow-based Guided Chatbot")

    name = st.text_input("What is your name?")
    age = st.number_input("Enter your age", min_value=1, max_value=120)
    gender = st.radio("Select Gender", ["Male", "Female", "Other"])
    symptoms = st.text_area("Describe your symptoms")

    if st.button("Submit"):
        st.success(f"Hello {name}, Age: {age}, Gender: {gender}. Symptoms noted: {symptoms}.")
