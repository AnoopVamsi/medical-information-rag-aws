import streamlit as st

st.set_page_config(
    page_title="Medical Information Evidence Assistant",
    page_icon="📚",
    layout="centered",
)

st.title("Medical Information Evidence Assistant")

st.write(
    "Explore diabetes education documents and inspect "
    "the published evidence behind answers."
)

st.warning(
    "For medical education only. This application does not provide "
    "personalized diagnoses, medication dosages, or treatment decisions. "
    "Do not enter personal health information."
)

st.info(
    "Setup stage: document retrieval and answer generation "
    "are not connected. Live AWS integration has not been verified."
)

with st.form("question_form"):
    question = st.text_input(
        "What would you like to learn about diabetes?",
        placeholder="For example: What is type 2 diabetes?",
        max_chars=1000,
    )
    submitted = st.form_submit_button("Check setup")

if submitted:
    if not question.strip():
        st.error("Please enter an educational question.")
    else:
        st.success("The interface is working.")
        st.write(
            "Your question has not been sent to an AI model. "
            "We will connect verified source documents in a later stage."
        )