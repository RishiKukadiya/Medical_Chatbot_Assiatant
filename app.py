# app.py

import streamlit as st
from streamlit_chat import message
from src.helper import load_patient_documents, get_vectorstore, get_answer_from_vectorstore

# ----------------------------
# Streamlit setup
# ----------------------------
st.set_page_config(page_title="Medical Chatbot", page_icon="💊", layout="centered")
st.title("💬 Medical Chatbot - Doctor Assistant")

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "bot", "content": "Hello! I am your medical assistant. How can I help you today?"}
    ]

# Sidebar input for doctor name
doctor_name = st.sidebar.text_input("Enter Doctor Name:")

# User question input
user_input = st.text_input("Type your question here:")

# ----------------------------
# Chatbot function
# ----------------------------
def chatbot_response(question, doctor, top_k=3):
    """
    Generate chatbot response using patient documents and vectorstore.
    """
    if not doctor:
        return "Please enter a doctor's name in the sidebar."

    # Step 1: Load patient documents
    documents = load_patient_documents(doctor)
    if not documents:
        return f"⚠️ No patient data found for Dr. {doctor}."

    # Step 2: Build or load vectorstore
    vector_store = get_vectorstore(documents, doctor)
    if not vector_store:
        return f"⚠️ Failed to create vectorstore for Dr. {doctor}."

    # Step 3: Get answer using helper wrapper
    return get_answer_from_vectorstore(question, vector_store, top_k=top_k)

# ----------------------------
# Streamlit interaction
# ----------------------------
if user_input:
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Get bot reply
    bot_reply = chatbot_response(user_input, doctor_name)

    # Add bot reply to chat history
    st.session_state["messages"].append({"role": "bot", "content": bot_reply})

# Display chat messages in order
for i, msg in enumerate(st.session_state.messages):
    message(
        msg["content"],
        is_user=(msg["role"] == "user"),
        key=f"message_{i}"  # unique key for each message
    )
