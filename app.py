# app.py
import streamlit as st
from src.doc_loader import load_patient_data
from src.embeddings_store import create_embeddings_and_vectorstore
from src.chain import get_answer, llm

# -------------------------------
st.title("Medical Chatbot - Doctor Assistant")

doctor_name = st.text_input("Enter Doctor Name:").lower()
user_question = st.text_input("Enter your question:").lower()

if st.button("Get Answer"):
    if not doctor_name or not user_question:
        st.warning("Please enter both doctor name and question.")
    else:
        try:
            # Load patient data
            documents = load_patient_data(doctor_name)

            # Create or load vectorstore
            vector_store = create_embeddings_and_vectorstore(documents, doctor_name)


            # Retrieve context for the question
            retriever = vector_store.as_retriever()
            retrieved_docs = retriever.get_relevant_documents(user_question)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Get LLM answer
            answer = get_answer(llm, context, user_question)
            st.success(answer)
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
