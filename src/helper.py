# src/helper.py

import os
import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import ast
from dotenv import load_dotenv

# ----------------------------
# LLM imports
# ----------------------------
try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    from langchain_community.llms import Ollama

# Initialize Ollama LLM (Gemma 2B)
llm = Ollama(model="gemma:2b")

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
MYSQL_URL = os.getenv("MYSQL_URL")

if not MYSQL_URL:
    raise ValueError("MYSQL_URL environment variable is not set. Please set it.")

engine = create_engine(MYSQL_URL)

# ----------------------------
# Function 1: Load patient documents
# ----------------------------
def load_patient_documents(doctor_name: str):
    """
    Load patient data from MySQL for a specific doctor and convert to LangChain Documents.
    """
    query = "SELECT * FROM medical_data WHERE doctor = %s"

    try:
        df = pd.read_sql(query, engine, params=(doctor_name,))
        if df.empty:
            print(f"⚠️ No data found for doctor: {doctor_name}")
            check_query = "SELECT DISTINCT doctor FROM medical_data LIMIT 10"
            try:
                available_doctors = pd.read_sql(check_query, engine)
                print("Available doctors (sample):", available_doctors)
            except Exception as e:
                print(f"Could not fetch available doctors: {e}")
            return []

        # Convert each row to document
        documents = [
            Document(page_content=str(row.to_dict()).lower())
            for _, row in df.iterrows()
        ]

        # Process documents into clean patient text
        processed_documents = []
        for doc in documents:
            patient = ast.literal_eval(doc.page_content)
            text = (
                f"Patient Name: {patient.get('name', '')}, Age: {patient.get('age', '')}, "
                f"Gender: {patient.get('gender', '')}, Blood Type: {patient.get('blood type', '')}, "
                f"Medical Condition: {patient.get('medical condition', '')}, "
                f"Date of Admission: {patient.get('date of admission', '')}, Doctor: {patient.get('doctor', '')}, "
                f"Hospital: {patient.get('hospital', '')}, Insurance: {patient.get('insurance provider', '')}, "
                f"Billing Amount: {patient.get('billing amount', '')}, Room Number: {patient.get('room number', '')}, "
                f"Admission Type: {patient.get('admission type', '')}, Discharge Date: {patient.get('discharge date', '')}, "
                f"Medication: {patient.get('medication', '')}, Test Results: {patient.get('test results', '')}"
            )
            processed_documents.append(Document(page_content=text))

        return processed_documents

    except Exception as e:
        print(f"❌ Error loading patient data: {e}")
        return []

# ----------------------------
# Function 2: Create FAISS vectorstore
# ----------------------------
def get_vectorstore(documents, doctor_name: str, persist_dir=None):
    """
    Create FAISS vectorstore from documents.
    """
    if not documents:
        print(f"⚠️ No documents provided for doctor: {doctor_name}")
        return None

    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embeddings)

        if persist_dir:
            vectorstore.save_local(persist_dir)
            print(f"✅ Vectorstore saved to '{persist_dir}'")

        return vectorstore

    except Exception as e:
        print(f"❌ Error creating vectorstore: {e}")
        return None

# ----------------------------
# Function 3: Generate answer from context using Ollama LLM
# ----------------------------
def get_answer(llm, context: str, question: str) -> str:
    """
    Generate a contextual answer for a doctor's query using local Ollama Gemma 2B.
    """
    if not context.strip():
        return "⚠️ No patient data found to answer your query."

    prompt = f"""
You are a helpful Medical Assistant for a doctor. Use ONLY the provided patient data below to answer the question. 
Do NOT include any information from other patients or doctors.

Patient Data:
{context}

Question: {question}

Answer in a clear and concise manner.
"""

    try:
        answer = llm.invoke(prompt)
        return answer.strip()
    except Exception as e:
        return f"⚠️ Error generating answer: {str(e)}"

# ----------------------------
# Wrapper function for app.py
# ----------------------------
def get_answer_from_vectorstore(question, vectorstore, top_k=3):
    """
    Retrieve relevant documents from vectorstore and generate answer using Ollama LLM.
    """
    if not vectorstore:
        return "⚠️ Vectorstore is not available."

    try:
        # Retrieve top-k relevant documents
        relevant_docs = vectorstore.similarity_search(question, k=top_k)
        if not relevant_docs:
            return "⚠️ No relevant information found in patient records."

        # Combine retrieved docs into context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Generate answer using existing get_answer function
        answer = get_answer(llm, context, question)
        return answer

    except Exception as e:
        return f"⚠️ Error generating answer: {str(e)}"

# ----------------------------
# Optional: run as script for testing
# ----------------------------
if __name__ == "__main__":
    doctor_name = "Matthew Smith"  # replace with actual doctor
    documents = load_patient_documents(doctor_name)
    vectorstore = get_vectorstore(documents, doctor_name, persist_dir="faiss_index")

    if vectorstore:
        print(f"✅ Vectorstore ready for {doctor_name}")
        # Example query
        context = documents[0].page_content if documents else ""
        question = "What is the patient's medical condition?"
        answer = get_answer(llm, context, question)
        print("Answer:", answer)
    else:
        print(f"❌ Failed to build vectorstore for {doctor_name}")
