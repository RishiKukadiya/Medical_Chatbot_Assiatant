# src/embeddings_store.py

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path to allow imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.doc_loader import load_patient_data

# -----------------------------
# Function to create embeddings and FAISS vectorstore
# -----------------------------
def create_embeddings_and_vectorstore(documents, doctor_name: str, persist_dir=None):
    """
    Create embeddings and FAISS vectorstore from documents.
    
    Args:
        documents: List of Document objects
        doctor_name: Name of the doctor (used for caching)
        persist_dir: Directory to save/load vectorstore (optional)
    
    Returns:
        FAISS vectorstore object or None if no documents
    """
    # Check if documents list is empty
    if not documents or len(documents) == 0:
        print(f"⚠️  No documents provided for doctor: {doctor_name}")
        print("Cannot create vectorstore with empty document list.")
        return None
    
    try:
        # Step 1: Initialize HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Step 2: Create FAISS vectorstore from documents
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        # Step 3: Save vectorstore locally if persist_dir is provided
        if persist_dir:
            vectorstore.save_local(persist_dir)
            print(f"✅ Vectorstore built and saved safely to '{persist_dir}'")
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error creating vectorstore: {e}")
        return None

# -----------------------------
# Function to build FAISS vectorstore (legacy function)
# -----------------------------
def build_vectorstore(doctor_name: str, persist_dir="faiss_index"):
    # Step 1: Load documents from MySQL
    docs = load_patient_data(doctor_name)

    # Step 2: Create vectorstore using the main function
    vectorstore = create_embeddings_and_vectorstore(docs, doctor_name, persist_dir)
    
    if vectorstore is None:
        print(f"❌ Failed to create vectorstore for doctor: {doctor_name}")
        return None
    
    return vectorstore

# -----------------------------
# Optional: run directly for testing
# -----------------------------
if __name__ == "__main__":
    # Use one of the actual doctor names from the database
    doctor = "Matthew Smith"  # or try "Kevin Wells", "Daniel Ferguson", etc.
    vectorstore = build_vectorstore(doctor)
    
    if vectorstore:
        print(f"✅ Successfully created vectorstore for {doctor}")
    else:
        print(f"❌ Failed to create vectorstore for {doctor}")
