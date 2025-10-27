# src/doc_loader.py

import os
import pandas as pd
from sqlalchemy import create_engine
from langchain.schema import Document
import ast
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Step 1: Get MySQL URL from environment variable
database_url = os.getenv('MYSQL_URL')

if not database_url:
    print("⚠️  MYSQL_URL environment variable is not set.")

    raise ValueError("MYSQL_URL environment variable is not set. Please set it.")

engine = create_engine(database_url)

# Step 3: Function to load patient data and convert to LangChain Documents
def load_patient_data(doctor_name: str):
    # Safe SQL query using parameterized query to prevent SQL injection
    query = "SELECT * FROM medical_data WHERE doctor = %s"
    
    try:
        df = pd.read_sql(query, engine, params=(doctor_name,))
        print(f"📊 Database query returned {len(df)} rows for doctor: {doctor_name}")
        
        if df.empty:
            print(f"⚠️  No data found for doctor: {doctor_name}")
            print("Available doctors in database:")
            # Let's check what doctors are available
            check_query = "SELECT DISTINCT doctor FROM medical_data LIMIT 10"
            try:
                available_doctors = pd.read_sql(check_query, engine)
                print(available_doctors)
            except Exception as e:
                print(f"Could not fetch available doctors: {e}")
            return []
        
        documents = [
            Document(page_content=str(row.to_dict()).lower())
            for _, row in df.iterrows()
        ]
        
    except Exception as e:
        print(f"❌ Error loading patient data: {e}")
        return []

    # Process documents into clean text
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
