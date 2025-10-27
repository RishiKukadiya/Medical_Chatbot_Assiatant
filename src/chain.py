try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    # Fallback to deprecated version
    from langchain_community.llms import Ollama

# Initialize Ollama LLM with available model
llm = Ollama(model="gemma:2b")  # using your available gemma:2b model

def get_answer(llm, context: str, question: str) -> str:
    """
    Generate a contextual answer for a doctor's query using local Ollama Gemma 2:B.
    """
    if not context.strip():
        return "⚠️ No patient data found to answer your query."

    # Build prompt for the LLM
    prompt = f"""
You are a helpful Medical Assistant for a doctor. Use ONLY the provided patient data below to answer the question. 
Do NOT include any information from other patients or doctors.

Patient Data:
{context}

Question: {question}

Answer in a clear and concise manner.
"""

    try:
        # Call the local Ollama model using invoke method
        answer = llm.invoke(prompt)
        return answer.strip()
    except Exception as e:
        return f"⚠️ Error generating answer: {str(e)}"

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    context = "Patient Name: John Doe\nAge: 45\nMedical Condition: Hypertension\nMedication: Lisinopril\n"
    question = "What is the patient's medical condition?"

    answer = get_answer(llm, context, question)
    print("Answer:", answer)
