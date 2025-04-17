# pdf_qa.py
import os
import tempfile
from pypdf import PdfReader
import openai

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, pdf_file.name)
    
    with open(temp_path, "wb") as f:
        f.write(pdf_file.getvalue())
    
    reader = PdfReader(temp_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # Clean up
    os.remove(temp_path)
    return text

def chunk_text(text, max_chunk_size=4000):
    """Split text into chunks suitable for sending to an LLM"""
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += paragraph
        else:
            chunks.append(current_chunk)
            current_chunk = paragraph
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def verify_northeastern_content(text):
    """Use an LLM to check if the document is related to Northeastern University"""
    # Take a sample of the document to keep token usage reasonable
    sample = text[:3000]  # First 3000 characters
    
    prompt = f"""
    Below is an excerpt from a document. Your task is to determine if this document is related to Northeastern University 
    or its Mechanical and Industrial Engineering (MIE) department.
    
    Document excerpt:
    {sample}
    
    Based only on this excerpt, is this document related to Northeastern University? 
    Answer with ONLY "yes" or "no".
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=10
        )
        
        answer = response["choices"][0]["message"]["content"].strip().lower()
        return "yes" in answer
    except Exception as e:
        print(f"Error verifying document content: {e}")
        # If there's an error, err on the side of caution and return False
        return False

def process_pdf(pdf_file):
    """Process a PDF file and verify if it's Northeastern-related"""
    print(f"Processing PDF: {pdf_file.name}")
    text = extract_text_from_pdf(pdf_file)
    print(f"Extracted {len(text)} characters of text")
    
    # Verify if the document is related to Northeastern using LLM
    is_northeastern_related = verify_northeastern_content(text)
    print(f"Document is Northeastern-related: {is_northeastern_related}")
    
    # Only chunk the text if it's Northeastern-related to save processing
    chunks = chunk_text(text) if is_northeastern_related else []
    if is_northeastern_related:
        print(f"Split into {len(chunks)} chunks")
    
    return {
        "filename": pdf_file.name,
        "text": text,
        "chunks": chunks,
        "is_northeastern_related": is_northeastern_related
    }

def answer_question(question, pdf_data):
    """Use LLM to answer a question based on the PDF content"""
    if not pdf_data:
        return "No PDF data available. Please upload a document first."
    
    # Check if the document is related to Northeastern University
    if not pdf_data.get("is_northeastern_related", False):
        return "I do not answer questions unrelated to Northeastern University. This document does not appear to be related to Northeastern or its departments. I can only assist with Northeastern-related inquiries."
    
    if not pdf_data.get("chunks"):
        return "Unable to process the document content. Please try uploading a different document."
    
    # Combine chunks into a single prompt if possible, or use the most relevant chunks
    context = "\n\n".join(pdf_data["chunks"][:3])  # Use first 3 chunks as a simple approach
    
    prompt = f"""
    You are an assistant helping answer questions about a document related to Northeastern University.

    Document: {pdf_data['filename']}
    Content: {context}

    User Question: {question}

    Please answer the question based only on the information provided in the document. 
    If the answer isn't in the document, simply state that you cannot find the information in the document.
    Include references to specific parts of the document that support your answer.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error getting LLM response: {e}")
        return f"I encountered an error processing your question about the document: {str(e)}"