from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uuid import uuid4

from app.services.pdf_service import PDFService
from app.services.rag_service import RAGService
from app.services.llm_service import LLMService


app = FastAPI(title="AI Document Assistant API")


# CORS Configuration (Mobile + Browser ready)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize services
pdf_service = PDFService()
rag_service = RAGService()
llm_service = LLMService()


class QuestionRequest(BaseModel):
    document_id: str
    question: str


@app.get("/")
def health_check():
    return {"status": "AI Document Assistant API running"}


@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    try:
        # Generate unique document ID
        document_id = f"doc_{uuid4().hex}"

        # Save uploaded file temporarily
        file_path = pdf_service.save_uploaded_file(file)

        # Extract text from PDF
        text = pdf_service.extract_text(file_path)

        # Create document-specific FAISS index
        chunk_count = rag_service.create_index(text, document_id)

        # Delete temporary file
        pdf_service.delete_file(file_path)

        return {
            "message": "Document uploaded and indexed successfully",
            "document_id": document_id,
            "chunks": chunk_count
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        # Retrieve relevant chunks
        retrieved_chunks = rag_service.search(
            request.question,
            request.document_id
        )

        # Generate final answer
        answer = llm_service.generate_answer(
            retrieved_chunks,
            request.question
        )

        return {
            "document_id": request.document_id,
            "question": request.question,
            "answer": answer
        }

    except Exception as e:
        return {"error": str(e)}

import os


@app.get("/documents")
def list_documents():
    try:
        storage_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "storage"
        )

        if not os.path.exists(storage_path):
            return {"documents": []}

        documents = [
            folder for folder in os.listdir(storage_path)
            if os.path.isdir(os.path.join(storage_path, folder))
        ]

        return {"documents": documents}

    except Exception as e:
        return {"error": str(e)}


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    try:
        doc_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "storage",
            document_id
        )

        if not os.path.exists(doc_path):
            return {"error": "Document not found"}

        # Remove all files inside folder
        for file in os.listdir(doc_path):
            os.remove(os.path.join(doc_path, file))

        # Remove folder
        os.rmdir(doc_path)

        return {"message": f"Document {document_id} deleted successfully"}

    except Exception as e:
        return {"error": str(e)}
