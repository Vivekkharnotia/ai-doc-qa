import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-doc-qa")

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.services.pdf_service import PDFService
from app.services.rag_service import RAGService
import logging
import os

app = FastAPI(title="AI Document QA System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-doc-qa")

pdf_service = PDFService()
rag_service = RAGService()


@app.get("/")
def home():
    return {"message": "AI Doc QA Running 🚀"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info(f"Upload request received: {file.filename}")

        contents = await file.read()

        text = pdf_service.extract_text(contents)

        doc_id = rag_service.create_index(text, file.filename)

        logger.info(f"Document processed successfully: {doc_id}")

        return {
            "message": "Document uploaded successfully",
            "document_id": doc_id
        }

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return {"error": str(e)}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        logger.info(f"Question received: {request.question}")

        retrieved_chunks = rag_service.search(request.question)

        answer = llm_service.generate_answer(
            retrieved_chunks,
            request.question
        )

        logger.info("Answer generated successfully")

        return {
            "answer": answer
        }

    except Exception as e:
        logger.error(f"Ask failed: {str(e)}")
        return {"error": str(e)}

import os

@app.get("/documents")
def list_documents():
    try:
        base_path = os.path.join(os.getcwd(), "app", "storage")

        docs = [
            d for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d)) and d.startswith("doc_")
        ]

        return {
            "total_documents": len(docs),
            "documents": docs
        }

    except Exception as e:
        return {"error": str(e)}