# AI Document Question Answering System (Semantic Search)

## Overview

The AI Document Question Answering System is a Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents and ask questions in natural language. Instead of relying on keyword matching, the system performs semantic search using text embeddings to retrieve the most relevant document content before generating an answer.

This project demonstrates the use of Natural Language Processing (NLP), vector embeddings, semantic similarity search, and transformer-based language models to build an intelligent document assistant.

---

## Features

* Upload PDF documents
* Automatic text extraction and preprocessing
* Intelligent document chunking
* Semantic search using Sentence Transformers
* Vector similarity search with FAISS
* Natural language question answering
* REST API built with FastAPI
* Docker support for deployment

---

## Tech Stack

### Backend

* Python
* FastAPI
* Uvicorn

### Machine Learning & NLP

* Sentence Transformers
* Hugging Face Transformers
* FLAN-T5 Small
* FAISS

### Libraries

* PyPDF2
* NumPy
* Torch
* Pydantic

---

## Project Workflow

1. User uploads a PDF document.
2. Text is extracted from the PDF.
3. The text is divided into smaller chunks.
4. Sentence Transformer generates embeddings for each chunk.
5. Embeddings are stored in a FAISS vector index.
6. When the user asks a question, the query is converted into an embedding.
7. FAISS retrieves the most relevant document chunks.
8. The retrieved context is passed to the language model.
9. The language model generates the final answer.

---

## Project Structure

```text
app/
│── main.py
│── services/
│── tests/

data/
requirements.txt
Dockerfile
README.md
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd ai-doc-qa
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
uvicorn app.main:app --reload
```

Open your browser:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Upload PDF

```
POST /upload
```

Uploads a PDF document and creates a semantic index.

---

### Ask Questions

```
POST /ask
```

Example Request

```json
{
  "question": "Summarize this document."
}
```

---

## Sample Questions

* What is this document about?
* Summarize the document.
* What are the key topics discussed?
* List the technical skills mentioned.
* What projects are included?
* Explain the conclusion of this document.

---

## Future Improvements

* Multiple document support
* Persistent vector database
* Hybrid search
* Cross-encoder re-ranking
* OCR support for scanned PDFs
* Web-based user interface
* Authentication and user management

---

## Author

**Vivek Kharnotia**

M.Tech – Artificial Intelligence & Machine Learning

Lovely Professional University

GitHub: https://github.com/<your-username>
