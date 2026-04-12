# 📄 AI Document QA System (RAG-Based)

## 🚀 Overview
This project implements a Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and ask questions. It retrieves relevant chunks using FAISS and generates answers using FLAN-T5.

## 🧠 Features
- PDF upload & processing
- Semantic search (SentenceTransformers + FAISS)
- Answer generation (FLAN-T5)
- Multi-document support
- FastAPI backend
- Docker support

## ⚙️ Tech Stack
Python, FastAPI, FAISS, Transformers, Docker

## ▶️ Run
uvicorn app.main:app --reload

Open: http://127.0.0.1:8000/docs