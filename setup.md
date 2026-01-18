# Maarifaa RAG System – Setup Guide

This document explains **step by step**, from start to end, how the Maarifaa Question Answering API was built using **FastAPI, ChromaDB, and Ollama**.

---

## 1. Project Overview

This project implements a **document-based Question Answering system** using the concept of **Retrieval-Augmented Generation (RAG)**.

Instead of answering questions from general internet knowledge, the system strictly answers based on **organization-specific documents** provided by the user. This makes the system suitable for:

* Internal knowledge bases
* Organisation FAQs
* Policy and vision documents
* Chatbots and automation systems

The system exposes a **REST API** that can be consumed by any client such as Postman, frontend applications, or WhatsApp automation platforms.

---

## 2. Technology Stack

### Backend & AI

* **Python 3.10+** – Core programming language
* **FastAPI** – High-performance web framework for building APIs
* **ChromaDB** – Vector database for storing and retrieving embeddings
* **Ollama** – Runs local Large Language Models and embedding models

### Models Used

* **nomic-embed-text** – For generating embeddings
* **LLM (via Ollama)** – For generating final answers

### Tools & Testing

* **Swagger UI** – Built-in API documentation and testing
* **Postman** – Manual API testing tool

---

## 3. Project Structure

```
chromadb/
│
├── chroma.py              # Document ingestion and vector storage
├── chroma_fastapi.py      # FastAPI application with /query endpoint
├── chroma_db/             # Persistent ChromaDB storage
├── maarifaa.txt           # Knowledge document
├── about.txt              # Knowledge document
├── policy.txt             # Knowledge document
└── SETUP.md               # Setup documentation
```

---

## 4. Step 1: Install Required Dependencies

Install all required Python packages:

```bash
pip install fastapi uvicorn chromadb ollama
```

Ensure Python version is 3.10 or above.

---

## 5. Step 2: Install and Run Ollama

1. Download Ollama from: [https://ollama.com](https://ollama.com)
2. Install Ollama
3. Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

4. Ensure Ollama server is running:

```bash
ollama serve
```

Ollama runs by default on:

```
http://localhost:11434
```

---

## 6. Step 3: Prepare Knowledge Files

Create text files containing organisational information:

* `maarifaa.txt`
* `about.txt`
* `policy.txt`

These files act as the **knowledge base** for the AI system.

---

## 7. Step 4: Ingest Documents into ChromaDB (`chroma.py`)

This is a **critical one-time setup step**.

The purpose of this step is to convert raw text documents into numerical vector embeddings so that semantic search can be performed later.

### What happens internally:

1. The script scans all configured `.txt` files
2. Each file is split into logical text chunks
3. Each chunk is converted into an embedding using Ollama
4. Embeddings are stored inside ChromaDB
5. Metadata (source file names) are saved along with each chunk

### Why this is important:

* LLMs cannot directly "search" text
* Vector embeddings enable semantic similarity search
* This step enables accurate context retrieval during queries

Run ingestion script:

```bash
python chroma.py
```

⚠️ This script should be re-run **only when documents change**.

Expected output:

```
✅ Ingested X documents
📦 Collection count: X
```

---

## 8. Step 5: Create FastAPI Application (`chroma_fastapi.py`)

This file exposes the **AI system as a REST API**.

### Responsibilities of this file:

* Initialize FastAPI server
* Load the existing ChromaDB collection
* Define request and response schemas
* Handle user questions
* Orchestrate retrieval and generation

### `/query` Endpoint Workflow:

1. Receives a JSON request containing a question
2. Converts the question into an embedding
3. Performs similarity search in ChromaDB
4. Extracts relevant document chunks
5. Sends context + question to the LLM
6. Returns answer and source references

This separation ensures:

* Clean architecture
* Reusability
* Easy integration with external systems

---

## 9. Step 6: Start the FastAPI Server

Run the server using:

```bash
uvicorn chroma_fastapi:app --reload
```

Server runs at:

```
http://127.0.0.1:8000
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## 10. Step 7: Test API Using Swagger UI

1. Open Swagger UI
2. Select **POST /query**
3. Click **Try it out**
4. Enter request body:

```json
{
  "question": "What is Maarifaa Vision?"
}
```

5. Click **Execute**

Expected response:

* Answer generated from documents
* Source files listed

---

## 11. Step 8: Test API Using Postman

1. Open Postman
2. Select **POST** method
3. Enter URL:

```
http://127.0.0.1:8000/query
```

4. Go to **Body → raw → JSON**
5. Add request:

```json
{
  "question": "What is Maarifaa Vision?"
}
```

6. Click **Send**

A `200 OK` response confirms the API is working.

---

## 12. Performance Notes

* The **first API request** may take significant time due to:

  * Model loading into memory
  * Embedding computation
  * CPU-only execution

* Subsequent requests are significantly faster

### Why this is expected:

* Ollama runs models locally
* No GPU acceleration is used
* No response caching is implemented

In production environments, performance can be improved using:

* GPU-based deployment
* Cached embeddings
* Request batching

---

## 13. Final Outcome

✔ Working RAG-based Question Answering API
✔ Persistent vector database
✔ Local LLM integration
✔ Successfully tested with Swagger and Postman

---

## 14. Future Enhancements

* GPU deployment for faster responses
* Integration with WhatsApp automation (Whatomate)
* UI-based chatbot
* Caching frequent queries

---

## 15. Conclusion

This project demonstrates a **complete end-to-end implementation** of a Retrieval-Augmented Generation system using open-source technologies.

It covers:

* Knowledge ingestion
* Vector search
* AI-powered answer generation
* API exposure
* Testing and validation

The system is production-ready in terms of architecture and can be extended to:

* Chatbots
* WhatsApp automation (e.g., Whatomate)
* Web applications
* Internal enterprise tools


