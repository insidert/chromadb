from fastapi import FastAPI
from pydantic import BaseModel
from ollama import Client
from chroma import get_collection

# -------------------- SETUP --------------------
app = FastAPI()
ollama = Client(host="http://localhost:11434")
collection = get_collection()

# -------------------- REQUEST MODEL --------------------
class QueryRequest(BaseModel):
    question: str

# -------------------- QUERY ENDPOINT --------------------
@app.post("/query")
def query_rag(request: QueryRequest):
    question = request.question

    results = collection.query(
        query_texts=[question],
        n_results=5
    )

    docs = results["documents"][0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return {
            "question": question,
            "answer": "I dont have that information in the provided documents."
        }

    context = "\n\n".join(docs)[:4000]

    prompt = f"""
Answer ONLY using the context below.
If the answer is not found, say:
"I dont have that information in the provided documents."

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": "You are a strict, factual assistant."},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.2}
    )

    sources = list({
        md.get("source", "unknown")
        for md in metas
        if isinstance(md, dict)
    })

    return {
        "question": question,
        "answer": response["message"]["content"],
        "sources": sources
    }

# -------------------- START SERVER --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
