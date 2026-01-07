import chromadb
import uuid
import os
from ollama import Client

# -----------------------------
# 1. Setup Ollama Client
# -----------------------------
ollama = Client(host="http://localhost:11434")

# -----------------------------
# 2. Setup Chroma (local, persistent)
# -----------------------------
client = chromadb.Client()
collection = client.get_or_create_collection(name="maarifaa")

# -----------------------------
# 3. Ingest text files
# -----------------------------
files = ["maarifaa.txt", "about.txt", "policy.txt"]
documents = []
metadatas = []

for fname in files:
    if os.path.exists(fname):
        with open(fname, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                # Split by blank lines
                parts = [p.strip() for p in content.split("\n\n") if p.strip()]
                for part in parts:
                    documents.append(part)
                    metadatas.append({"source": fname})
    else:
        print(f"Warning: {fname} not found, skipping")

# Add documents only once
if documents and collection.count() == 0:
    collection.add(
        ids=[str(uuid.uuid4()) for _ in documents],
        documents=documents,
        metadatas=metadatas
    )
    print(f"Ingested {len(documents)} documents.")
elif collection.count() > 0:
    print("Documents already ingested. Skipping ingestion.")
else:
    print("No documents found to ingest.")

# -----------------------------
# 4. Questions
# -----------------------------
questions = [
    # "What is maarifaa?",
    "What is maarifaa vision?",
    # "What is maarifaa policy?"
]

# -----------------------------
# 5. Query + LLM (One question at a time)
# -----------------------------
for question in questions:
    results = collection.query(
        query_texts=[question],
        n_results=5
    )

    docs = results["documents"][0]
    metas = results.get("metadatas", [[]])[0]

    print("\n--------------------------------")
    print(f"Question: {question}")

    if not docs:
        print("Answer: I dont have that information in the provided documents.")
        continue

    # Build context safely
    context = "\n\n".join(docs)[:4000]

    prompt = f"""
You are a customer support assistant.
Answer ONLY using the context below.
If the answer is not found, say:
"I dont have that information in the provided documents."
Do not add external knowledge.

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
        options={
            "temperature": 0.2
        }
    )

    answer = response["message"]["content"]
    print(f"Answer: {answer}")

    # Print sources
    sources = set(
        md.get("source", "unknown")
        for md in metas
        if isinstance(md, dict)
    )

    if sources:
        print(f"Sources: {', '.join(sources)}")

    print("--------------------------------")
