import chromadb
import uuid
import os
# use local Chroma client (persistent DB stored by Chroma)
client = chromadb.Client()
collection = client.get_or_create_collection(name="maarifaa")
# list of files to ingest
files = ["maarifaa.txt", "about.txt", "policy.txt"]
documents = []
metadatas = []
for fname in files:
    if os.path.exists(fname):
        with open(fname, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                # split into paragraphs by blank lines to create smaller passages
                parts = [p.strip() for p in content.split("\n\n") if p.strip()]
                for part in parts:
                    documents.append(part)
                    metadatas.append({"source": fname})
    else:
        print(f"Warning: {fname} not found, skipping")

if not documents:
    print("No documents found to ingest. Please add .txt files and retry.")
else:
    # Only add if the collection is empty. Re-run logic or deduplication can be added later.
    if collection.count() == 0:
        collection.add(
            ids=[str(uuid.uuid4()) for _ in documents],
            documents=documents,
            metadatas=metadatas
        )
questions = [
    "What is maarifaa?",
    "what is maarifaa vision?",
    "what is maarifaa policy?"
]

results = collection.query(
    query_texts=questions,
    n_results=5
)

for i, docs in enumerate(results["documents"]):
    print(f"\n Question {i+1}: {questions[i]}")
    print("Relevant Information:")
    for j, doc in enumerate(docs, start=1):
        source = "unknown"
        if results.get("metadatas") and len(results["metadatas"]) > i and len(results["metadatas"][i]) >= j:
            md = results["metadatas"][i][j-1]
            source = md.get("source", "unknown") if isinstance(md, dict) else str(md)
        print(f"  {j}. {doc} (source: {source})")
