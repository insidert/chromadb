import chromadb
import uuid
import os
from chromadb.utils import embedding_functions
from chromadb.config import Settings

# -----------------------------
# EMBEDDINGS (MANDATORY)
# -----------------------------
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434"
)

# -----------------------------
# PERSISTENT CHROMA CLIENT
# -----------------------------
client = chromadb.Client(
    Settings(
        persist_directory="chroma_db"
    )
)

# -----------------------------
# DELETE OLD COLLECTION (IMPORTANT)
# -----------------------------
try:
    client.delete_collection(name="maarifaa")
    print("🗑️ Old collection deleted.")
except Exception as e:
    print("ℹ️ No old collection found.")

# -----------------------------
# CREATE NEW COLLECTION
# -----------------------------
collection = client.get_or_create_collection(
    name="maarifaa",
    embedding_function=ollama_ef
)

# -----------------------------
# INGEST FILES
# -----------------------------
files = ["maarifaa.txt", "about.txt", "policy.txt"]
documents = []
metadatas = []

for fname in files:
    if os.path.exists(fname):
        with open(fname, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                parts = [p.strip() for p in content.split("\n\n") if p.strip()]
                for part in parts:
                    documents.append(part)
                    metadatas.append({"source": fname})
    else:
        print(f"⚠️ {fname} not found")

# -----------------------------
# ADD DOCUMENTS
# -----------------------------
if documents:
    collection.add(
        ids=[str(uuid.uuid4()) for _ in documents],
        documents=documents,
        metadatas=metadatas
    )
    print(f"✅ Ingested {len(documents)} documents")

print("📦 Collection count:", collection.count())

# -----------------------------
# IMPORTANT: Add this function
# -----------------------------
def get_collection():
    return collection
