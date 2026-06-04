import os
import uuid
import chromadb
from chromadb.utils import embedding_functions

# ---------------- EMBEDDINGS ----------------
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434"
)

# ---------------- DB PATH ----------------
DB_PATH = os.path.abspath("./chroma_db")

client = chromadb.PersistentClient(path=DB_PATH)

COLLECTION_NAME = "maarifaa"

# ---------------- RESET COLLECTION ----------------
try:
    client.delete_collection(COLLECTION_NAME)
    print("🗑️ Old collection deleted")
except:
    pass

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ollama_ef
)

documents, metadatas, ids = [], [], []
seen = set()

# ---------------- CHUNKING ----------------
def add_text_chunks(text, source):
    chunk_size = 300
    overlap = 50
    step = chunk_size - overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size].strip()

        if not chunk or chunk in seen:
            continue

        seen.add(chunk)

        documents.append(chunk)
        ids.append(str(uuid.uuid4()))
        metadatas.append({"source": source})

# ---------------- LOAD FILES ----------------
folder = "Maarifaa AI"

file_count = 0

for root, _, files in os.walk(folder):
    for f in files:
        if f.endswith(".txt"):
            path = os.path.join(root, f)
            file_count += 1

            try:
                with open(path, "r", encoding="utf-8") as file:
                    add_text_chunks(file.read(), path)
            except Exception as e:
                print("❌ Error:", e)

# ---------------- INSERT ----------------
batch_size = 20

for i in range(0, len(documents), batch_size):
    collection.add(
        ids=ids[i:i+batch_size],
        documents=documents[i:i+batch_size],
        metadatas=metadatas[i:i+batch_size]
    )

    print(f"✅ Batch {i//batch_size + 1}")

print("\n📂 Files:", file_count)
print("📄 Chunks:", len(documents))
print("🧠 DB size:", collection.count())