import os
import uuid
import chromadb
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
# DELETE OLD COLLECTION (OPTIONAL BUT CLEAN)
# -----------------------------
COLLECTION_NAME = "maarifaa"

try:
    client.delete_collection(name=COLLECTION_NAME)
    print("🗑️ Old collection deleted.")
except Exception:
    print("ℹ️ No old collection found (fresh start).")

# -----------------------------
# CREATE NEW COLLECTION
# -----------------------------
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ollama_ef
)

# -----------------------------
# INGEST (ROOT FILES + FOLDER FILES)
# -----------------------------
documents = []
metadatas = []
ids = []
seen = set()  # to avoid duplicates

def add_text_chunks(text: str, source_path: str, base_dir: str = ""):
    """Split text into chunks and add to documents list with dedupe."""
    global documents, metadatas, ids, seen

    text = (text or "").strip()
    if not text:
        return

    # split by double newline (your current logic)
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]

    for part in parts:
        key = part  # dedupe by exact text
        if key in seen:
            continue
        seen.add(key)

        ids.append(str(uuid.uuid4()))
        documents.append(part)

        meta = {
            "source": source_path,
            "filename": os.path.basename(source_path),
        }

        # nice relative path for debugging
        if base_dir:
            try:
                meta["relative_path"] = os.path.relpath(source_path, base_dir)
            except Exception:
                meta["relative_path"] = source_path

        metadatas.append(meta)


# 1) Root files (same folder as this script)
root_files = ["maarifaa.txt", "about.txt", "policy.txt"]

for fname in root_files:
    if os.path.exists(fname):
        with open(fname, "r", encoding="utf-8") as f:
            add_text_chunks(f.read(), source_path=fname)
    else:
        print(f"⚠️ Root file not found: {fname}")

# 2) Folder files (Maarifaa AI + subfolders)
data_folder = "Maarifaa AI"

if os.path.isdir(data_folder):
    for root, dirs, files in os.walk(data_folder):
        for fname in files:
            if fname.lower().endswith(".txt"):
                path = os.path.join(root, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        add_text_chunks(
                            f.read(),
                            source_path=path,
                            base_dir=data_folder
                        )
                except Exception as e:
                    print(f"⚠️ Could not read {path}: {e}")
else:
    print(f"⚠️ Folder not found: {data_folder}")

# -----------------------------
# ADD DOCUMENTS TO CHROMA
# -----------------------------
if documents:
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    print(f"✅ Ingested {len(documents)} chunks (deduped).")
else:
    print("⚠️ No documents found to ingest.")

print("📦 Collection count:", collection.count())

# -----------------------------
# IMPORTANT: Add this function
# -----------------------------
def get_collection():
    return collection
