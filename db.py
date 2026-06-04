import chromadb
from chromadb.utils import embedding_functions
import os

DB_PATH = os.path.abspath("./chroma_db")

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434"
)

client = chromadb.PersistentClient(path=DB_PATH)

COLLECTION_NAME = "maarifaa"

def get_collection():
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ollama_ef
    )