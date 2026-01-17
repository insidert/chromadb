import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from ollama import Client

# -----------------------------
# OLLAMA
# -----------------------------
ollama = Client(host="http://localhost:11434")

# -----------------------------
# EMBEDDINGS (MUST MATCH chroma.py)
# -----------------------------
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434"
)

# -----------------------------
# CHROMA (MUST MATCH chroma.py)
# -----------------------------
client = chromadb.Client(
    Settings(
        persist_directory="chroma_db"
    )
)

collection = client.get_or_create_collection(
    name="maarifaa",
    embedding_function=ollama_ef
)

# -----------------------------
# QUESTIONS
# -----------------------------
questions = [
    "What is Maarifaa?",
    "What is Maarifaa vision?",
    "What is Maarifaa policy?"
]

# -----------------------------
# QUERY LOOP
# -----------------------------
for question in questions:
    results = collection.query(
        query_texts=[question],
        n_results=5
    )

    docs = results["documents"][0]

    print("\n----------------------------")
    print("Question:", question)

    if not docs:
        print("Answer: I dont have that information in the provided documents.")
        continue

    context = "\n\n".join(docs)[:4000]

    prompt = f"""
Answer ONLY using the context below.
If not found, say:
"I dont have that information in the provided documents."

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": "You are a strict assistant."},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.2}
    )

    print("Answer:", response["message"]["content"])
