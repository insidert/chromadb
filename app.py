import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from rapidfuzz import process, fuzz
from db import get_collection


# ============================================================
# ENVIRONMENT
# ============================================================

# Load .env manually
if os.path.exists(".env"):
    for line in open(".env"):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ[k] = v

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

if not OPENROUTER_API_KEY:
    print("⚠️  OPENROUTER_API_KEY not set in .env")

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# OPENROUTER (remote LLM, ~instant)
# ============================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

CHAT_MODEL = "openai/gpt-4o-mini"

# ============================================================
# CHROMADB
# ============================================================

collection = get_collection()

print("📦 Total docs in DB:", collection.count())

# ============================================================
# REQUEST MODEL
# ============================================================

class Query(BaseModel):
    question: str

# ============================================================
# SPELLING DATASET
# ============================================================

KNOWN_QUESTIONS = [
    "what is maarifaa",
    "about maarifaa",
    "maarifaa mission",
    "study in india",
    "courses in india",
    "visa process",
    "student visa",
    "scholarships",
    "scholarships for african students",
    "living in india",
    "admission process",
    "student support",
    "hostel in india",
    "fees in india",
    "mbbs admission",
    "neet eligibility",
    "engineering courses",
    "medical courses",
    "career guidance",
    "can african students study mbbs in india",
    "african students mbbs",
    "biology chemistry physics medicine",
    "indian university admission",
    "english taught programs",
    "part time work in india",
    "international student visa",
    "campus placement international students",
    "indian education recognition",
    "african student associations india",
    "is india safe for african students",
    "cook own food in india",
    "open bank account in india",
    "maap program",
    "maap aptitude assessment",
    "internship medical students",
    "tuition fee in india",
    "frro registration",
    "discrimination african students india",
    "african national boards acceptance",
]

# ============================================================
# HELPERS
# ============================================================

def normalize(text):
    return text.lower().strip()

def correct_query(question):

    result = process.extractOne(
        question,
        KNOWN_QUESTIONS,
        scorer=fuzz.WRatio
    )

    if result:
        match, score, _ = result

        if score >= 80:
            return match

    return question

# ============================================================
# SUGGESTIONS
# ============================================================



# ============================================================
# SUGGESTIONS
# ============================================================

TOPIC_SUGGESTIONS = {
    "visa": ["Scholarships", "Study in India", "Hostel in India"],
    "scholarship": ["Visa process", "Courses in India", "Admission process"],
    "course": ["Eligibility", "Visa process", "Scholarships"],
    "admission": ["Courses in India", "Visa process", "Hostel in India"],
    "fee": ["Scholarships", "Living in India", "Admission process"],
    "hostel": ["Living in India", "Fees in India", "Student support"],
    "mbbs": ["NEET eligibility", "Medical courses", "Career guidance"],
    "medical": ["MBBS admission", "NEET eligibility", "Career guidance"],
    "engineering": ["Courses in India", "Scholarships", "Career guidance"],
    "career": ["Courses in India", "Study in India", "Scholarships"],
    "maarifaa": ["Study in India", "MAAP program", "Student support"],
}

def get_suggestions(mode, question):
    if mode == "GREETING":
        return ["What is Maarifaa?", "Study in India", "Scholarships"]

    q = question.lower()
    for keyword, sugs in TOPIC_SUGGESTIONS.items():
        if keyword in q:
            return sugs

    if mode == "GENERAL_KNOWLEDGE":
        return ["Study in India", "Visa process", "Scholarships"]

    return ["Study in India", "Visa process", "Scholarships"]

# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat")
def chat(query: Query):

    question = normalize(query.question)

    # ========================================================
    # DOMAIN DETECTION — skip RAG for off-topic questions
    # ========================================================

    DOMAIN_KEYWORDS = [
        "maarifaa", "india", "study", "student", "course", "visa",
        "scholarship", "mbbs", "medical", "engineering", "admission",
        "hostel", "fee", "university", "college", "exam", "neet",
        "career", "job", "internship", "placement", "degree",
        "diploma", "africa", "african", "international",
        "maap", "aptitude", "ielts", "toefl", "english",
        "biology", "chemistry", "physics", "mathematics",
        "eligibility", "syllabus", "board", "waec", "kcse", "neco",
        "food", "safety", "discrimination", "association",
        "frro", "bank", "part-time", "work",
    ]

    is_domain = any(kw in question for kw in DOMAIN_KEYWORDS)
    print("In domain:", is_domain)

    if not is_domain:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": query.question}],
            max_tokens=300,
            temperature=0.3,
        )
        return {
            "answer": response.choices[0].message.content.strip(),
            "corrected_question": question,
            "mode": "GENERAL_KNOWLEDGE",
            "suggestions": ["What is Maarifaa?", "Study in India", "Scholarships"],
        }

    # ========================================================
    # COMMON MISSPELLINGS
    # ========================================================

    SPELL_FIXES = {
        "inida": "india",
        "africa": "african",
    }
    for wrong, correct in SPELL_FIXES.items():
        question = question.replace(wrong, correct)

    # ========================================================
    # GREETING
    # ========================================================

    if question in ["hi", "hello", "hey"]:
        return {
            "answer": "Hello 👋 Welcome to Maarifaa. Ask me anything.",
            "corrected_question": question,
            "mode": "GREETING",
            "suggestions": [
                "What is Maarifaa?",
                "Study in India",
                "Scholarships"
            ]
        }

    # ========================================================
    # CHROMA SEARCH
    # ========================================================

    results = collection.query(
        query_texts=[question],
        n_results=4,
        include=["documents", "distances"]
    )

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    print("\n===== RETRIEVED DOCS =====")

    for i, doc in enumerate(docs):
        print(f"\nDOC {i+1}:")
        print(doc[:300])

    print("=========================\n")

    print("Distances:", distances)

    if distances:
        print("Best Distance:", distances[0])
    print("\n========================")
    print("Question :", question)
    print("Docs Found:", len(docs))
    print("Distances:", distances)
    print("========================")

    # ========================================================
    # DETERMINE RAG OR GENERAL AI
    # ========================================================

    # ========================================================
# DETERMINE RAG OR GENERAL AI
# ========================================================

    if distances and distances[0] < 0.8:
        use_rag = True
    else:
        use_rag = False

    print("Use RAG:", use_rag)

    

    # ========================================================
    # RAG MODE
    # ========================================================

    if use_rag:

        context = "\n\n".join(docs[:3])[:1000]

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant for Maarifaa. Answer using the context first. If the context is missing details, use your own knowledge to complete the answer. Never say you can't find the answer."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ]

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=300,
            temperature=0,
        )

        answer = response.choices[0].message.content.strip()
        mode = "RAG"

    # ========================================================
    # GENERAL KNOWLEDGE MODE
    # ========================================================

    else:
        context = ""

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": question}],
            max_tokens=300,
            temperature=0,
        )

        answer = response.choices[0].message.content.strip()
        mode = "GENERAL_KNOWLEDGE"

    

    # ========================================================
    # SUGGESTIONS
    # ========================================================

    suggestions = get_suggestions(mode, question)

    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "answer": answer,
        "corrected_question": question,
        "mode": mode,
        "suggestions": suggestions
    }