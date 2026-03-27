import sys
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from pydantic import BaseModel
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GCS_BUCKET = os.getenv("GCS_BUCKET_NAME")


# ── GCS Upload ──────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(f"notes/{file.filename}")
        contents = await file.read()
        blob.upload_from_string(contents, content_type="application/pdf")
        return {"message": f"{file.filename} uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GCS Document List ────────────────────────────────────────────────────────

@app.get("/documents")
def list_documents():
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blobs = bucket.list_blobs()
        docs = []
        for blob in blobs:
            if blob.name.endswith(".pdf"):
                docs.append({
                    "name": blob.name,
                    "size": f"{round(blob.size / 1024, 1)} KB",
                    "uploadDate": blob.time_created.strftime("%Y-%m-%d") if blob.time_created else "Unknown",
                    "url": f"https://storage.googleapis.com/{GCS_BUCKET}/{blob.name}"
                })
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── RAG Q&A ──────────────────────────────────────────────────────────────────

@app.post("/ask")
async def ask_question(payload: dict):
    question = payload.get("question", "")
    if not question:
        raise HTTPException(status_code=400, detail="No question provided.")
    try:
        from models.rag import query_rag
        answer = await query_rag(question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Quiz Builder ─────────────────────────────────────────────────────────────

class QuizRequest(BaseModel):
    topic: str
    num_questions: int = 3
    difficulty: str = "medium"
    style: str = "conceptual"
    question_type: str = "mcq"
    num_options: int = 4
    source_filter: Optional[str] = None


@app.get("/quiz-topics")
async def get_quiz_topics():
    try:
        from scripts.quiz_builder import load_config, fetch_available_topics
        cfg = load_config()
        topics = await fetch_available_topics(cfg)
        return {"topics": topics}
    except Exception:
        return {"topics": [
            "Relational Model & Algebra", "Modern SQL", "Database Storage I",
            "Hash Tables", "Joins Algorithms", "Query Planning & Optimization",
            "Concurrency Control Theory", "Database Recovery",
        ]}


@app.post("/generate-quiz")
async def generate_quiz(req: QuizRequest):
    try:
        from scripts.quiz_builder import load_config, build_quiz
        cfg = load_config()
        quiz = await build_quiz(
            cfg=cfg,
            user_text=req.topic,
            num_questions=req.num_questions,
            difficulty=req.difficulty,
            retrieval_k=6,
            max_docs=12,
            style=req.style,
            num_options=req.num_options,
            source_filter=req.source_filter,
            question_type=req.question_type,
        )
        return quiz
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))