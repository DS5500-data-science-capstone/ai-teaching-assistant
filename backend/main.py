import sys
import os
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GCS_BUCKET = os.getenv("GCS_BUCKET_NAME")


# ── In-memory discussion store ───────────────────────────────────────────────

class ThreadModel(BaseModel):
    author: str
    role: str
    message: str

class ReplyPayload(BaseModel):
    author: str
    role: str
    message: str

class AIReplyRequest(BaseModel):
    thread_id: str
    question: str

_threads: list = []




@app.get("/discussion")
def get_threads():
    return {"threads": _threads}


@app.post("/discussion")
def post_thread(body: ThreadModel):
    thread = {
        "id": str(uuid.uuid4()),
        "author": body.author,
        "role": body.role,
        "message": body.message,
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "replies": [],
    }
    _threads.insert(0, thread)
    return thread


@app.post("/discussion/ai-reply")
async def ai_reply(req: AIReplyRequest):
    try:
        from models.rag import query_discussion
        answer = await query_discussion(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    for thread in _threads:
        if thread["id"] == req.thread_id:
            reply = {
                "id": str(uuid.uuid4()),
                "author": "AI Assistant",
                "role": "ai",
                "message": answer,
                "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            }
            thread["replies"].append(reply)
            return reply

    raise HTTPException(status_code=404, detail="Thread not found")


@app.post("/discussion/{thread_id}/reply")
def post_reply(thread_id: str, body: ReplyPayload):
    for thread in _threads:
        if thread["id"] == thread_id:
            reply = {
                "id": str(uuid.uuid4()),
                "author": body.author,
                "role": body.role,
                "message": body.message,
                "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            }
            thread["replies"].append(reply)
            return reply
    raise HTTPException(status_code=404, detail="Thread not found")


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
    




# ── slide - maker ──────────────────────────────────────────────────────────

class SlideRequest(BaseModel):
    week: int
    topic: str
    difficulty: str = "Medium"
    description: str = ""

class DownloadRequest(BaseModel):
    week: int
    topic: str
    difficulty: str
    slides: list
    format: str = "pptx"   # "pptx" or "pdf"


# ── At-Risk Detection ─────────────────────────────────────────────────────────

class StudentRiskRequest(BaseModel):
    name:        str
    grade:       float
    attendance:  float
    assignments: int
    quiz_avg:    Optional[float] = None


@app.post("/predict-risk")
def predict_risk(req: StudentRiskRequest):
    try:
        from models.risk import predict_risk as _predict, send_risk_alert
        result = _predict(req.dict())
        # Send alert for high/medium risk
        if result["at_risk"] and result["level"] in ("high", "medium"):
            send_risk_alert(req.name, result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-risk-batch")
def predict_risk_batch(students: list[StudentRiskRequest]):
    try:
        from models.risk import predict_risk as _predict
        results = []
        for s in students:
            r = _predict(s.dict())
            results.append({"name": s.name, **r})
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Rubric Criteria Suggestion ────────────────────────────────────────────────

class RubricRequest(BaseModel):
    assignment_title:       str
    assignment_type:        str  # Homework | Project | Quiz
    criterion_name:         str
    criterion_description:  str = ""
    total_points:           int = 10


@app.post("/suggest-rubric-criteria")
async def suggest_rubric_criteria(req: RubricRequest):
    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        import json, re

        prompt = ChatPromptTemplate.from_template("""
You are a professor at Northeastern University creating a grading rubric for CS 5200 Database Management Systems.

Assignment: {title} ({type})
Criterion: {criterion} — {description}
Total points for this criterion: {points}

Generate level descriptors for exactly these 4 levels: Excellent, Good, Satisfactory, Needs Improvement.
Return ONLY a JSON array, no explanation, no markdown:
[
  {{"level": "Excellent",         "description": "..."}},
  {{"level": "Good",              "description": "..."}},
  {{"level": "Satisfactory",      "description": "..."}},
  {{"level": "Needs Improvement", "description": "..."}}
]
""")
        chain = prompt | ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
        ) | StrOutputParser()

        raw = await chain.ainvoke({
            "title":       req.assignment_title,
            "type":        req.assignment_type,
            "criterion":   req.criterion_name,
            "description": req.criterion_description or req.criterion_name,
            "points":      req.total_points,
        })

        raw = re.sub(r"```json|```", "", raw).strip()
        start, end = raw.find("["), raw.rfind("]") + 1
        levels = json.loads(raw[start:end])
        return {"levels": levels}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ── slide-maker ──────────────────────────────────────────────────────────

@app.post("/generate-slides")
async def generate_slides_endpoint(req: SlideRequest):
    try:
        from models.slides import generate_slides
        slides = await generate_slides(req.week, req.topic, req.difficulty, req.description)
        return {"slides": slides}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/download-slides")
async def download_slides(req: DownloadRequest):
    import tempfile
    from fastapi.responses import Response
    try:
        from models.slides import build_pptx, build_pdf
        with tempfile.TemporaryDirectory() as tmp:
            if req.format == "pptx":
                path = build_pptx(req.week, req.topic, req.slides, tmp)
                media = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            else:
                path = build_pdf(req.week, req.topic, req.slides, tmp)
                media = "application/pdf"
            with open(path, "rb") as f:
                content = f.read()
        fname = f"Week{req.week}_{req.topic.replace(' ', '_')}.{req.format}"
        return Response(
            content=content,
            media_type=media,
            headers={"Content-Disposition": f"attachment; filename={fname}"}
        )
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