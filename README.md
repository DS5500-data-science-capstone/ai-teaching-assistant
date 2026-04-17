# AI Teaching Assistant — CS 5200 Database Management Systems

An AI-powered teaching assistant for faculty managing a university database systems course. The system provides at-risk student detection, RAG-based Q&A grounded in course materials, AI quiz generation, discussion management, course planning, and automated slide generation — all accessible through a React faculty dashboard and a student hub.

## Live Deployment

| Service | URL |
|---------|-----|
| Faculty Dashboard | https://ait-frontend-690999534101.us-east1.run.app |
| Student Hub | https://ait-frontend-690999534101.us-east1.run.app/#student |
| Backend API | https://ait-backend-690999534101.us-east1.run.app |

## Key Features

- **At-Risk Student Detection** — Logistic Regression model trained on 300 synthetic students; flags students by grade, attendance, assignment completion, and quiz average with risk score and contributing factors; emails faculty on high/medium risk detection
- **Learning Analytics Dashboard** — Grade distribution histogram, attendance vs grade scatter plot, assignment completion bar chart, ML risk assessment with per-student drill-down
- **RAG-Based Course Q&A** — Category-aware prompt (logistics / concept / homework hints) grounded strictly in uploaded course PDFs via pgvector similarity search; shows source citations with page numbers
- **AI Quiz Builder** — Generate MCQ, fill-in-the-blank, true/false, and long-answer quizzes from course content; marks target with equal or per-type distribution; source badges on each question; distractor explanations; push to students; PDF download (questions only or answer key)
- **Faculty Q&A Review Flow** — Faculty generates an AI draft reply to student questions, edits it in a text area, then posts manually — AI never auto-posts
- **Course Planner & PPT Generator** — 16-week course plan builder with per-week topic, difficulty, and description; AI generates 10-15 slides per topic from course materials; download as PPTX or PDF with Northeastern red/white branding
- **Rubric Builder** — Build homework, project, or quiz rubrics with AI-suggested level descriptors (Excellent / Good / Satisfactory / Needs Improvement); PDF download with Northeastern branding
- **Student Hub** — Student login, grade/attendance/quiz dashboard, quiz taking with distractor explanations after submission, Q&A board
- **Automated PDF Chunking** — Uploading a PDF to GCS triggers a Cloud Function that chunks and embeds it into the vector store automatically
- **Document Management** — Upload and browse lecture PDFs from the UI; stored in GCS under `notes/`

## Team

| Member | Role |
|--------|------|
| **Priyadharshan Sengutuvan** | System Architecture, Backend API, Frontend |
| **Anjana Deivasigamani** | Quiz Builder, Evaluation Metrics, Risk Detection, LangGraph Integration |
| **Raghu Ram Baskaran** | RAG Pipeline, Monitoring, GCP Infrastructure, Documentation |

**Instructor:** Prof. Cristiano  
**Course:** DS 5500 — Data Science Capstone, Spring 2026  
**University:** Northeastern University, Khoury College of Computer Sciences

## Project Structure

```
ai-teaching-assistant/
├── frontend/                   # React + Vite + Tailwind
│   └── src/
│       ├── App.jsx
│       ├── store.js             # localStorage quiz sync
│       └── components/
│           ├── Dashboard.jsx    # Analytics + ML risk model
│           ├── Students.jsx     # Student list + email
│           ├── Lectures.jsx     # Document upload + RAG Q&A
│           ├── Discussion.jsx   # Q&A board (faculty draft flow)
│           ├── Quiz.jsx         # Quiz builder (faculty)
│           ├── StudentView.jsx  # Student hub
│           ├── CoursePlanner.jsx# Course plan + PPT generator
│           └── Rubrics.jsx      # Rubric builder
├── backend/
│   ├── main.py                  # FastAPI — all API routes
│   └── requirements.txt
├── models/
│   ├── rag.py                   # query_rag + query_discussion
│   ├── slides.py                # PPTX + PDF slide generation
│   └── risk.py                  # At-risk inference + email alert
│   └── risk_model.pkl           # Trained logistic regression model
├── scripts/
│   ├── query_data.py            # RAG pipeline (Groq + pgvector)
│   ├── quiz_builder.py          # Quiz generation (Groq + RAG)
│   ├── train_risk_model.py      # Train at-risk model on synthetic data
│   └── migrate_db.py            # Cloud SQL table creation
├── cloud/                       # GCP Cloud Function (auto-chunking)
│   ├── main.py
│   └── requirements.txt
├── docker-compose.yml
├── .env.example
└── pytest.ini
```

## Architecture

```
Faculty Dashboard (React + Vite)          Student Hub (#student)
         ↓                                        ↓
              nginx reverse proxy (/api/)
                        ↓
              FastAPI Backend (Cloud Run)
    ├── /discussion          → in-memory Q&A threads
    ├── /discussion/ai-draft → Groq RAG draft (not auto-posted)
    ├── /upload              → GCS (notes/)
    ├── /documents           → list GCS PDFs
    ├── /ask                 → RAG Q&A + TA email
    ├── /generate-quiz       → Groq + pgvector
    ├── /predict-risk-batch  → scikit-learn logistic regression
    ├── /generate-slides     → Groq slide content
    ├── /download-slides     → python-pptx / reportlab
    └── /suggest-rubric-criteria → Groq rubric levels
                        ↓
         ┌──────────────────────────────┐
         │     Google Cloud Platform    │
         │  ├── Cloud Run (backend)     │
         │  ├── Cloud Run (frontend)    │
         │  ├── GCS Bucket             │ ← PDF storage
         │  ├── Cloud Function         │ ← auto-chunk on upload
         │  ├── Cloud SQL (pgvector)   │ ← embeddings
         │  └── Secret Manager        │ ← API keys
         └──────────────────────────────┘
                        ↓
         OpenAI (embeddings) + Groq (llama-3.1-8b-instant)
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite, Tailwind CSS, Recharts, jsPDF |
| Backend | FastAPI, Uvicorn |
| ML | scikit-learn (Logistic Regression), pandas, NumPy |
| Slide Generation | python-pptx, reportlab |
| Cloud Functions | Google Cloud Functions Gen 2 (Python 3.12) |
| Storage | Google Cloud Storage |
| Vector DB | Cloud SQL (PostgreSQL + pgvector) |
| Embeddings | OpenAI `text-embedding-ada-002` |
| LLM | Groq (`llama-3.1-8b-instant`) |
| RAG Framework | LangChain LCEL, langchain-google-cloud-sql-pg |
| Infrastructure | Docker Compose, nginx, Google Cloud Run |
| Secrets | GCP Secret Manager |

## Local Development

### Prerequisites

- Python 3.12+
- Node.js 20+
- Docker Desktop
- Google Cloud SDK (`gcloud`)
- GCP project with billing enabled
- OpenAI API key
- Groq API key

### Setup

```bash
git clone https://github.com/DS5500-data-science-capstone/ai-teaching-assistant.git
cd ai-teaching-assistant
cp .env.example .env
# Fill in credentials in .env
```

### Run with Docker (recommended)

```bash
docker-compose up -d --build
```

Open `http://localhost` — faculty dashboard.  
Open `http://localhost/#student` — student hub.

### Run without Docker

```bash
# Backend
pip install -r backend/requirements.txt
cd backend && uvicorn main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev
# Open http://localhost:5173
```

### Train the At-Risk Model

```bash
pip install scikit-learn pandas numpy
python scripts/train_risk_model.py
# Saves models/risk_model.pkl
```

## Cloud Deployment (Google Cloud Run)

```bash
# Build and push images
docker buildx build --platform linux/amd64 -t gcr.io/<PROJECT_ID>/ait-backend -f backend/Dockerfile . --push
docker buildx build --platform linux/amd64 -t gcr.io/<PROJECT_ID>/ait-frontend ./frontend --push

# Deploy backend
gcloud run deploy ait-backend --image gcr.io/<PROJECT_ID>/ait-backend \
  --platform managed --region us-east1 --allow-unauthenticated \
  --set-env-vars="GCP_PROJECT_ID=<PROJECT_ID>,GCP_REGION=us-central1,..." \
  --set-secrets="OPENAI_API_KEY=OPENAI_API_KEY:latest,..."

# Deploy frontend
gcloud run deploy ait-frontend --image gcr.io/<PROJECT_ID>/ait-frontend \
  --platform managed --region us-east1 --allow-unauthenticated --port 8080
```

## Cloud Function Deployment

```bash
cd cloud
gcloud functions deploy process-pdf \
  --region=us-central1 \
  --runtime=python312 \
  --entry-point=process_new_pdf \
  --memory=2GB \
  --timeout=540 \
  --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
  --trigger-event-filters="bucket=<GCS_BUCKET_NAME>" \
  --gen2 \
  --source=.
```

## Running Tests

```bash
pytest
pytest --cov=scripts tests/
```

## Environment Variables

```
OPENAI_API_KEY=
GROQ_API_KEY=
GCS_BUCKET_NAME=
GCP_PROJECT_ID=
GCP_REGION=
CLOUD_SQL_INSTANCE=
CLOUD_SQL_DATABASE=
CLOUD_SQL_USER=
CLOUD_SQL_PASSWORD=
GMAIL_EMAIL=
GMAIL_APP_PASSWORD=
TA_EMAIL=
```

## Project Status

**Current Iteration: Final — April 2026**

- [x] React faculty dashboard (Dashboard, Students, Lectures, Q&A, Quiz Builder, Course Planner, Rubrics)
- [x] Student hub (login, grades, quiz taking, Q&A)
- [x] FastAPI backend with all routes
- [x] RAG pipeline — category-aware prompt, source citations, query enhancement
- [x] At-risk student detection — Logistic Regression, email alerts, analytics charts
- [x] AI Quiz Builder — marks target, per-type distribution, source badges, distractor explanations
- [x] Faculty Q&A draft review flow — AI generates draft, faculty edits before posting
- [x] Course Planner + PPT/PDF slide generation with Northeastern branding
- [x] Rubric Builder with AI-suggested level descriptors
- [x] Dockerized with nginx reverse proxy
- [x] Deployed on Google Cloud Run (free tier)
- [x] GCP Secret Manager for API key management
- [ ] Canvas API integration
- [ ] Real student data from Canvas gradebook
- [ ] Persistent discussion threads (Cloud SQL)