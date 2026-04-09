# AI Teaching Assistant for Canvas LMS

An AI-powered teaching assistant that integrates with Canvas LMS to provide automated learning analytics, at-risk student detection, RAG-based tutoring, and AI-generated quizzes for faculty.

## Project Overview

This capstone project develops an intelligent system that helps faculty enhance student learning through data-driven insights and AI-powered support. The system processes course materials uploaded to Google Cloud Storage, automatically chunks and embeds them into a vector database, and exposes a React-based faculty dashboard for monitoring students, managing documents, generating quizzes, and querying course content.

### Key Features

- **At-Risk Student Detection** — Automated identification of students needing intervention based on grade and activity data
- **Learning Analytics Dashboard** — Visual insights into class performance, attendance, and assignment completion
- **AI-Powered Communication** — Generate personalized check-in emails for struggling students
- **RAG-Based Course Assistant** — Answer faculty questions using course materials stored in Cloud SQL + pgvector
- **AI Quiz Builder** — Generate MCQ, fill-in-the-blank, true/false, and long-answer quizzes from course content using Groq LLM; download as PDF (questions only or answer key)
- **Automated PDF Chunking** — Uploading a PDF to GCS automatically triggers a Cloud Function that chunks and embeds it into the vector store
- **Document Management** — Upload and browse lecture PDFs directly from the UI; files are stored in GCS under `notes/`

## Team Members

| Member | Role |
|--------|------|
| **Priyadharshan Sengutuvan** | System Architecture, RAG Pipeline, Backend API, GCP Infrastructure, Frontend |
| **Anjana Deivasigamani** | Quiz Builder, Evaluation Metrics, Risk Detection, LangGraph Integration |
| **Raghu Ram Baskaran** | Dashboard Development, Monitoring, Documentation |

## Project Structure

```
ai-teaching-assistant/
├── frontend/               # React + Vite + Tailwind faculty dashboard
│   └── src/
│       ├── App.jsx
│       └── components/
│           ├── Dashboard.jsx
│           ├── Students.jsx
│           ├── Lectures.jsx
│           ├── Discussion.jsx
│           └── Quiz.jsx
├── backend/                # FastAPI server
│   ├── main.py             # Upload, document list, RAG Q&A, quiz generation
│   └── requirements.txt
├── cloud/                  # GCP Cloud Function (auto-chunking)
│   ├── main.py             # Triggered on GCS file upload
│   └── requirements.txt
├── scripts/                # One-off data pipeline scripts
│   ├── create_database.py  # Chunk & embed PDFs into Cloud SQL (skips already processed)
│   ├── quiz_builder.py     # Quiz generation logic (Groq + RAG)
│   └── setup_db.py
├── models/
│   └── rag.py              # RAG query logic
├── requirements.txt        # Unified project dependencies
└── .env.example
```

## Architecture

```
Faculty Dashboard (React + Vite)
         ↓
    FastAPI Backend (port 8000)
    ├── /upload        → GCS (notes/)
    ├── /documents     → list GCS PDFs
    ├── /ask           → RAG query
    └── /generate-quiz → Groq LLM + RAG
         ↓
    ┌────────────────────────┐
    │  Google Cloud Platform │
    │  ├── GCS Bucket        │ ← PDF storage
    │  ├── Cloud Function    │ ← auto-chunk on upload
    │  └── Cloud SQL (pg)    │ ← vector embeddings
    └────────────────────────┘
         ↓
    OpenAI (embeddings) + Groq (quiz generation)
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React, Vite, Tailwind CSS |
| Backend | FastAPI, Uvicorn |
| Cloud Functions | Google Cloud Functions (Gen 2) |
| Storage | Google Cloud Storage |
| Vector DB | Cloud SQL (PostgreSQL + pgvector) |
| Embeddings | OpenAI `text-embedding-ada-002` |
| LLM (Quiz) | Groq (`llama-3.1-8b-instant`) |
| RAG Framework | LangChain, LangGraph |
| Data Processing | Pandas, NumPy, scikit-learn |

## Prerequisites

- Python 3.12+
- Node.js 18+
- Google Cloud SDK (`gcloud`)
- GCP project with billing enabled
- OpenAI API key
- Groq API key

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/DS5500-data-science-capstone/ai-teaching-assistant.git
cd ai-teaching-assistant
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
# Fill in your credentials
```

Required variables:
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
```

### 3. Install Python Dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 5. Initialize the Vector Database (first time only)

```bash
python scripts/create_database.py
```

This loads all PDFs from `gs://<GCS_BUCKET>/notes/`, chunks them, and stores embeddings in Cloud SQL. Already-processed files are skipped on subsequent runs.

### 6. Run the Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 7. Run the Frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` and sign in with any email and password.

## Cloud Function Deployment

The Cloud Function auto-processes any PDF uploaded to GCS:

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
pytest --cov=src tests/
```

## Project Status

**Current Iteration: Final — March 2026**

- [x] Repository setup and project structure
- [x] React faculty dashboard (Dashboard, Students, Lectures, Discussion, Quiz Builder)
- [x] FastAPI backend (upload, document listing, RAG Q&A, quiz generation)
- [x] GCS document storage with `notes/` folder structure
- [x] Cloud Function for automated PDF chunking on upload
- [x] Cloud SQL + pgvector for embedding storage
- [x] RAG pipeline with OpenAI embeddings
- [x] AI Quiz Builder with Groq LLM (MCQ, fill-blank, true/false, long answer)
- [x] PDF quiz download (questions only + answer key with color-coded answers)
- [x] Mobile-responsive UI
- [x] Session persistence (no re-login on page reload)
- [x] Unified requirements.txt
- [ ] Canvas API integration
- [ ] Real student data integration
- [ ] Email notification system
- [ ] Deployed production URL

**Last Updated:** March 2026
**Academic Term:** Spring 2026
**Course:** DS 5500 — Data Science Capstone