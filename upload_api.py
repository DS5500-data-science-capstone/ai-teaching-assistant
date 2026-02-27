from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME")
GCS_PREFIX = "notes"


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET)
        blob_name = f"{GCS_PREFIX}/{file.filename}"
        blob = bucket.blob(blob_name)
        contents = await file.read()
        blob.upload_from_string(contents, content_type="application/pdf")
        return {
            "status": "success",
            "message": f"{file.filename} uploaded to GCS.",
            "bucket": GCS_BUCKET,
            "path": blob_name,
            "size_mb": round(len(contents) / (1024 * 1024), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def bucket_status():
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET)
        pdfs = [b.name for b in bucket.list_blobs(prefix=GCS_PREFIX) if b.name.endswith(".pdf")]
        return {"total_pdfs": len(pdfs), "files": pdfs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))