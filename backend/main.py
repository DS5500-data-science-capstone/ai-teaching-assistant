from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GCS_BUCKET = os.getenv("GCS_BUCKET_NAME")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(file.filename)
        contents = await file.read()
        blob.upload_from_string(contents, content_type="application/pdf")
        return {"message": f"{file.filename} uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
