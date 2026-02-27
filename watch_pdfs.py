
import os
import time
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import storage
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_community import GCSFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_google_cloud_sql_pg import PostgresVectorStore, PostgresEngine

load_dotenv()

PROJECT_ID  = os.getenv("GCP_PROJECT_ID")
REGION      = os.getenv("GCP_REGION")
INSTANCE    = os.getenv("CLOUD_SQL_INSTANCE")
DATABASE    = os.getenv("CLOUD_SQL_DATABASE")
DB_USER     = os.getenv("CLOUD_SQL_USER")
DB_PASS     = os.getenv("CLOUD_SQL_PASSWORD")
GCS_BUCKET  = os.getenv("GCS_BUCKET_NAME")
GCS_PREFIX  = "notes"
TABLE_NAME  = "course_embeddings"
POLL_INTERVAL  = 30          # seconds between checks
PROCESSED_LOG  = "processed_pdfs.txt"   # local file tracking done PDFs

def load_processed() -> set:
    if not os.path.exists(PROCESSED_LOG):
        return set()
    with open(PROCESSED_LOG) as f:
        return set(line.strip() for line in f if line.strip())

def mark_processed(blob_name: str):
    with open(PROCESSED_LOG, "a") as f:
        f.write(blob_name + "\n")

# ─── Pipeline ──────────
def clean_text(text: str) -> str:
    return text.replace("\x00", "").strip()

def load_single_pdf(blob_name: str) -> list[Document]:
    loader = GCSFileLoader(
        project_name=PROJECT_ID,
        bucket=GCS_BUCKET,
        blob=blob_name,
        loader_func=PyPDFLoader
    )
    return loader.load()

def split_text(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(documents)

async def save_to_cloud_sql(chunks: list[Document]) -> bool:
    try:
        engine = await PostgresEngine.afrom_instance(
            project_id=PROJECT_ID,
            region=REGION,
            instance=INSTANCE,
            database=DATABASE,
            user=DB_USER,
            password=DB_PASS,
        )
        await engine.ainit_vectorstore_table(
            table_name=TABLE_NAME,
            vector_size=1536,
            overwrite_existing=False,   # don't wipe existing data
        )
        vector_store = await PostgresVectorStore.create(
            engine,
            embedding_service=OpenAIEmbeddings(),
            table_name=TABLE_NAME,
        )
        texts     = [clean_text(c.page_content) for c in chunks]
        metadatas = [c.metadata for c in chunks]

        BATCH_SIZE = 100
        for i in range(0, len(texts), BATCH_SIZE):
            await vector_store.aadd_texts(
                texts=texts[i:i+BATCH_SIZE],
                metadatas=metadatas[i:i+BATCH_SIZE]
            )
        return True
    except Exception as e:
        print(f"     ⚠️  DB error: {e}")
        return False

async def process_pdf(blob_name: str) -> dict:
    status = {"name": blob_name, "pages": 0, "chunks": 0, "vector_db": False, "error": None}
    try:
        docs = load_single_pdf(blob_name)
        status["pages"] = len(docs)
        chunks = split_text(docs)
        status["chunks"] = len(chunks)
        status["vector_db"] = await save_to_cloud_sql(chunks)
    except Exception as e:
        status["error"] = str(e)
    return status

# ─── Summary printer ───────────────
def print_summary(new_pdfs: list, all_pdfs: list, processed: set, results: list):
    print("\n" + "═" * 58)
    print(f"  PIPELINE SUMMARY  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 58)
    print(f"  Total PDFs in bucket (notes/) : {len(all_pdfs)}")
    print(f"  Already processed             : {len(processed)}")
    print(f"  New PDFs found this run       : {len(new_pdfs)}")
    print("─" * 58)

    if results:
        for r in results:
            if r["error"]:
                print(f"  {os.path.basename(r['name'])}")
                print(f"     Error: {r['error']}")
            else:
                db_icon = "Done" if r["vector_db"] else "❌"
                print(f"   {os.path.basename(r['name'])}")
                print(f"     Pages : {r['pages']}  |  Chunks : {r['chunks']}")
                print(f"     Vector DB updated : {db_icon}")
    else:
        print("  No new PDFs — nothing to process.")
    print("═" * 58 + "\n")

# ─── Main watcher loop ─────────────────
async def watch():
    print(f" Watching gs://{GCS_BUCKET}/{GCS_PREFIX}/  (every {POLL_INTERVAL}s)\n")
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)

    while True:
        processed = load_processed()
        all_pdfs  = [b.name for b in bucket.list_blobs(prefix=GCS_PREFIX) if b.name.endswith(".pdf")]
        new_pdfs  = [p for p in all_pdfs if p not in processed]

        results = []
        for blob_name in new_pdfs:
            print(f"  Processing {blob_name}...")
            result = await process_pdf(blob_name)
            results.append(result)
            if not result["error"] and result["vector_db"]:
                mark_processed(blob_name)

        print_summary(new_pdfs, all_pdfs, processed, results)
        await asyncio.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    asyncio.run(watch())