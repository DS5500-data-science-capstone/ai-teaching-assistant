import functions_framework
import asyncio
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_community import GCSFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_cloud_sql_pg import PostgresVectorStore, PostgresEngine

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION     = os.getenv("GCP_REGION")
INSTANCE   = os.getenv("CLOUD_SQL_INSTANCE")
DATABASE   = os.getenv("CLOUD_SQL_DATABASE")
DB_USER    = os.getenv("CLOUD_SQL_USER")
DB_PASS    = os.getenv("CLOUD_SQL_PASSWORD")
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME")
TABLE_NAME = "course_embeddings"
BATCH_SIZE = 100

def clean_text(text: str) -> str:
    return text.replace("\x00", "").strip()

@functions_framework.cloud_event
def process_new_pdf(cloud_event):
    data      = cloud_event.data
    bucket    = data["bucket"]
    blob_name = data["name"]
    if not blob_name.lower().endswith(".pdf"):
        print(f"Skipping non-PDF: {blob_name}")
        return
    print(f"Processing: gs://{bucket}/{blob_name}")
    asyncio.run(process_pdf(bucket, blob_name))

async def process_pdf(bucket_name, blob_name):
    loader = GCSFileLoader(
        project_name=PROJECT_ID,
        bucket=bucket_name,
        blob=blob_name,
        loader_func=PyPDFLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    engine = await PostgresEngine.afrom_instance(
        project_id=PROJECT_ID,
        region=REGION,
        instance=INSTANCE,
        database=DATABASE,
        user=DB_USER,
        password=DB_PASS
    )
    await engine.ainit_vectorstore_table(
        table_name=TABLE_NAME,
        vector_size=1536,
        overwrite_existing=False
    )
    vector_store = await PostgresVectorStore.create(
        engine,
        embedding_service=OpenAIEmbeddings(),
        table_name=TABLE_NAME
    )

    texts = [clean_text(c.page_content) for c in chunks]
    metadatas = [c.metadata for c in chunks]

    for i in range(0, len(texts), BATCH_SIZE):
        await vector_store.aadd_texts(
            texts=texts[i:i+BATCH_SIZE],
            metadatas=metadatas[i:i+BATCH_SIZE]
        )
        print(f"Uploaded {min(i+BATCH_SIZE, len(texts))}/{len(texts)} chunks")

    print(f"✅ Done: {len(chunks)} chunks saved to {TABLE_NAME}")