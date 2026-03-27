from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_google_cloud_sql_pg import PostgresVectorStore, PostgresEngine
from langchain_google_community import GCSDirectoryLoader
from langchain_google_community import GCSFileLoader
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

DATA_PATH = "data/books"

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION")
INSTANCE = os.getenv("CLOUD_SQL_INSTANCE")
DATABASE = os.getenv("CLOUD_SQL_DATABASE")
DB_USER = os.getenv("CLOUD_SQL_USER")
DB_PASS = os.getenv("CLOUD_SQL_PASSWORD")
TABLE_NAME = "course_embeddings"

GCS_BUCKET = os.getenv("GCS_BUCKET_NAME")
GCS_PREFIX = "notes"  # folder inside your bucket


def main():
    asyncio.run(generate_data_store())


async def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    await save_to_cloud_sql(chunks)

def clean_text(text: str) -> str:
    # Remove null bytes and other problematic characters
    return text.replace("\x00", "").strip()

def load_documents():
    from google.cloud import storage
    
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)
    blobs = bucket.list_blobs(prefix=GCS_PREFIX)
    
    documents = []
    for blob in blobs:
        if blob.name.endswith(".pdf"):
            loader = GCSFileLoader(
                project_name=PROJECT_ID,
                bucket=GCS_BUCKET,
                blob=blob.name,
                loader_func=PyPDFLoader  # use PyPDF instead of unstructured
            )
            documents.extend(loader.load())
            print(f"Loaded: {blob.name}")
    
    print(f"Loaded {len(documents)} pages from GCS.")
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


async def save_to_cloud_sql(chunks: list[Document]):
    print("\nConnecting to Cloud SQL...")
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
        overwrite_existing=True,  # table already exists, don't recreate it
    )

    vector_store = await PostgresVectorStore.create(
        engine,
        embedding_service=OpenAIEmbeddings(),
        table_name=TABLE_NAME,
    )

    # Extract texts and metadatas from Document objects
    texts = [clean_text(chunk.page_content) for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    # Upload in batches
    BATCH_SIZE = 100
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_metadatas = metadatas[i:i + BATCH_SIZE]
        await vector_store.aadd_texts(texts=batch_texts, metadatas=batch_metadatas)
        print(f"Uploaded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks...")

    print(f"\nSuccessfully saved {len(chunks)} chunks to Cloud SQL table '{TABLE_NAME}'.")


if __name__ == "__main__":
    main()