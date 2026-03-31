from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_google_cloud_sql_pg import PostgresVectorStore, PostgresEngine, Column
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

CHROMA_PATH = "chroma"

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION")
INSTANCE = os.getenv("CLOUD_SQL_INSTANCE")
DATABASE = os.getenv("CLOUD_SQL_DATABASE")
DB_USER = os.getenv("CLOUD_SQL_USER")
DB_PASS = os.getenv("CLOUD_SQL_PASSWORD")
TABLE_NAME = "course_embeddings_test"  # separate test table


async def test_upload():
    # Step 1: Pull a small sample from Chroma
    print("Loading from Chroma...")
    chroma_db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OpenAIEmbeddings()
    )

    total = chroma_db._collection.count()
    print(f"Total chunks in Chroma: {total}")

    # Grab just 10 chunks for testing
    sample = chroma_db._collection.peek(10)
    texts = sample["documents"]
    metadatas = sample["metadatas"]
    print(f"Grabbed {len(texts)} chunks for test upload.")

    # Step 2: Connect to Cloud SQL
    print("\nConnecting to Cloud SQL...")
    engine = await PostgresEngine.afrom_instance(
        project_id=PROJECT_ID,
        region=REGION,
        instance=INSTANCE,
        database=DATABASE,
        user=DB_USER,
        password=DB_PASS,
    )

    # Step 3: Create test table
    await engine.ainit_vectorstore_table(
        table_name=TABLE_NAME,
        vector_size=1536,
        overwrite_existing=True,
    )

    # Step 4: Upload the sample chunks
    vector_store = await PostgresVectorStore.create(
        engine,
        embedding_service=OpenAIEmbeddings(),
        table_name=TABLE_NAME,
    )

    await vector_store.aadd_texts(texts=texts, metadatas=metadatas)
    print(f"\nSuccessfully uploaded {len(texts)} chunks to Cloud SQL table '{TABLE_NAME}'.")

    # Step 5: Verify by running a similarity search
    query = "your test query here"
    print(f"\nRunning test similarity search for: '{query}'")
    results = await vector_store.asimilarity_search(query, k=3)
    for i, result in enumerate(results):
        print(f"\nResult {i+1}: {result.page_content[:200]}...")
        print(f"Metadata: {result.metadata}")


if __name__ == "__main__":
    asyncio.run(test_upload())