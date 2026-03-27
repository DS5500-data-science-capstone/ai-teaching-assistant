from langchain_google_cloud_sql_pg import PostgresVectorStore, PostgresEngine
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION")
INSTANCE = os.getenv("CLOUD_SQL_INSTANCE")
DATABASE = os.getenv("CLOUD_SQL_DATABASE")
DB_USER = os.getenv("CLOUD_SQL_USER")
DB_PASS = os.getenv("CLOUD_SQL_PASSWORD")
TABLE_NAME = "course_embeddings"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

PROMPT_TEMPLATE = """
You are a helpful teaching assistant. Answer the student's question 
based only on the following context from the course materials.
If the answer is not in the context, say "I don't have enough 
information in the course materials to answer this question."

Context:
{context}

Student Question: {question}

Answer:
"""


async def query_rag(question: str):
    # Step 1: Connect to Cloud SQL
    engine = await PostgresEngine.afrom_instance(
        project_id=PROJECT_ID,
        region=REGION,
        instance=INSTANCE,
        database=DATABASE,
        user=DB_USER,
        password=DB_PASS,
    )

    # Step 2: Load vector store and retrieve relevant chunks
    vector_store = await PostgresVectorStore.create(
        engine,
        embedding_service=OpenAIEmbeddings(),
        table_name=TABLE_NAME,
    )

    results = await vector_store.asimilarity_search(question, k=5)

    if not results:
        print("No relevant context found.")
        return

    # Step 3: Build context from retrieved chunks
    context = "\n\n---\n\n".join([doc.page_content for doc in results])

    # Step 4: Build prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=question)

    # Step 5: Call Groq
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",  # fast and free
    )

    response = llm.invoke(prompt)

    # Step 6: Print results
    print("\n=== Answer ===")
    print(response.content)
    print("\n=== Sources ===")
    for i, doc in enumerate(results):
        print(f"Source {i+1}: Page {doc.metadata.get('page', 'N/A')} - {doc.metadata.get('source', 'N/A')}")


def main():
    question = input("Ask a question: ")
    asyncio.run(query_rag(question))


if __name__ == "__main__":
    main()