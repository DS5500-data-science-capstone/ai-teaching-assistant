from langchain_google_cloud_sql_pg import PostgresVectorStore, PostgresEngine
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import asyncio
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION")
INSTANCE = os.getenv("CLOUD_SQL_INSTANCE")
DATABASE = os.getenv("CLOUD_SQL_DATABASE")
DB_USER = os.getenv("CLOUD_SQL_USER")
DB_PASS = os.getenv("CLOUD_SQL_PASSWORD")
TABLE_NAME = "course_embeddings"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Email configuration
OUTLOOK_EMAIL = os.getenv("OUTLOOK_EMAIL")       # your Outlook email
OUTLOOK_PASSWORD = os.getenv("OUTLOOK_PASSWORD") # your Outlook password
TA_EMAIL = os.getenv("TA_EMAIL")                 # TA's email address

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


def send_ta_notification(question: str, answer: str, sources: list):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sources_text = "".join([
            f"<li>Page {s.get('page', 'N/A')} | {s.get('source', 'N/A')}</li>"
            for s in sources
        ])

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 700px; margin: auto;">
            <div style="background-color: #CC0000; padding: 20px; border-radius: 8px 8px 0 0;">
                <h2 style="color: white; margin: 0;">CS 6200 – New AI Draft Answer for Review</h2>
                <p style="color: #ffcccc; margin: 5px 0 0 0;">{timestamp}</p>
            </div>
            <div style="border: 1px solid #ddd; padding: 20px; border-radius: 0 0 8px 8px;">
                <h3 style="color: #CC0000;">Student Question</h3>
                <p style="background: #f5f5f5; padding: 12px; border-left: 4px solid #CC0000;">
                    {question}
                </p>
                <h3 style="color: #CC0000;">AI Generated Answer</h3>
                <p style="background: #f9f9f9; padding: 12px; border-left: 4px solid #333; white-space: pre-wrap;">
                    {answer}
                </p>
                <h3 style="color: #CC0000;">Sources Used</h3>
                <ul>{sources_text}</ul>
                <div style="margin-top: 20px; padding: 12px; background: #fff8e1; border: 1px solid #ffe082;">
                    <strong>Action Required:</strong> Please review this AI-generated answer before it is shown to the student.
                </div>
            </div>
        </body>
        </html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[CS6200] AI Draft Answer Needs Review – {timestamp}"
        msg["From"] = os.getenv("GMAIL_EMAIL")
        msg["To"] = os.getenv("TA_EMAIL")
        msg.attach(MIMEText(html_body, "html"))

        # Gmail SMTP
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(
                os.getenv("GMAIL_EMAIL"),
                os.getenv("GMAIL_APP_PASSWORD")
            )
            server.sendmail(
                os.getenv("GMAIL_EMAIL"),
                os.getenv("TA_EMAIL"),
                msg.as_string()
            )

        print(f"✅ TA notified at {os.getenv('TA_EMAIL')}")

    except Exception as e:
        print(f"❌ Failed to send email notification: {e}")


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

    results = await vector_store.asimilarity_search(question, k=5)when do you have office hours @
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
        model_name="llama-3.1-8b-instant",
    )

    response = llm.invoke(prompt)
    answer = response.content

    # Step 6: Print results
    print("\n=== Answer ===")
    print(answer)
    print("\n=== Sources ===")
    sources = []
    for i, doc in enumerate(results):
        meta = doc.metadata
        sources.append(meta)
        print(f"Source {i+1}: Page {meta.get('page', 'N/A')} - {meta.get('source', 'N/A')}")

    # Step 7: Notify TA via email
    print("\nNotifying TA...")
    send_ta_notification(question, answer, sources)


def main():
    question = input("Ask a question: ")
    asyncio.run(query_rag(question))


if __name__ == "__main__":
    main()