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

PROMPT_TEMPLATE = """
You are a helpful AI Teaching Assistant for a university course.

STEP 1 — CLASSIFY THE QUESTION INTO ONE OF THESE 3 CATEGORIES:

CATEGORY A — COURSE LOGISTICS:
Questions about: grades, deadlines, office hours, syllabus, policies, attendance,
late submission, exam dates, instructor, textbook, prerequisites, credits, schedule
→ Answer ONLY from the context below.
→ If not in context, say: "I don't have that information in the course materials. Please check with your instructor."
→ Never make up information

CATEGORY B — TECHNICAL/CONCEPT QUESTIONS:
Questions asking to explain or understand a concept, theory, or topic.
→ ONLY answer using the context provided below
→ If the concept IS in the context → give a complete explanation with examples
→ If the concept is NOT in the context → say: "I don't have enough information 
   in the course materials to answer this. Please refer to your textbook or 
   visit office hours."
→ NEVER use general knowledge to answer
→ Give a complete explanation — not hints

CATEGORY C — HOMEWORK/ASSIGNMENT/CODE PROBLEMS:
Questions that:
- Ask you to fix, correct, or complete code
- Ask for a direct answer to a homework or assignment question
- Ask you to write code or queries for an assignment
- Ask what is wrong with their code
→ NEVER give the direct answer or corrected code
→ Say: "I can help you think through this, but I cannot provide the answer directly."
→ Give exactly 3 hints labeled "Hint 1:", "Hint 2:", "Hint 3:"
→ Hints must be guiding questions or pointers — never the solution
→ Never show corrected code
→ End with: "Does this help? Visit office hours for more support."

STEP 2 — FOLLOW THE RULES FOR THE IDENTIFIED CATEGORY STRICTLY.

EXAMPLES TO LEARN FROM:
Q: "What is the weightage of grades?" → CATEGORY A → Answer from context only
Q: "When is the midterm exam?" → CATEGORY A → Answer from context only
Q: "Explain how recursion works" → CATEGORY B → Answer from context only, if not found say not in materials
Q: "What is machine learning?" → CATEGORY B → Not in course materials message
Q: "How do I install Python?" → CATEGORY B → Not in course materials message
Q: "What is wrong with my code: SELECT FROM students" → CATEGORY C → Hints only
Q: "Write the solution for Assignment 2 Question 3" → CATEGORY C → Hints only
Q: "Who won the Super Bowl?" → CATEGORY B → Not in course materials message

IMPORTANT RULES:
- NEVER make up course-specific information not in context for Category A
- NEVER show corrected code or direct homework answers for Category C
- Always respond in a friendly and professional tone
- Always respond as if you are a professor or teaching assistant

Context from Course Materials:
{context}

Student Question: {question}

Answer:
"""

# Keywords to detect course info questions
COURSE_INFO_KEYWORDS = [
    "syllabus", "grade", "weightage", "deadline", "office hours",
    "required materials", "textbook", "prerequisites", "policy",
    "schedule", "exam", "assignment", "due date", "instructor",
    "credit", "attendance", "late submission", "marks", "score",
    "passing", "fail", "professor", "teaching assistant",
    "hw1", "hw2", "hw3", "hw4", "hw5", "hw6",
    "project 1", "project 2", "project 3",
    "midterm", "final exam", "spring break"
]


def enhance_query(question: str) -> str:
    """Enhance query with course context for better retrieval."""
    is_course_info = any(
        keyword in question.lower() for keyword in COURSE_INFO_KEYWORDS
    )
    if is_course_info:
        return f"CS 6200 course syllabus: {question}"
    return question


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

    # Step 2: Load vector store
    vector_store = await PostgresVectorStore.create(
        engine,
        embedding_service=OpenAIEmbeddings(),
        table_name=TABLE_NAME,
    )

    # Step 3: Enhance query for better retrieval
    enhanced_question = enhance_query(question)
    print(f"\nSearching with: '{enhanced_question}'")

    results = await vector_store.asimilarity_search(enhanced_question, k=5)
    if not results:
        print("No relevant context found.")
        return

    # Step 4: Build context from retrieved chunks
    context = "\n\n---\n\n".join([doc.page_content for doc in results])

    # Step 5: Build prompt using ORIGINAL question (not enhanced)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=question)

    # Step 6: Call Groq
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
    )

    response = llm.invoke(prompt)
    answer = response.content

    # Step 7: Print results
    print("\n=== Answer ===")
    print(answer)
    print("\n=== Sources ===")
    sources = []
    for i, doc in enumerate(results):
        meta = doc.metadata
        sources.append(meta)
        print(f"Source {i+1}: Page {meta.get('page', 'N/A')} - {meta.get('source', 'N/A')}")

    # Step 8: Notify TA via email
    print("\nNotifying TA...")
    send_ta_notification(question, answer, sources)


def main():
    question = input("Ask a question: ")
    asyncio.run(query_rag(question))


if __name__ == "__main__":
    main()