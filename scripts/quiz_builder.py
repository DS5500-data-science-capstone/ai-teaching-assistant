#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import aiohttp
from dotenv import load_dotenv
from google.cloud import storage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_google_cloud_sql_pg import PostgresEngine, PostgresVectorStore
from langchain_google_community import GCSFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, ValidationError, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if google_credentials:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration for ingestion and quiz generation."""

    project_id: str
    region: str
    instance: str
    database: str
    db_user: str
    db_pass: str
    gcs_bucket: str
    gcs_prefix: str
    gcs_out_prefix: str
    groq_api_key: str
    groq_model: str
    groq_base_url: str
    table_name: str
    vector_size: int


class TopicItem(BaseModel):
    """Validated topic item for quiz generation."""

    topic: str = Field(description="A short and precise subtopic.")

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Topic is too short.")
        if len(value.split()) > 6:
            raise ValueError("Topic is too long.")
        return value


class TopicList(BaseModel):
    """Validated list of subtopics."""

    topics: List[TopicItem]

    @field_validator("topics")
    @classmethod
    def validate_topics(cls, value: List[TopicItem]) -> List[TopicItem]:
        if not value:
            raise ValueError("At least one topic is required.")
        return value


class QuestionItem(BaseModel):
    """Validated quiz question."""

    question: str
    sources: List[str]

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 10:
            raise ValueError("Question is too short.")
        if len(value) > 300:
            raise ValueError("Question is too long.")
        return value

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("Question must include at least one source.")
        return value[:3]


class AnswerItem(BaseModel):
    """Validated correct answer object."""

    answer: str
    explanation: str
    sources: List[str]

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Answer is missing.")
        return value

    @field_validator("explanation")
    @classmethod
    def validate_explanation(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 5:
            raise ValueError("Explanation is too short.")
        return value

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("Answer must include at least one source.")
        return value[:3]


class DistractorItem(BaseModel):
    """Validated distractor object."""

    incorrect_answer: str
    explanation: str
    sources: List[str]

    @field_validator("incorrect_answer")
    @classmethod
    def validate_incorrect_answer(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Incorrect answer is too short.")
        return value

    @field_validator("explanation")
    @classmethod
    def validate_explanation(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 5:
            raise ValueError("Explanation is too short.")
        return value

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("Distractor must include at least one source.")
        return value[:3]


def load_config() -> AppConfig:
    """Load configuration directly from environment variables."""

    def must(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise RuntimeError(f"Missing required env var: {name}")
        return value

    return AppConfig(
        project_id=must("GCP_PROJECT_ID"),
        region=must("GCP_REGION"),
        instance=must("CLOUD_SQL_INSTANCE"),
        database=must("CLOUD_SQL_DATABASE"),
        db_user=must("CLOUD_SQL_USER"),
        db_pass=must("CLOUD_SQL_PASSWORD"),
        gcs_bucket=must("GCS_BUCKET_NAME"),
        gcs_prefix=os.getenv("GCS_PDF_PREFIX", "notes"),
        gcs_out_prefix=os.getenv("GCS_OUT_PREFIX", "knowledge_base/quizzes"),
        groq_api_key=must("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        groq_base_url=os.getenv(
            "GROQ_BASE_URL",
            "https://api.groq.com/openai/v1/chat/completions",
        ),
        table_name=os.getenv("VECTOR_TABLE_NAME", "course_embeddings"),
        vector_size=int(os.getenv("VECTOR_SIZE", "384")),
    )


def utc_compact_ts() -> str:
    """Return a compact UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace for deduplication and prompt stability."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """Remove problematic characters before embedding or prompting."""
    return text.replace("\x00", "").strip()


def short_sha1(text: str) -> str:
    """Return a short deterministic hash string."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def safe_json_extract(text: str) -> Any:
    """Extract the first JSON object or array from model output."""
    array_match = re.search(r"(\[\s*[\s\S]*\s*\])", text)
    if array_match:
        return json.loads(array_match.group(1))

    object_match = re.search(r"(\{\s*[\s\S]*\s*\})", text)
    if object_match:
        return json.loads(object_match.group(1))

    raise ValueError("No JSON payload found in model output.")


def dedupe_documents(documents: Sequence[Document], max_docs: int) -> List[Document]:
    """Deduplicate retrieved documents based on normalized content."""
    seen = set()
    results: List[Document] = []

    for doc in documents:
        key = normalize_whitespace(doc.page_content)[:240]
        if key in seen:
            continue
        seen.add(key)
        results.append(doc)
        if len(results) >= max_docs:
            break

    return results


def format_context_blocks(documents: Sequence[Document]) -> str:
    """Format retrieved documents into chunk-labeled prompt blocks."""
    blocks: List[str] = []

    for index, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source", "")
        page = metadata.get("page", metadata.get("page_number", ""))
        chunk_id = metadata.get("chunk_id", "")
        blocks.append(
            f"[CHUNK {index}] chunk_id={chunk_id} source={source} page={page}\n{doc.page_content}"
        )

    return "\n\n".join(blocks)


def list_pdf_blob_names(
    project_id: str,
    bucket_name: str,
    prefix: str,
    max_pdfs: int,
) -> List[str]:
    """List PDF blob names from GCS under a prefix."""
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    blob_names: List[str] = []
    for blob in client.list_blobs(bucket, prefix=prefix):
        if blob.name.lower().endswith(".pdf"):
            blob_names.append(blob.name)
            if len(blob_names) >= max_pdfs:
                break

    return blob_names


def load_documents_from_gcs(cfg: AppConfig, max_pdfs: int) -> List[Document]:
    """Load PDF pages from GCS into LangChain Document objects."""
    blob_names = list_pdf_blob_names(
        project_id=cfg.project_id,
        bucket_name=cfg.gcs_bucket,
        prefix=cfg.gcs_prefix,
        max_pdfs=max_pdfs,
    )

    if not blob_names:
        raise RuntimeError(f"No PDFs found at gs://{cfg.gcs_bucket}/{cfg.gcs_prefix}")

    documents: List[Document] = []
    for index, blob_name in enumerate(blob_names, start=1):
        loader = GCSFileLoader(
            project_name=cfg.project_id,
            bucket=cfg.gcs_bucket,
            blob=blob_name,
            loader_func=PyPDFLoader,
        )
        loaded_docs = loader.load()
        documents.extend(loaded_docs)
        print(f"[LOAD] {index}/{len(blob_names)} {blob_name} pages={len(loaded_docs)}")

    print(f"Loaded {len(documents)} pages from GCS.")
    return documents

def split_text(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Split source documents into chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        source = metadata.get("source", "")
        metadata["chunk_id"] = f"{short_sha1(str(source))}_{index}"
        chunk.metadata = metadata
        chunk.page_content = clean_text(chunk.page_content)

    chunks = [chunk for chunk in chunks if chunk.page_content]
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


async def get_engine(cfg: AppConfig) -> PostgresEngine:
    """Create a Cloud SQL Postgres engine."""
    return await PostgresEngine.afrom_instance(
        project_id=cfg.project_id,
        region=cfg.region,
        instance=cfg.instance,
        database=cfg.database,
        user=cfg.db_user,
        password=cfg.db_pass,
    )


async def init_vector_table(cfg: AppConfig, overwrite: bool) -> None:
    """Create or recreate the vector table used for retrieval."""
    engine = await get_engine(cfg)
    await engine.ainit_vectorstore_table(
        table_name=cfg.table_name,
        vector_size=cfg.vector_size,
        overwrite_existing=overwrite,
    )


async def get_vector_store(cfg: AppConfig) -> PostgresVectorStore:
    """Create a Postgres vector store backed by Cloud SQL pgvector."""
    engine = await get_engine(cfg)
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return await PostgresVectorStore.create(
        engine,
        embedding_service=embedding_model,
        table_name=cfg.table_name,
    )


async def save_to_cloud_sql(cfg: AppConfig, chunks: List[Document], batch_size: int) -> Dict[str, Any]:
    """Embed chunk texts and upload them into the vector table."""
    print("\nConnecting to Cloud SQL...")
    vector_store = await get_vector_store(cfg)

    texts = [clean_text(chunk.page_content) for chunk in chunks]
    metadatas = [dict(chunk.metadata or {}) for chunk in chunks]

    start_time = time.perf_counter()
    uploaded = 0

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_metadatas = metadatas[start : start + batch_size]
        await vector_store.aadd_texts(texts=batch_texts, metadatas=batch_metadatas)
        uploaded += len(batch_texts)
        print(f"Uploaded {uploaded}/{len(texts)} chunks...")

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    print(f"\nSuccessfully saved {len(chunks)} chunks to Cloud SQL table '{cfg.table_name}'.")
    return {
        "chunks_uploaded": len(chunks),
        "elapsed_ms": elapsed_ms,
        "table_name": cfg.table_name,
    }


async def retrieve_context(cfg: AppConfig, query_text: str, retrieval_k: int, max_docs: int) -> List[Document]:
    """Retrieve relevant chunks from the vector store."""
    vector_store = await get_vector_store(cfg)

    queries = [
        query_text,
        f"{query_text} database systems",
        f"{query_text} CMU DBMS",
    ]

    all_docs: List[Document] = []
    for query in queries:
        docs = await vector_store.asimilarity_search(query, k=retrieval_k)
        all_docs.extend(docs)

    return dedupe_documents(all_docs, max_docs=max_docs)


def build_topics_prompt(user_text: str, documents: List[Document], num_questions: int) -> str:
    """Build the prompt for generating quiz subtopics."""
    context = format_context_blocks(documents)
    return f"""
You are generating quiz subtopics for a Database Systems course.

User request:
{user_text}

Using only the context below, produce exactly {num_questions} distinct subtopics.
Each subtopic must be concise, specific, and useful for writing one quiz question.

Output JSON only in this format:
{{
  "topics": [
    {{"topic": "string"}}
  ]
}}

Context:
{context}
""".strip()


def build_question_prompt(subtopic: str, documents: List[Document]) -> str:
    """Build the prompt for generating one question from a subtopic."""
    context = format_context_blocks(documents)
    return f"""
You are generating a single quiz question for a Database Systems course.

Subtopic:
{subtopic}

Rules:
- Use only the context below.
- Write one precise question.
- Include 1 to 3 supporting chunk references.

Output JSON only in this format:
{{
  "question": "string",
  "sources": ["CHUNK 1", "CHUNK 2"]
}}

Context:
{context}
""".strip()


def build_correct_answer_prompt(question: str, documents: List[Document]) -> str:
    """Build the prompt for generating the correct answer and explanation."""
    context = format_context_blocks(documents)
    return f"""
You are generating the correct answer for a quiz question in a Database Systems course.

Question:
{question}

Rules:
- Use only the context below.
- Answer precisely.
- Provide a short explanation.
- Include 1 to 3 supporting chunk references.

Output JSON only in this format:
{{
  "answer": "string",
  "explanation": "string",
  "sources": ["CHUNK 1", "CHUNK 2"]
}}

Context:
{context}
""".strip()


def build_distractor_prompt(
    question: str,
    correct_answer: str,
    previous_answers: List[str],
    documents: List[Document],
) -> str:
    """Build the prompt for generating one plausible distractor."""
    context = format_context_blocks(documents)
    return f"""
You are generating one plausible but incorrect answer for a quiz question.

Question:
{question}

Correct answer:
{correct_answer}

Previous incorrect answers that must not be repeated:
{json.dumps(previous_answers)}

Rules:
- Use only the context below.
- The incorrect answer must be realistic and concise.
- It must be clearly wrong, but plausible enough to mislead a student.
- It must differ meaningfully from previous incorrect answers.
- Provide a short explanation of why it is incorrect.
- Include 1 to 3 chunk references.

Output JSON only in this format:
{{
  "incorrect_answer": "string",
  "explanation": "string",
  "sources": ["CHUNK 1"]
}}

Context:
{context}
""".strip()


async def call_groq_json(cfg: AppConfig, prompt: str) -> Any:
    """Call Groq's chat-completions endpoint and parse JSON from the response."""
    headers = {
        "Authorization": f"Bearer {cfg.groq_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": cfg.groq_model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise quiz-generation assistant. Return valid JSON only.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            cfg.groq_base_url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as response:
            response.raise_for_status()
            body = await response.json()

    raw_text = body["choices"][0]["message"]["content"].strip()
    return safe_json_extract(raw_text)


async def generate_topics(
    cfg: AppConfig,
    user_text: str,
    documents: List[Document],
    num_questions: int,
) -> List[str]:
    """Generate validated quiz subtopics."""
    payload = await call_groq_json(
        cfg,
        build_topics_prompt(user_text=user_text, documents=documents, num_questions=num_questions),
    )
    validated = TopicList.model_validate(payload)
    return [item.topic for item in validated.topics[:num_questions]]


async def generate_question_for_topic(
    cfg: AppConfig,
    subtopic: str,
    documents: List[Document],
) -> QuestionItem:
    """Generate one validated question for a given subtopic."""
    payload = await call_groq_json(
        cfg,
        build_question_prompt(subtopic=subtopic, documents=documents),
    )
    return QuestionItem.model_validate(payload)


async def generate_correct_answer(
    cfg: AppConfig,
    question: str,
    documents: List[Document],
) -> AnswerItem:
    """Generate one validated correct answer."""
    payload = await call_groq_json(
        cfg,
        build_correct_answer_prompt(question=question, documents=documents),
    )
    return AnswerItem.model_validate(payload)


async def generate_distractors(
    cfg: AppConfig,
    question: str,
    correct_answer: str,
    documents: List[Document],
    num_distractors: int = 3,
) -> List[DistractorItem]:
    """Generate distractors one at a time."""
    distractors: List[DistractorItem] = []
    previous_answers: List[str] = []

    for _ in range(num_distractors):
        payload = await call_groq_json(
            cfg,
            build_distractor_prompt(
                question=question,
                correct_answer=correct_answer,
                previous_answers=previous_answers,
                documents=documents,
            ),
        )
        item = DistractorItem.model_validate(payload)

        if item.incorrect_answer in previous_answers or item.incorrect_answer == correct_answer:
            continue

        distractors.append(item)
        previous_answers.append(item.incorrect_answer)

    return distractors


async def build_quiz_from_rag(
    cfg: AppConfig,
    user_text: str,
    num_questions: int,
    difficulty: str,
    question_types: List[str],
    retrieval_k: int,
    max_docs: int,
) -> Dict[str, Any]:
    """Generate a quiz using the current RAG pipeline and Groq for completions."""
    if "mcq" not in question_types:
        raise ValueError("This pipeline currently supports MCQ generation only.")

    documents = await retrieve_context(
        cfg=cfg,
        query_text=user_text,
        retrieval_k=retrieval_k,
        max_docs=max_docs,
    )

    if not documents:
        raise RuntimeError("No documents retrieved from vector store. Run ingest first.")

    topics = await generate_topics(
        cfg=cfg,
        user_text=user_text,
        documents=documents,
        num_questions=num_questions,
    )

    quiz_questions: List[Dict[str, Any]] = []

    for idx, topic in enumerate(topics, start=1):
        try:
            question_item = await generate_question_for_topic(cfg=cfg, subtopic=topic, documents=documents)
            answer_item = await generate_correct_answer(cfg=cfg, question=question_item.question, documents=documents)
            distractor_items = await generate_distractors(
                cfg=cfg,
                question=question_item.question,
                correct_answer=answer_item.answer,
                documents=documents,
                num_distractors=3,
            )

            options = [answer_item.answer] + [item.incorrect_answer for item in distractor_items]
            random.shuffle(options)

            quiz_questions.append(
                {
                    "id": f"q_{idx}",
                    "type": "mcq",
                    "topic": topic,
                    "question": question_item.question,
                    "options": options,
                    "answer": answer_item.answer,
                    "explanation": answer_item.explanation,
                    "sources": sorted(
                        list(
                            {
                                *question_item.sources,
                                *answer_item.sources,
                                *[src for item in distractor_items for src in item.sources],
                            }
                        )
                    ),
                    "incorrect_answers": [
                        {
                            "answer": item.incorrect_answer,
                            "explanation": item.explanation,
                            "sources": item.sources,
                        }
                        for item in distractor_items
                    ],
                }
            )
        except ValidationError as exc:
            print(f"[WARN] Validation failure for topic '{topic}': {exc}")
        except Exception as exc:
            print(f"[WARN] Generation failure for topic '{topic}': {exc}")

    if not quiz_questions:
        raise RuntimeError("Quiz generation failed for all topics.")

    quiz_id = f"quiz_{utc_compact_ts()}"

    return {
        "quiz_id": quiz_id,
        "difficulty": difficulty,
        "question_type": "mcq",
        "questions": quiz_questions,
        "run_metadata": {
            "generated_at_utc": utc_compact_ts(),
            "model": cfg.groq_model,
            "vector_table": cfg.table_name,
            "retrieval_k": retrieval_k,
            "max_docs": max_docs,
            "requested_questions": num_questions,
            "generated_questions": len(quiz_questions),
            "retrieved_chunks": [
                {
                    "chunk_id": (doc.metadata or {}).get("chunk_id"),
                    "source": (doc.metadata or {}).get("source"),
                    "page": (doc.metadata or {}).get("page", (doc.metadata or {}).get("page_number")),
                }
                for doc in documents
            ],
        },
    }


def write_json_to_gcs(cfg: AppConfig, object_path: str, payload: Any) -> str:
    """Write JSON to GCS and return the gs:// URI."""
    client = storage.Client(project=cfg.project_id)
    bucket = client.bucket(cfg.gcs_bucket)
    blob = bucket.blob(object_path)

    blob.upload_from_string(
        json.dumps(payload, indent=2, ensure_ascii=False),
        content_type="application/json",
    )
    return f"gs://{cfg.gcs_bucket}/{object_path}"


async def persist_quiz(cfg: AppConfig, quiz: Dict[str, Any]) -> str:
    """Persist the generated quiz JSON to GCS."""
    quiz_id = quiz.get("quiz_id", f"quiz_{utc_compact_ts()}")
    object_path = f"{cfg.gcs_out_prefix.rstrip('/')}/{quiz_id}.json"
    return write_json_to_gcs(cfg, object_path, quiz)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", required=True, choices=["ingest", "quiz"])
    parser.add_argument("--overwrite-table", action="store_true")

    parser.add_argument("--max-pdfs", type=int, default=40)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=500)
    parser.add_argument("--ingest-batch-size", type=int, default=100)

    parser.add_argument(
        "--text",
        type=str,
        default="Generate a database systems quiz from the course material.",
    )
    parser.add_argument("--num-questions", type=int, default=8)
    parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--types", type=str, default="mcq")
    parser.add_argument("--retrieval-k", type=int, default=6)
    parser.add_argument("--max-docs", type=int, default=12)

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    cfg = load_config()
    args = parse_args()

    if args.mode == "ingest":

        async def run_ingest() -> None:
            await init_vector_table(cfg, overwrite=bool(args.overwrite_table))

            documents = load_documents_from_gcs(
                cfg=cfg,
                max_pdfs=int(args.max_pdfs),
            )

            chunks = split_text(
                documents=documents,
                chunk_size=int(args.chunk_size),
                chunk_overlap=int(args.chunk_overlap),
            )

            result = await save_to_cloud_sql(
                cfg=cfg,
                chunks=chunks,
                batch_size=int(args.ingest_batch_size),
            )
            print(json.dumps({"ingest_result": result}, indent=2))

        asyncio.run(run_ingest())
        return

    if args.mode == "quiz":
        question_types = [item.strip() for item in str(args.types).split(",") if item.strip()]

        async def run_quiz() -> None:
            quiz = await build_quiz_from_rag(
                cfg=cfg,
                user_text=str(args.text).strip(),
                num_questions=int(args.num_questions),
                difficulty=str(args.difficulty),
                question_types=question_types,
                retrieval_k=int(args.retrieval_k),
                max_docs=int(args.max_docs),
            )

            uri = await persist_quiz(cfg, quiz)
            print("\nQuiz saved to:")
            print(uri)

        asyncio.run(run_quiz())
        return


if __name__ == "__main__":
    main()