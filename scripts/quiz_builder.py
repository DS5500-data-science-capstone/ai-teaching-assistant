#!/usr/bin/env python3
"""
quiz_builder.py

Generates multiple-choice quizzes from course material stored in Cloud SQL.
Retrieval is powered by OpenAI embeddings (must match create_database.py).
Quiz generation (topics, questions, answers, distractors) is powered by Groq.

Workflow:
    1. Load config from .env
    2. Connect to Cloud SQL vector store
    3. Retrieve relevant chunks using semantic search
    4. Generate subtopics → questions → answers → distractors via Groq
    5. Save the finished quiz JSON to GCS

Usage:
    python scripts/quiz_builder.py --text "database indexing" --num-questions 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import aiohttp
from dotenv import load_dotenv
from google.cloud import storage
from langchain_core.documents import Document
from langchain_google_cloud_sql_pg import PostgresEngine, PostgresVectorStore
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field, ValidationError, field_validator

# ---------------------------------------------------------------------------
# Environment setup
# Resolve the project root (one level above /scripts) and load the .env file.
# Also forwards the Google credentials path into the environment if set.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if google_credentials:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials


# ---------------------------------------------------------------------------
# Configuration
# A frozen dataclass that holds all runtime settings loaded from .env.
# Using a dataclass keeps config explicit and prevents accidental mutation.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration for quiz generation."""

    project_id: str       # GCP project ID
    region: str           # Cloud SQL region
    instance: str         # Cloud SQL instance name
    database: str         # Database name
    db_user: str          # Database user
    db_pass: str          # Database password
    gcs_bucket: str       # GCS bucket where quizzes are saved
    gcs_out_prefix: str   # GCS folder path for output quiz JSON files
    groq_api_key: str     # Groq API key for LLM calls
    groq_model: str       # Groq model name (e.g. llama-3.3-70b-versatile)
    groq_base_url: str    # Groq chat completions endpoint
    openai_api_key: str   # OpenAI API key for embeddings (must match create_database.py)
    table_name: str       # pgvector table name in Cloud SQL
    vector_size: int      # Embedding dimension — must be 1536 to match OpenAI


# ---------------------------------------------------------------------------
# Pydantic models
# Each model validates one piece of LLM output before it's used downstream.
# If the LLM returns malformed JSON, Pydantic raises a ValidationError which
# is caught in build_quiz_from_rag() so one bad question doesn't stop the run.
# ---------------------------------------------------------------------------

class TopicItem(BaseModel):
    """A single validated subtopic used to generate one quiz question."""

    topic: str = Field(description="A short and precise subtopic.")

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str) -> str:
        # Reject topics that are too vague (< 3 chars) or too broad (> 6 words)
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Topic is too short.")
        if len(value.split()) > 6:
            raise ValueError("Topic is too long.")
        return value


class TopicList(BaseModel):
    """A validated list of subtopics returned by the first Groq call."""

    topics: List[TopicItem]

    @field_validator("topics")
    @classmethod
    def validate_topics(cls, value: List[TopicItem]) -> List[TopicItem]:
        if not value:
            raise ValueError("At least one topic is required.")
        return value


class QuestionItem(BaseModel):
    """A validated quiz question with source chunk references."""

    question: str
    sources: List[str]

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        # Enforce a reasonable question length
        value = value.strip()
        if len(value) < 10:
            raise ValueError("Question is too short.")
        if len(value) > 300:
            raise ValueError("Question is too long.")
        return value

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, value: List[str]) -> List[str]:
        # Cap at 3 sources and require at least 1
        if not value:
            raise ValueError("Question must include at least one source.")
        return value[:3]


class AnswerItem(BaseModel):
    """A validated correct answer with explanation and source references."""

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
    """A validated incorrect answer (distractor) with explanation and sources."""

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


# ---------------------------------------------------------------------------
# Config loader
# Reads all required values from environment variables.
# Raises a clear RuntimeError if any required variable is missing so the user
# knows exactly what to add to their .env file.
# ---------------------------------------------------------------------------
def load_config() -> AppConfig:
    """Load configuration from environment variables."""

    def must(name: str) -> str:
        # Helper that raises immediately if a required env var is missing
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
        gcs_out_prefix=os.getenv("GCS_OUT_PREFIX", "knowledge_base/quizzes"),
        groq_api_key=must("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        groq_base_url=os.getenv(
            "GROQ_BASE_URL",
            "https://api.groq.com/openai/v1/chat/completions",
        ),
        openai_api_key=must("OPENAI_API_KEY"),
        table_name=os.getenv("VECTOR_TABLE_NAME", "course_embeddings"),
        vector_size=int(os.getenv("VECTOR_SIZE", "1536")),
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def utc_compact_ts() -> str:
    """Return a compact UTC timestamp string (e.g. 20240315T142500Z).
    Used to create unique quiz IDs and filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs into a single space.
    Used when generating deduplication keys so minor formatting differences
    don't cause the same content to appear twice."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """Strip null bytes and surrounding whitespace from text.
    PDF extraction sometimes leaves null bytes that break SQL inserts
    and cause issues in prompt construction."""
    return text.replace("\x00", "").strip()


def safe_json_extract(text: str) -> Any:
    """Extract the first valid JSON object or array from raw LLM output.
    Groq sometimes wraps JSON in markdown fences or adds explanation text —
    this function finds and parses just the JSON portion."""
    # Try to find a JSON array first (e.g. list of topics)
    array_match = re.search(r"(\[\s*[\s\S]*\s*\])", text)
    if array_match:
        return json.loads(array_match.group(1))

    # Fall back to finding a JSON object (e.g. a single question or answer)
    object_match = re.search(r"(\{\s*[\s\S]*\s*\})", text)
    if object_match:
        return json.loads(object_match.group(1))

    raise ValueError("No JSON payload found in model output.")


def dedupe_documents(documents: Sequence[Document], max_docs: int) -> List[Document]:
    """Remove duplicate chunks and cap the result at max_docs.
    Because we run multiple queries per topic, the same chunk can come back
    multiple times. Deduplication is based on the first 240 chars of content."""
    seen = set()
    results: List[Document] = []

    for doc in documents:
        # Use a normalized prefix as the dedup key
        key = normalize_whitespace(doc.page_content)[:240]
        if key in seen:
            continue
        seen.add(key)
        results.append(doc)
        if len(results) >= max_docs:
            break

    return results


def format_context_blocks(documents: Sequence[Document]) -> str:
    """Format a list of chunks into labeled blocks for use in prompts.
    Each block is labeled [CHUNK N] with its metadata so Groq can cite
    specific chunks in its responses (e.g. 'sources: ["CHUNK 1", "CHUNK 3"]')."""
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


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

async def get_engine(cfg: AppConfig) -> PostgresEngine:
    """Create and return an async Cloud SQL Postgres engine.
    This is the base connection used by the vector store."""
    return await PostgresEngine.afrom_instance(
        project_id=cfg.project_id,
        region=cfg.region,
        instance=cfg.instance,
        database=cfg.database,
        user=cfg.db_user,
        password=cfg.db_pass,
    )


async def get_vector_store(cfg: AppConfig) -> PostgresVectorStore:
    """Connect to the pgvector table using OpenAI embeddings.
    IMPORTANT: Must use OpenAIEmbeddings here to match what create_database.py
    used when ingesting — mismatched embeddings will return garbage results."""
    engine = await get_engine(cfg)
    # Set the API key in the environment so OpenAIEmbeddings can find it
    os.environ["OPENAI_API_KEY"] = cfg.openai_api_key
    embedding_model = OpenAIEmbeddings()  # produces 1536-dim vectors
    return await PostgresVectorStore.create(
        engine,
        embedding_service=embedding_model,
        table_name=cfg.table_name,
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

async def retrieve_context(cfg: AppConfig, query_text: str, retrieval_k: int, max_docs: int) -> List[Document]:
    """Retrieve the most relevant chunks from the vector store for a given query.

    Runs three variations of the query to improve recall:
      - The raw user query
      - Query + 'database systems' (domain context)
      - Query + 'CMU DBMS' (course-specific context)

    Results are deduplicated and capped at max_docs before being returned.
    """
    vector_store = await get_vector_store(cfg)

    # Multi-query strategy: slight variations improve recall from the vector store
    queries = [
        query_text,
        f"{query_text} database systems",
        f"{query_text} CMU DBMS",
    ]

    all_docs: List[Document] = []
    for query in queries:
        docs = await vector_store.asimilarity_search(query, k=retrieval_k)
        all_docs.extend(docs)

    # Deduplicate and cap results
    return dedupe_documents(all_docs, max_docs=max_docs)


# ---------------------------------------------------------------------------
# Prompt builders
# Each function constructs a prompt for one step of the quiz generation chain.
# All prompts include the retrieved chunks as context so Groq only uses
# information from the course material (not general knowledge).
# ---------------------------------------------------------------------------

def build_topics_prompt(user_text: str, documents: List[Document], num_questions: int) -> str:
    """Build the prompt that asks Groq to generate quiz subtopics.
    This is the first step — subtopics guide the question generation."""
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
    """Build the prompt that asks Groq to write one MCQ question for a subtopic."""
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
    """Build the prompt that asks Groq to generate the correct answer and explanation."""
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
    """Build the prompt that asks Groq to generate one plausible wrong answer.
    Previous distractors are passed in so Groq doesn't repeat them."""
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


# ---------------------------------------------------------------------------
# Groq API call
# ---------------------------------------------------------------------------

async def call_groq_json(cfg: AppConfig, prompt: str) -> Any:
    """Send a prompt to Groq and return the parsed JSON response.
    The system prompt instructs Groq to return JSON only, and safe_json_extract
    handles any extra text or markdown fences that sneak through."""
    headers = {
        "Authorization": f"Bearer {cfg.groq_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": cfg.groq_model,
        "temperature": 0.2,  # Low temperature = more deterministic, better JSON compliance
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

    # Extract the text content from the response and parse out the JSON
    raw_text = body["choices"][0]["message"]["content"].strip()
    return safe_json_extract(raw_text)


# ---------------------------------------------------------------------------
# Generation steps (each wraps one Groq call with Pydantic validation)
# ---------------------------------------------------------------------------

async def generate_topics(
    cfg: AppConfig,
    user_text: str,
    documents: List[Document],
    num_questions: int,
) -> List[str]:
    """Step 1: Ask Groq for a list of subtopics based on the user's request.
    Returns a plain list of topic strings, one per intended quiz question."""
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
    """Step 2: Ask Groq to write one question for a given subtopic.
    Returns a validated QuestionItem with the question text and source refs."""
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
    """Step 3: Ask Groq to generate the correct answer for a question.
    Returns a validated AnswerItem with the answer, explanation, and sources."""
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
    """Step 4: Ask Groq to generate plausible wrong answers one at a time.
    Each distractor call includes the previous ones so they stay distinct.
    Skips any distractor that duplicates a previous one or the correct answer."""
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

        # Skip if Groq repeated a previous distractor or the correct answer
        if item.incorrect_answer in previous_answers or item.incorrect_answer == correct_answer:
            continue

        distractors.append(item)
        previous_answers.append(item.incorrect_answer)

    return distractors


# ---------------------------------------------------------------------------
# Main quiz builder
# Orchestrates the full RAG + generation pipeline for one quiz run.
# ---------------------------------------------------------------------------

async def build_quiz_from_rag(
    cfg: AppConfig,
    user_text: str,
    num_questions: int,
    difficulty: str,
    retrieval_k: int,
    max_docs: int,
) -> Dict[str, Any]:
    """Orchestrate the full quiz generation pipeline.

    Steps:
        1. Retrieve relevant chunks from Cloud SQL via semantic search
        2. Generate subtopics from those chunks
        3. For each subtopic: generate a question, correct answer, and 3 distractors
        4. Shuffle answer options so the correct answer isn't always first
        5. Return the complete quiz as a structured dictionary

    Failed questions are skipped with a warning rather than stopping the run.
    """
    # Step 1: Retrieve relevant chunks from the vector store
    documents = await retrieve_context(
        cfg=cfg,
        query_text=user_text,
        retrieval_k=retrieval_k,
        max_docs=max_docs,
    )

    if not documents:
        raise RuntimeError("No documents retrieved from vector store. Run create_database.py first.")

    # Step 2: Generate subtopics to guide question creation
    topics = await generate_topics(
        cfg=cfg,
        user_text=user_text,
        documents=documents,
        num_questions=num_questions,
    )

    quiz_questions: List[Dict[str, Any]] = []

    # Steps 3 & 4: Generate one full MCQ per topic
    for idx, topic in enumerate(topics, start=1):
        try:
            # Generate the question
            question_item = await generate_question_for_topic(cfg=cfg, subtopic=topic, documents=documents)
            # Generate the correct answer
            answer_item = await generate_correct_answer(cfg=cfg, question=question_item.question, documents=documents)
            # Generate 3 distractors
            distractor_items = await generate_distractors(
                cfg=cfg,
                question=question_item.question,
                correct_answer=answer_item.answer,
                documents=documents,
                num_distractors=3,
            )

            # Shuffle options so the correct answer appears in a random position
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
                    # Merge and deduplicate all source references across the question
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
            # Pydantic rejected the LLM output — skip and warn
            print(f"[WARN] Validation failure for topic '{topic}': {exc}")
        except Exception as exc:
            # Any other failure (network, parse error, etc.) — skip and warn
            print(f"[WARN] Generation failure for topic '{topic}': {exc}")

    if not quiz_questions:
        raise RuntimeError("Quiz generation failed for all topics.")

    quiz_id = f"quiz_{utc_compact_ts()}"

    # Return the full quiz with metadata for traceability
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
            # Record which chunks were used so results are reproducible
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


# ---------------------------------------------------------------------------
# GCS output
# ---------------------------------------------------------------------------

def write_json_to_gcs(cfg: AppConfig, object_path: str, payload: Any) -> str:
    """Serialize payload to JSON and upload it to GCS.
    Returns the gs:// URI of the saved file."""
    client = storage.Client(project=cfg.project_id)
    bucket = client.bucket(cfg.gcs_bucket)
    blob = bucket.blob(object_path)

    blob.upload_from_string(
        json.dumps(payload, indent=2, ensure_ascii=False),
        content_type="application/json",
    )
    return f"gs://{cfg.gcs_bucket}/{object_path}"


async def persist_quiz(cfg: AppConfig, quiz: Dict[str, Any]) -> str:
    """Build the GCS output path from the quiz ID and save the quiz JSON.
    Output path format: <gcs_out_prefix>/<quiz_id>.json"""
    quiz_id = quiz.get("quiz_id", f"quiz_{utc_compact_ts()}")
    object_path = f"{cfg.gcs_out_prefix.rstrip('/')}/{quiz_id}.json"
    return write_json_to_gcs(cfg, object_path, quiz)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments for quiz generation."""
    parser = argparse.ArgumentParser(description="Generate a quiz from course materials.")

    parser.add_argument(
        "--text",
        type=str,
        default="Generate a database systems quiz from the course material.",
        help="Topic or prompt to generate the quiz around.",
    )
    parser.add_argument("--num-questions", type=int, default=8, help="Number of questions to generate.")
    parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--retrieval-k", type=int, default=6, help="Number of chunks to retrieve per query.")
    parser.add_argument("--max-docs", type=int, default=12, help="Max chunks to pass into the prompt context.")

    return parser.parse_args()


def main() -> None:
    """Entry point: load config, run the quiz pipeline, and save the result."""
    cfg = load_config()
    args = parse_args()

    async def run_quiz() -> None:
        quiz = await build_quiz_from_rag(
            cfg=cfg,
            user_text=args.text.strip(),
            num_questions=args.num_questions,
            difficulty=args.difficulty,
            retrieval_k=args.retrieval_k,
            max_docs=args.max_docs,
        )
        uri = await persist_quiz(cfg, quiz)
        print("\nQuiz saved to:")
        print(uri)

    asyncio.run(run_quiz())


if __name__ == "__main__":
    main()