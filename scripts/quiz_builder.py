#!/usr/bin/env python3
"""
quiz_builder.py

GCP quiz builder — Cloud SQL + OpenAI embeddings + Groq generation.

Usage:
    python scripts/quiz_builder.py --interactive
    python scripts/quiz_builder.py --text "database indexing" --num-questions 5
    python scripts/quiz_builder.py --text "database indexing" --questions-per-topic 3
    python scripts/quiz_builder.py --config my_quiz.json
    python scripts/quiz_builder.py --tag-topics
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import aiohttp
from dotenv import load_dotenv
from google.cloud import storage
from langchain_core.documents import Document
from langchain_google_cloud_sql_pg import PostgresEngine, PostgresVectorStore
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field, ValidationError, field_validator

# Configure module-level logger — all output goes through this instead of print()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Resolve project root so imports and .env work regardless of where the script is called from
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Prevent HuggingFace from making network calls when loading sentence-transformers offline
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Forward GCP credentials path into the environment if set in .env
creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if creds:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds

# GCS paths and bucket names used across the module
OUTPUT_PATH        = Path(os.getenv("OUTPUT_PATH", str(PROJECT_ROOT / "output")))
TAG_CACHE_PATH     = "knowledge_base/tags/chunk_tags.json"
QUIZ_OUTPUT_BUCKET = "quizzes-output"

# Default quiz configuration — used as a baseline for CLI and interactive modes
DEFAULT_CONFIG: Dict[str, Any] = {
    "topics":         [],
    "question_types": ["mcq"],
    "difficulty":     "medium",
    "style":          "conceptual",
    "num_options":    4,
    "marks_per_type": {"mcq": 1, "fill_blank": 1, "long_answer": 5, "true_false": 1},
    "source_filter":  None,
    "retrieval_k":    6,
    "max_docs":       12,
}

# Human-readable display names for each question type used in CLI output and prompts
QUESTION_TYPE_LABELS: Dict[str, str] = {
    "mcq":         "Multiple choice (MCQ)",
    "fill_blank":  "Fill in the blank",
    "long_answer": "Long answer / essay",
    "true_false":  "True or false",
}

# 24 topics from the CMU Database Systems syllabus — used for topic tagging and validation
SYLLABUS_TOPICS: List[str] = [
    "Relational Model & Algebra", "Modern SQL", "Database Storage I",
    "Memory Management", "Database Storage II", "Storage Models & Compression",
    "Hash Tables", "Indexes & Filters I", "Indexes & Filters II",
    "Index Concurrency Control", "Sorting & Aggregations Algorithms", "Joins Algorithms",
    "Query Execution I", "Query Execution II", "Query Planning & Optimization",
    "Concurrency Control Theory", "Two-Phase Locking Concurrency Control",
    "Timestamp Ordering Concurrency Control", "Multi-Version Concurrency Control",
    "Database Logging", "Database Recovery", "Introduction to Distributed Databases",
    "Distributed OLTP Database Systems", "Distributed OLAP Database Systems",
]

# Groq API pricing per token — used to calculate session cost
COST_PER_INPUT_TOKEN  = 0.05 / 1_000_000
COST_PER_OUTPUT_TOKEN = 0.08 / 1_000_000

# Hard spending cap per session — generation stops if this is reached
BUDGET_LIMIT_USD = 10.0

# Per-model rate limits for Groq — used to warn when approaching 80% usage
GROQ_MODEL_LIMITS: Dict[str, Dict[str, int]] = {
    "llama-3.3-70b-versatile": {"req_per_min": 500, "tokens_per_min": 100_000},
    "llama-3.1-8b-instant":    {"req_per_min": 500, "tokens_per_min": 100_000},
}

# In-memory prompt cache — avoids repeat API calls for identical prompts within a session
_groq_cache: Dict[str, Any] = {}

# In-memory budget tracker — resets to zero each time the script starts
_budget: Dict[str, Any] = {
    "requests_this_minute": 0,
    "tokens_this_minute":   0,
    "minute_start":         0.0,
    "total_requests":       0,
    "total_tokens":         0,
    "total_cost_usd":       0.0,
}


def validate_topic_weights(topics: List[Dict]) -> None:
    """Raise ValueError if topics list is empty or weights do not sum to exactly 100."""
    if not topics:
        raise ValueError("At least one topic is required.")
    total = sum(t.get("weight", 0) for t in topics)
    if total != 100:
        raise ValueError(
            f"Topic weights must sum to exactly 100%. "
            f"Current total: {total}%. Please adjust your topic weights."
        )


def validate_marks_per_type(marks_per_type: Dict[str, int], question_types: List[str]) -> None:
    """Raise ValueError if any active question type has marks less than 1."""
    for qtype in question_types:
        if marks_per_type.get(qtype, 0) < 1:
            raise ValueError(
                f"Marks for '{QUESTION_TYPE_LABELS.get(qtype, qtype)}' must be >= 1."
            )


def compute_total_marks(
    topics: List[Dict],
    marks_per_type: Dict[str, int],
    question_types: List[str],
) -> int:
    """Compute exact total marks based on actual question type distribution across all topics."""
    total = 0
    for t in topics:
        n = t["num_questions"]
        # Tile the type list to cover n questions then slice — ensures exact count
        assigned_types = (question_types * (n // len(question_types) + 1))[:n]
        total += sum(marks_per_type.get(qt, 1) for qt in assigned_types)
    return total


def compute_max_possible_marks(
    topics: List[Dict],
    marks_per_type: Dict[str, int],
    question_types: List[str],
) -> int:
    """
    Compute the maximum possible total marks if every question used the highest-value type.

    Used to detect impossible targets — e.g. target=100 but max achievable is 50.
    """
    max_mark = max(marks_per_type.get(qt, 1) for qt in question_types)
    total_questions = sum(t["num_questions"] for t in topics)
    return max_mark * total_questions


def compute_min_possible_marks(
    topics: List[Dict],
    marks_per_type: Dict[str, int],
    question_types: List[str],
) -> int:
    """
    Compute the minimum possible total marks if every question used the lowest-value type.

    Used alongside compute_max_possible_marks to show faculty the achievable range.
    """
    min_mark = min(marks_per_type.get(qt, 1) for qt in question_types)
    total_questions = sum(t["num_questions"] for t in topics)
    return min_mark * total_questions


def validate_total_marks_target(
    target:         int,
    topics:         List[Dict],
    marks_per_type: Dict[str, int],
    question_types: List[str],
) -> None:
    """
    Validate that a faculty-supplied total marks target is achievable and matched exactly.

    Checks:
    - Target is a positive integer.
    - All active question types have an explicit mark defined.
    - The computed total from the actual type distribution equals the target.
    - The target falls within the min/max achievable range.

    Raises ValueError with a clear message for each failure case.
    """
    if target < 5 or target % 5 != 0:
        lower = max(5, (target // 5) * 5)
        upper = lower + 5
        raise ValueError(
            f"Total marks target must be a multiple of 5 (minimum 5). "
            f"Got {target} — try {lower} or {upper}."
        )

    # Warn about any types falling back to default mark of 1 because they are missing
    missing = [qt for qt in question_types if qt not in marks_per_type]
    if missing:
        labels = [QUESTION_TYPE_LABELS.get(qt, qt) for qt in missing]
        raise ValueError(
            f"Marks not set for: {', '.join(labels)}. "
            f"Please add them to marks_per_type before setting a total target."
        )

    # Check whether the target is mathematically reachable
    min_marks = compute_min_possible_marks(topics, marks_per_type, question_types)
    max_marks = compute_max_possible_marks(topics, marks_per_type, question_types)
    if target < min_marks or target > max_marks:
        raise ValueError(
            f"Target of {target} marks is not achievable. "
            f"With your current settings the range is {min_marks}–{max_marks} marks. "
            f"Adjust marks_per_type or question counts to reach your target."
        )

    # Check that the actual computed distribution matches the target exactly
    computed = compute_total_marks(topics, marks_per_type, question_types)
    if computed != target:
        raise ValueError(
            f"Marks configuration produces {computed} marks but target is {target}. "
            f"Adjust marks_per_type values to match the target exactly."
        )


def _cache_key(prompt: str) -> str:
    """Return a SHA-256 hex digest of the prompt for use as a cache key."""
    return hashlib.sha256(prompt.encode()).hexdigest()


def _reset_minute_budget() -> None:
    """Reset per-minute counters if 60 seconds have elapsed since the last reset."""
    now = time.time()
    if now - _budget["minute_start"] >= 60:
        _budget["requests_this_minute"] = 0
        _budget["tokens_this_minute"]   = 0
        _budget["minute_start"]         = now


def _record_usage(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """
    Record token usage and cost for a single Groq API call.

    Logs a budget line, warns when rate limits approach 80%, and raises
    RuntimeError if the session spending limit has been reached.
    """
    _reset_minute_budget()
    total = prompt_tokens + completion_tokens
    cost  = (prompt_tokens * COST_PER_INPUT_TOKEN) + (completion_tokens * COST_PER_OUTPUT_TOKEN)

    _budget["requests_this_minute"] += 1
    _budget["tokens_this_minute"]   += total
    _budget["total_requests"]       += 1
    _budget["total_tokens"]         += total
    _budget["total_cost_usd"]       += cost

    session_cost = _budget["total_cost_usd"]
    remaining    = max(0.0, BUDGET_LIMIT_USD - session_cost)

    limits  = GROQ_MODEL_LIMITS.get(model, {"req_per_min": 30, "tokens_per_min": 6_000})
    req_pct = (_budget["requests_this_minute"] / limits["req_per_min"]) * 100
    tok_pct = (_budget["tokens_this_minute"] / limits["tokens_per_min"]) * 100

    logger.info(
        "Session: $%.4f / $%.2f  |  Remaining: $%.4f  |  %d req  |  %d tokens",
        session_cost, BUDGET_LIMIT_USD, remaining,
        _budget["total_requests"], _budget["total_tokens"],
    )

    if req_pct >= 80 or tok_pct >= 80:
        logger.warning(
            "Rate limit approaching: %.0f%% req/min, %.0f%% token/min used", req_pct, tok_pct
        )

    if session_cost >= BUDGET_LIMIT_USD:
        raise RuntimeError(
            f"[BUDGET] Hard stop: reached ${BUDGET_LIMIT_USD:.2f} session spending limit."
        )


def print_budget_summary() -> None:
    """Print a formatted session budget summary to stdout."""
    session_cost = _budget["total_cost_usd"]
    remaining    = max(0.0, BUDGET_LIMIT_USD - session_cost)
    print(f"\n[BUDGET] ── Session Summary ──────────────────────────────")
    print(f"[BUDGET]   Requests      : {_budget['total_requests']}")
    print(f"[BUDGET]   Tokens used   : {_budget['total_tokens']}")
    print(f"[BUDGET]   Session cost  : ${session_cost:.4f}")
    print(f"[BUDGET]   Limit         : ${BUDGET_LIMIT_USD:.2f}")
    print(f"[BUDGET]   Remaining     : ${remaining:.4f}")
    print(f"[BUDGET] ───────────────────────────────────────────────────")


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration loaded from environment variables."""

    project_id:     str
    region:         str
    instance:       str
    database:       str
    db_user:        str
    db_pass:        str
    gcs_bucket:     str
    gcs_out_prefix: str
    groq_api_key:   str
    groq_model:     str
    groq_base_url:  str
    openai_api_key: str
    table_name:     str
    vector_size:    int


def load_config() -> AppConfig:
    """
    Load and validate all required environment variables into an AppConfig.

    Raises RuntimeError for any missing required variable.
    """
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
        gcs_out_prefix=os.getenv("GCS_OUT_PREFIX", "knowledge_base/quizzes"),
        groq_api_key=must("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        groq_base_url=os.getenv(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions"
        ),
        openai_api_key=must("OPENAI_API_KEY"),
        table_name=os.getenv("VECTOR_TABLE_NAME", "course_embeddings"),
        vector_size=int(os.getenv("VECTOR_SIZE", "1536")),
    )


class TopicItem(BaseModel):
    """A single validated quiz subtopic string."""

    topic: str = Field(description="A short and precise subtopic.")

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Topic too short.")
        return v


class TopicList(BaseModel):
    """A non-empty list of TopicItem objects."""

    topics: List[TopicItem]

    @field_validator("topics")
    @classmethod
    def validate_topics(cls, v):
        if not v:
            raise ValueError("At least one topic required.")
        return v


class QuestionItem(BaseModel):
    """Validated question text with source citations."""

    question: str
    sources:  List[str]

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 10:
            raise ValueError("Question too short.")
        if len(v) > 500:
            raise ValueError("Question too long.")
        return v

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v):
        if not v:
            raise ValueError("Must include at least one source.")
        return v[:3]  # cap at 3 sources per question


class AnswerItem(BaseModel):
    """Validated correct answer with explanation and source citations."""

    answer:      str
    explanation: str
    sources:     List[str]

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Answer missing.")
        return v

    @field_validator("explanation")
    @classmethod
    def validate_explanation(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 5:
            raise ValueError("Explanation too short.")
        return v

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v):
        if not v:
            raise ValueError("Must include at least one source.")
        return v[:3]


class DistractorItem(BaseModel):
    """Validated MCQ distractor with explanation and source citations."""

    incorrect_answer: str
    explanation:      str
    sources:          List[str]

    @field_validator("incorrect_answer")
    @classmethod
    def validate_incorrect_answer(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Incorrect answer too short.")
        return v

    @field_validator("explanation")
    @classmethod
    def validate_explanation(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 5:
            raise ValueError("Explanation too short.")
        # If the explanation confirms the distractor rather than refuting it,
        # it is likely a true statement used as a wrong option.
        # Phrases like "this is correct", "this is true", "guarantees" used
        # affirmatively without "not" or "but" suggest the distractor may be true.
        lower = v.lower()
        affirmative_without_refutation = (
            ("guarantees" in lower or "always" in lower or "is correct" in lower)
            and "not" not in lower
            and "but" not in lower
            and "however" not in lower
            and "incorrect" not in lower
            and "false" not in lower
        )
        if affirmative_without_refutation:
            raise ValueError(
                "Distractor explanation appears to confirm rather than refute the option. "
                "The explanation must explain WHY this option is wrong, not why it sounds right."
            )
        return v

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v):
        if not v:
            raise ValueError("Must include at least one source.")
        return v[:3]


def utc_compact_ts() -> str:
    """Return the current UTC time as a compact string suitable for use in filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalise_sources(sources: Any) -> List[str]:
    """
    Normalise a raw sources value from the LLM into a clean list of CHUNK strings.

    The model sometimes returns bare numbers ("6"), bracket-wrapped ("[CHUNK 6]"),
    or plain strings ("CHUNK 6"). This function standardises all forms to "CHUNK N".
    """
    if not sources:
        return []
    if isinstance(sources, str):
        sources = [sources]
    result = []
    for s in sources:
        s = str(s).strip().strip("[]")
        # Already in correct form
        if re.match(r"^CHUNK\s+\d+$", s, re.IGNORECASE):
            result.append(f"CHUNK {s.split()[-1]}")
        # Bare number like "6" or "4"
        elif re.match(r"^\d+$", s):
            result.append(f"CHUNK {s}")
        # Something like "chunk6" or "chunk-6"
        elif re.match(r"^chunk\W*(\d+)$", s, re.IGNORECASE):
            num = re.search(r"\d+", s).group()
            result.append(f"CHUNK {num}")
        else:
            # Keep as-is if we can't parse it, but strip brackets
            result.append(s)
    return sorted(set(result))


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace sequences (tabs, newlines, spaces) into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """Replace null bytes with a space and strip leading/trailing whitespace."""
    return text.replace("\x00", " ").strip()


def sanitize_llm_output(text: str) -> str:
    """
    Aggressively clean raw LLM output before JSON parsing.

    Strips markdown fences, removes control characters, normalises smart quotes
    and non-breaking spaces, and escapes literal newlines/tabs inside JSON string
    values so json.loads does not raise Invalid control character errors.
    """
    # Strip markdown code fences that models sometimes wrap JSON in
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Remove ASCII control characters except tab (\t), newline (\n), carriage return (\r)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Replace curly/smart quotes with standard ASCII equivalents
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u00a0", " ")  # non-breaking space

    # Fix garbled mojibake characters that appear when PDF bullet points and dashes
    # are double-encoded. e.g. â€¢ is the latin-1 misread of UTF-8 bullet U+2022.
    # These appear in distractor explanations when chunk content comes from PDFs.
    text = text.replace("\u00e2\u0080\u00a2", "-")   # â€¢  -> bullet -> hyphen
    text = text.replace("\u00e2\u0080\u0093", "-")   # â€"  -> en-dash
    text = text.replace("\u00e2\u0080\u0094", "-")   # â€"  -> em-dash
    text = text.replace("\u00e2\u0080\u009c", '"')   # â€œ  -> open double quote
    text = text.replace("\u00e2\u0080\u009d", '"')   # â€   -> close double quote
    text = text.replace("\u2022", "-")               # bullet point -> hyphen
    text = text.replace("\u2013", "-")               # en-dash
    text = text.replace("\u2014", "-")               # em-dash

    # Escape literal unescaped newlines and tabs that appear inside JSON string values.
    # json.loads raises "Invalid control character" when a raw \n sits inside a quoted
    # string. This regex finds quoted strings and escapes control chars within them.
    def _escape_in_string(m: re.Match) -> str:
        inner = m.group(1)
        inner = inner.replace("\n", "\\n")
        inner = inner.replace("\r", "\\r")
        inner = inner.replace("\t", "\\t")
        return '"' + inner + '"'

    text = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', _escape_in_string, text)
    return text.strip()

def safe_json_extract(text: str) -> Any:
    """
    Extract and parse the first JSON object or array found in an LLM response string.

    Raises ValueError if no valid JSON structure is present.
    """
    text = sanitize_llm_output(text)
    # Try object first, then array — most Groq responses are objects
    obj = re.search(r"(\{\s*[\s\S]*\s*\})", text)
    if obj:
        return json.loads(obj.group(1))
    arr = re.search(r"(\[\s*[\s\S]*\s*\])", text)
    if arr:
        return json.loads(arr.group(1))
    raise ValueError("No JSON found in model output.")


def dedupe_documents(documents: Sequence[Document], max_docs: int) -> List[Document]:
    """
    Remove duplicate documents from a retrieval result set.

    Uses the first 240 normalised characters of page_content as a fingerprint.
    Returns at most max_docs unique documents in original order.
    """
    seen:    set             = set()
    results: List[Document] = []
    for doc in documents:
        # Truncate to 240 chars so near-duplicate chunks with slight differences are still deduped
        key = normalize_whitespace(doc.page_content)[:240]
        if key in seen:
            continue
        seen.add(key)
        results.append(doc)
        if len(results) >= max_docs:
            break
    return results


def format_context_blocks(documents: Sequence[Document]) -> str:
    """
    Format retrieved documents into numbered context blocks for LLM prompts.

    Each block includes the chunk index, chunk_id, source filename, page number,
    and raw page content so the model can cite sources accurately.
    """
    blocks = []
    for i, doc in enumerate(documents, 1):
        m        = doc.metadata or {}
        src      = m.get("source", "")
        page     = m.get("page", m.get("page_number", ""))
        chunk_id = m.get("chunk_id", "")
        blocks.append(
            f"[CHUNK {i}] chunk_id={chunk_id} source={src} page={page}\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


async def get_vector_store(cfg: AppConfig) -> PostgresVectorStore:
    """
    Create and return a PostgresVectorStore connected to Cloud SQL.

    Sets the OPENAI_API_KEY environment variable required by OpenAIEmbeddings.
    """
    engine = await PostgresEngine.afrom_instance(
        project_id=cfg.project_id,
        region=cfg.region,
        instance=cfg.instance,
        database=cfg.database,
        user=cfg.db_user,
        password=cfg.db_pass,
    )
    # OpenAIEmbeddings reads the key from the environment at instantiation time
    os.environ["OPENAI_API_KEY"] = cfg.openai_api_key
    return await PostgresVectorStore.create(
        engine,
        embedding_service=OpenAIEmbeddings(),
        table_name=cfg.table_name,
    )


async def retrieve_context(
    cfg:           AppConfig,
    query_text:    str,
    retrieval_k:   int,
    max_docs:      int,
    source_filter: Optional[str] = None,
    vs:            Optional[PostgresVectorStore] = None,
) -> List[Document]:
    """
    Retrieve and deduplicate relevant chunks from the vector store for a given query.

    Runs three query variants to maximise semantic coverage of the course material:
    plain query, with 'database systems', and with 'CMU DBMS'. Optionally filters
    results to a specific source PDF. Returns at most max_docs unique documents.
    """
    if vs is None:
        vs = await get_vector_store(cfg)

    # Three variants give broader retrieval coverage than a single query
    queries = [
        query_text,
        f"{query_text} database systems",
        f"{query_text} CMU DBMS",
    ]
    all_docs: List[Document] = []
    for q in queries:
        docs = await vs.asimilarity_search(q, k=retrieval_k)
        if source_filter:
            # Filter to chunks from a specific PDF if the faculty set a source filter
            docs = [
                d for d in docs
                if source_filter.lower() in str(d.metadata.get("source", "")).lower()
            ]
        all_docs.extend(docs)

    return dedupe_documents(all_docs, max_docs=max_docs)


def load_tag_cache(cfg: AppConfig) -> Dict[str, Dict]:
    """
    Load the chunk-to-topic tag cache JSON from GCS.

    Returns an empty dict if the blob does not exist or any error occurs,
    so callers can safely treat a missing cache as an empty one.
    """
    try:
        client = storage.Client(project=cfg.project_id)
        blob   = client.bucket(cfg.gcs_bucket).blob(TAG_CACHE_PATH)
        if not blob.exists():
            return {}
        logger.info("Loaded tag cache from gs://%s/%s", cfg.gcs_bucket, TAG_CACHE_PATH)
        return json.loads(blob.download_as_text())
    except Exception as e:
        logger.warning("Could not load tag cache: %s", e)
        return {}


def save_tag_cache(cfg: AppConfig, cache: Dict[str, Dict]) -> None:
    """Serialise and upload the tag cache dict to GCS as JSON."""
    try:
        client = storage.Client(project=cfg.project_id)
        blob   = client.bucket(cfg.gcs_bucket).blob(TAG_CACHE_PATH)
        blob.upload_from_string(
            json.dumps(cache, indent=2, ensure_ascii=False),
            content_type="application/json",
        )
        logger.info("Tag cache saved to gs://%s/%s", cfg.gcs_bucket, TAG_CACHE_PATH)
    except Exception as e:
        logger.warning("Could not save tag cache: %s", e)


async def tag_topics_with_keybert(cfg: AppConfig, force: bool = False) -> None:
    """
    Tag all chunks in the vector store with their closest syllabus topic using KeyBERT.

    Uses sentence-transformers (all-MiniLM-L6-v2) offline to embed chunk keywords
    and syllabus topics, then assigns each chunk the topic with the highest cosine
    similarity. Skips already-tagged chunks unless force=True.
    """
    try:
        from keybert import KeyBERT
        from sklearn.metrics.pairwise import cosine_similarity
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError("Run: pip install keybert sentence-transformers scikit-learn")

    cache = {} if force else load_tag_cache(cfg)
    if cache and not force:
        logger.info("Tag cache has %d entries. Only tagging new chunks.", len(cache))

    logger.info("Loading sentence transformer model...")
    model            = SentenceTransformer("all-MiniLM-L6-v2")
    kw_model         = KeyBERT(model=model)
    # Pre-encode all syllabus topics once so we can compare each chunk against them
    topic_embeddings = model.encode(SYLLABUS_TOPICS)

    vs = await get_vector_store(cfg)
    logger.info("Fetching all chunks from Cloud SQL...")
    results = await vs.asimilarity_search("database", k=500)
    logger.info("Found %d chunks.", len(results))

    new_tags = 0
    for doc in results:
        # Use MD5 of first 200 chars as a stable chunk fingerprint
        key = hashlib.md5(doc.page_content[:200].encode()).hexdigest()
        if key in cache and not force:
            continue
        if new_tags % 100 == 0 and new_tags > 0:
            logger.info("Tagged %d new chunks...", new_tags)
        try:
            kws     = kw_model.extract_keywords(
                doc.page_content, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=5
            )
            kw_text = " ".join(kw for kw, _ in kws) if kws else doc.page_content[:200]
            from sklearn.metrics.pairwise import cosine_similarity
            sims  = cosine_similarity(model.encode([kw_text]), topic_embeddings)[0]
            best  = int(sims.argmax())
            topic = SYLLABUS_TOPICS[best]
            conf  = float(sims[best])
        except Exception:
            # Fall back gracefully if keyword extraction fails for a chunk
            topic, conf = "General Database Systems", 0.0

        src  = Path(str(doc.metadata.get("source", ""))).stem
        page = doc.metadata.get("page", 0)
        cache[key] = {
            "topic":      topic,
            "topic_page": f"{topic} — p.{page}",
            "keywords":   topic,
            "source":     src,
            "confidence": round(conf, 3),
        }
        new_tags += 1

    if new_tags == 0:
        logger.info("All chunks already tagged. Use --force-retag to re-run.")
        return

    save_tag_cache(cfg, cache)
    logger.info("Done — %d chunks tagged.", new_tags)
    for topic, count in sorted(
        Counter(v["topic"] for v in cache.values()).items(), key=lambda x: -x[1]
    ):
        print(f"  {topic:<50} {count} chunks")


async def validate_topic_relevance(cfg: AppConfig, topic: str, threshold: int = 2) -> bool:
    """Return True if at least `threshold` chunks are retrievable for the given topic."""
    try:
        docs = await retrieve_context(cfg, topic, retrieval_k=3, max_docs=5)
        return len(docs) >= threshold
    except Exception:
        # If the check fails for any reason, allow the topic through
        return True


async def fetch_available_topics(cfg: AppConfig) -> List[str]:
    """
    Return the subset of SYLLABUS_TOPICS covered by chunks in the tag cache.

    Falls back to the full syllabus list if the cache is unavailable or empty.
    """
    try:
        cache = load_tag_cache(cfg)
        if cache:
            covered = {v["topic"] for v in cache.values()}
            return [t for t in SYLLABUS_TOPICS if t in covered]
    except Exception as e:
        logger.warning("Could not load tag cache: %s", e)
    return list(SYLLABUS_TOPICS)


def build_mcq_combined_prompt(
    subtopic:    str,
    documents:   List[Document],
    difficulty:  str = "medium",
    style:       str = "conceptual",
    num_options: int = 4,
) -> str:
    """
    Build a prompt that instructs the LLM to generate a complete MCQ in one response.

    Includes the question, correct answer, explanation, and exactly
    (num_options - 1) distractors, all grounded in the provided context chunks.
    """
    ctx    = format_context_blocks(documents)
    styles = {
        "conceptual": "Ask about the underlying concept or principle.",
        "scenario":   "Present a real-world scenario and ask the student to apply knowledge.",
        "definition": "Ask the student to identify or explain a key term.",
    }
    diffs = {
        "easy":   "Simple and direct. Answer should be straightforward.",
        "medium": "Requires some reasoning. Answer not immediately obvious.",
        "hard":   "Requires deep understanding or multi-step reasoning.",
    }
    n = num_options - 1  # number of distractors = options - 1 correct answer
    return f"""You are generating a complete multiple choice question for a Database Systems course.
Subtopic: {subtopic}
Style: {styles[style]}
Difficulty: {diffs[difficulty]}

GROUNDING RULES — you MUST follow these:
1. Read the context chunks below carefully.
2. Your question, correct answer, and all distractors must be based ONLY on the context.
3. For the answer and each distractor explanation, first quote the relevant sentence from the context, then explain it.
4. Do NOT introduce facts, terms, or concepts that do not appear in the context.
5. The "sources" field on the correct answer must list the CHUNK numbers that directly support it.
6. Every incorrect_answer object MUST have a non-empty "sources" list — if a distractor contrasts
   with the correct answer, cite the same chunk. Never leave sources as an empty list.

SELF-CHECK before returning — verify ALL of these:
- The "answer" field is a plain string, not an object or list.
- The "answer" is directly supported by the context — not just plausible.
- The "explanation" describes WHY the answer is correct, consistent with the answer text.
- The "answer" and "explanation" do not contradict each other.
- Each distractor is DEFINITELY FALSE based on the context — not a partially true statement.
- A distractor that is actually true (even if it sounds wrong) must be replaced with something clearly false.
- If a distractor describes what a DIFFERENT concept does (e.g. describing dynamic hashing as a
  wrong answer for a static hashing question), verify the statement is actually wrong for the
  concept being asked about.

Generate the question, correct answer, and exactly {n} incorrect answers in ONE response.
Return ONLY a single JSON object with keys: "question", "answer", "explanation", "incorrect_answers", "sources"
"incorrect_answers" must be a list of {n} objects each with keys: "incorrect_answer", "explanation", "sources"
The "answer" field must be a SHORT phrase of 15 words or fewer. Do NOT repeat the question in the answer.
The "incorrect_answer" fields must also be short phrases of 15 words or fewer.
Use only ASCII characters. No special quotes or unicode.
Context:
{ctx}""".strip()


def build_topics_prompt(user_text: str, documents: List[Document], n: int, question_index: int = 0) -> str:
    """
    Build a prompt that asks the LLM to generate exactly n subtopics from the provided context.

    question_index is passed so that when build_quiz is called once per question, each call
    requests a different angle on the topic rather than generating the same subtopic repeatedly.
    """
    ctx = format_context_blocks(documents)

    # Build a hint about which aspects to AVOID based on question_index so successive
    # single-question calls cover different angles of the same topic
    avoid_hints = [
        "",  # q0 — no constraint, pick the most important concept
        "Do NOT ask about static vs dynamic hashing or hash function trade-offs — those were already covered.",
        "Do NOT ask about static hashing, hash function trade-offs, or collision avoidance — those were already covered.",
        "Do NOT ask about static hashing, hash function design, collisions, or rebuilding — those were already covered.",
        "Do NOT ask about any concept already covered in previous questions — pick a completely different aspect.",
    ]
    avoid_hint = avoid_hints[min(question_index, len(avoid_hints) - 1)]

    return f"""You are generating a quiz subtopic for a Database Systems course.
User request: {user_text}
Question number: {question_index + 1}
{avoid_hint}

Using ONLY the context chunks below, produce exactly {n} distinct subtopic(s).

DIVERSITY RULES — you MUST follow these:
1. Each subtopic must cover a DIFFERENT concept, mechanism, or property found in the context.
2. Do NOT generate subtopics that ask about the same idea in different words.
3. Spread subtopics across different chunks — do not pull all subtopics from one chunk.
4. Bad example (too similar): "Hash function speed", "Hash function collision rate", "Choosing a hash function"
   These all ask about the same thing.
5. Good example (diverse): "Static vs dynamic hashing", "Collision handling strategies",
   "Hash function design trade-offs", "Linear probing", "Extendible hashing"
   Each covers a different aspect.
6. Each subtopic must be directly mentioned or clearly described in the context.
7. Do NOT invent subtopics not covered in the context.

Output JSON only: {{"topics": [{{"topic": "string"}}]}}
Use only ASCII characters.
Context:
{ctx}""".strip()


def build_fill_blank_prompt(
    subtopic:   str,
    documents:  List[Document],
    difficulty: str = "medium",
) -> str:
    """
    Build a prompt for generating a fill-in-the-blank question.

    The question must contain exactly one blank represented as _____.
    """
    ctx   = format_context_blocks(documents)
    diffs = {
        "easy":   "The blank should be a single well-known keyword.",
        "medium": "The blank should be a key term that requires understanding.",
        "hard":   "The blank should require deep knowledge to fill correctly.",
    }
    return f"""You are generating a fill-in-the-blank question for a Database Systems course.
Subtopic: {subtopic}
Difficulty: {diffs[difficulty]}

GROUNDING RULES — you MUST follow these:
1. The question and answer must come ONLY from the context chunks below.
2. The blank must replace a key term that appears in the context.
3. The explanation must quote the sentence from the context that contains the answer.
4. Do NOT use terms or facts that do not appear in the context.
5. The "sources" field must list the CHUNK numbers that contain the answer term.

SELF-CHECK before returning — verify ALL of these:
- The "answer" is a plain string directly found in the context.
- The "question" contains exactly one _____ blank.
- The "explanation" quotes the sentence from the context that contains the answer.
- The answer and explanation are consistent with each other.

Return ONLY a single JSON object with keys "question", "answer", "explanation", "sources".
The question must contain exactly one blank as _____.
Use only ASCII characters. No special quotes or unicode.
Context:
{ctx}""".strip()


def build_long_answer_prompt(
    subtopic:   str,
    documents:  List[Document],
    difficulty: str = "medium",
    style:      str = "conceptual",
) -> str:
    """
    Build a prompt for generating a long-answer question with a model answer and key points.

    Supports conceptual, scenario, and definition styles across three difficulty levels.
    """
    ctx    = format_context_blocks(documents)
    styles = {
        "conceptual": "Ask the student to explain a concept in their own words.",
        "scenario":   "Present a scenario and ask the student to analyse it.",
        "definition": "Ask the student to define and give examples of a key term.",
    }
    diffs = {
        "easy":   "Straightforward, 2-3 sentences expected.",
        "medium": "Requires understanding with examples, 1 paragraph expected.",
        "hard":   "Requires critical analysis or comparison, multiple paragraphs.",
    }
    return f"""You are generating a long-answer question for a Database Systems course.
Subtopic: {subtopic}
Style: {styles[style]}
Difficulty: {diffs[difficulty]}

GROUNDING RULES — you MUST follow these:
1. The question must ask about something explicitly covered in the context chunks below.
2. The model_answer must be based ONLY on the context. Do not add outside knowledge.
3. Each key point must correspond to a specific fact or concept from the context.
4. Quote relevant sentences from the context in your model_answer to ground it.
5. Do NOT include facts or explanations that are not supported by the context.
6. The "sources" field must list the CHUNK numbers that support your model_answer.

SELF-CHECK before returning — verify ALL of these:
- The "model_answer" only contains facts from the context — no outside knowledge.
- Each "key_point" maps to a specific fact in the context.
- The model_answer is consistent with the question being asked.

Return ONLY a single JSON object with keys "question", "model_answer", "key_points", "sources".
Use only ASCII characters. No special quotes or unicode.
Context:
{ctx}""".strip()


def build_true_false_prompt(
    subtopic:   str,
    documents:  List[Document],
    difficulty: str = "medium",
) -> str:
    """
    Build a prompt for generating a true/false question.

    The answer field in the returned JSON must be exactly 'True' or 'False'.
    """
    ctx   = format_context_blocks(documents)
    diffs = {
        "easy":   "The statement should be clearly true or false.",
        "medium": "The statement should be plausible and require knowledge to judge.",
        "hard":   "The statement should be subtle — a common misconception or edge case.",
    }
    return f"""You are generating a true/false question for a Database Systems course.
Subtopic: {subtopic}
Difficulty: {diffs[difficulty]}

GROUNDING RULES — you MUST follow these:
1. The statement must be directly based on something stated in the context chunks below.
2. For a True statement: it must be something explicitly confirmed in the context.
3. For a False statement: it must contradict something the context explicitly states.
4. The explanation must quote the specific sentence from the context that makes it true or false.
5. Do NOT create statements about topics not covered in the context.
6. The "sources" field must list the CHUNK numbers that contain the relevant sentence.

SELF-CHECK before returning — verify ALL of these:
- The "answer" is exactly "True" or "False", nothing else.
- The "explanation" quotes the specific sentence from the context that proves the answer.
- The statement and explanation are consistent with each other.
- A True statement is explicitly confirmed in the context.
- A False statement directly contradicts something in the context.

Return ONLY a single JSON object with keys "statement", "answer", "explanation", "sources".
"answer" must be exactly "True" or "False".
Use only ASCII characters. No special quotes or unicode.
Context:
{ctx}""".strip()


async def call_groq_json(cfg: AppConfig, prompt: str, max_retries: int = 15) -> Any:
    """
    Send a prompt to the Groq API and return the parsed JSON response.

    Features:
    - SHA-256 prompt caching to avoid duplicate API calls within a session.
    - Exponential backoff retry on HTTP 429 rate-limit responses (up to 60s wait).
    - Token usage and cost tracking via _record_usage.
    - Raises RuntimeError after max_retries consecutive rate-limit failures.
    """
    key = _cache_key(prompt)
    if key in _groq_cache:
        logger.info("Cache hit — skipping API call")
        return _groq_cache[key]

    _reset_minute_budget()
    headers = {
        "Authorization": f"Bearer {cfg.groq_api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       cfg.groq_model,
        "temperature": 0.2,  # low temperature keeps outputs focused and deterministic
        "messages": [
            {
                "role":    "system",
                "content": (
                    "You are a precise quiz-generation assistant for a university course. "
                    "CRITICAL RULE: You must base every question, answer, and explanation "
                    "ONLY on the context chunks provided in the user message. "
                    "Do NOT use your own training knowledge. "
                    "If the context does not contain enough information to write a question, "
                    "write the best question you can using ONLY what is in the context. "
                    "Every answer must be directly supported by the context. "
                    "Return valid JSON only. Use only ASCII characters in all responses."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    for attempt in range(max_retries):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                cfg.groq_base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=180),
            ) as resp:
                if resp.status == 429:
                    # Exponential backoff capped at 60s so we don't wait indefinitely
                    wait = min(2 ** attempt, 60)
                    logger.warning(
                        "Rate limit — waiting %ds (attempt %d/%d)", wait, attempt + 1, max_retries
                    )
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                body = await resp.json()

        usage = body.get("usage", {})
        _record_usage(
            cfg.groq_model,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
        result           = safe_json_extract(body["choices"][0]["message"]["content"].strip())
        _groq_cache[key] = result  # cache the result to avoid repeat calls for the same prompt
        return result

    raise RuntimeError("Groq rate limit exceeded after max retries.")


async def generate_topics(
    cfg:            AppConfig,
    text:           str,
    docs:           List[Document],
    n:              int,
    question_index: int = 0,
) -> List[str]:
    """
    Generate up to n focused subtopics from the provided documents using Groq.

    question_index is forwarded to the prompt so successive calls for the same topic
    ask for different angles rather than repeating the same subtopic.
    Handles both list-of-strings and list-of-dicts response formats from the model.
    Strips whitespace and filters empty strings from the result.
    """
    payload = await call_groq_json(cfg, build_topics_prompt(text, docs, n, question_index))

    # Handle case where model returns a plain list of strings instead of a dict
    if isinstance(payload, list):
        if all(isinstance(t, str) for t in payload):
            raw = [t.strip() for t in payload[:n]]
        else:
            payload = {"topics": payload}
            raw = []
    else:
        raw = []

    if isinstance(payload, dict) and "topics" in payload:
        for t in payload["topics"][:n]:
            if isinstance(t, str):
                raw.append(t.strip())
            elif isinstance(t, dict):
                raw.append(t.get("topic", "").strip())

    # Deduplicate subtopics — the model sometimes returns near-identical topics
    # e.g. "Hash function speed" and "Choosing a hash function" which produce the
    # same question. Keep the first occurrence of each unique normalised topic.
    seen: set = set()
    unique: List[str] = []
    for topic in raw:
        if not topic:
            continue
        # Normalise to lowercase with no punctuation for comparison
        key = re.sub(r"[^a-z0-9 ]", "", topic.lower()).strip()
        if key not in seen:
            seen.add(key)
            unique.append(topic)
        else:
            logger.warning("Duplicate subtopic filtered out: '%s'", topic)

    if len(unique) < n:
        logger.warning(
            "Only %d unique subtopics generated for %d requested — "
            "context may not cover enough distinct concepts.", len(unique), n
        )

    return unique


async def build_quiz(
    cfg:            AppConfig,
    user_text:      str,
    num_questions:  int,
    difficulty:     str,
    retrieval_k:    int,
    max_docs:       int,
    style:          str = "conceptual",
    num_options:    int = 4,
    source_filter:  Optional[str] = None,
    question_type:  str = "mcq",
    vs:             Optional[PostgresVectorStore] = None,
    question_index: int = 0,
) -> Dict[str, Any]:
    """
    Orchestrate the full quiz generation pipeline for a single topic and question type.

    Steps:
    1. Retrieve relevant chunks from the vector store.
    2. Generate focused subtopics from those chunks via Groq.
    3. For each subtopic, call Groq with a type-specific prompt and parse the response.
    4. Validate each response through Pydantic models.
    5. Retry once on failure with a cache-busting suffix; skip after two failures.
    6. Return a structured quiz dict with questions, metadata, and retrieved chunk info.

    Raises RuntimeError if no documents are retrieved or all questions fail generation.
    """
    docs = await retrieve_context(cfg, user_text, retrieval_k, max_docs, source_filter, vs=vs)
    if not docs:
        raise RuntimeError("No documents retrieved from Cloud SQL. Run create_database.py first.")

    topics         = await generate_topics(cfg, user_text, docs, num_questions, question_index)
    quiz_questions: List[Dict[str, Any]] = []

    for idx, topic in enumerate(topics, 1):
        # Unique suffix per question busts the cache on retry without changing the core prompt
        suffix = f"\n# q={question_index + idx}"
        entry: Dict[str, Any] = {"id": f"q_{idx}", "type": question_type, "topic": topic}
        success = False

        for attempt in range(2):  # try up to 2 times per question before skipping
            try:
                if question_type == "mcq":
                    raw = await call_groq_json(
                        cfg,
                        build_mcq_combined_prompt(topic, docs, difficulty, style, num_options) + suffix,
                    )
                    if isinstance(raw, list):
                        raise ValueError("Unexpected list from Groq")

                    # Guard: the model sometimes returns the answer as a distractor
                    # object {incorrect_answer: ...} instead of a plain string.
                    # Extract the string value if that happens.
                    raw_answer = raw.get("answer", "")
                    if isinstance(raw_answer, dict):
                        raw_answer = (
                            raw_answer.get("incorrect_answer")
                            or raw_answer.get("answer")
                            or ""
                        )
                        logger.warning(
                            "MCQ answer was a dict — extracted string: '%s'", raw_answer
                        )
                    if not isinstance(raw_answer, str) or not raw_answer.strip():
                        raise ValueError("MCQ answer is missing or not a string")

                    # Reject answers that are too long — the model sometimes writes a full
                    # sentence as the answer instead of a short phrase. 25 words is generous
                    # but catches the worst cases like Q2 above (40 words).
                    answer_word_count = len(raw_answer.split())
                    if answer_word_count > 25:
                        logger.warning(
                            "MCQ answer too long (%d words) for '%s' — retrying.",
                            answer_word_count, topic
                        )
                        raise ValueError(
                            f"Answer is {answer_word_count} words — must be 25 words or fewer. "
                            "Write a short phrase, not a full sentence."
                        )

                    # Semantic consistency check — reject if the explanation contradicts the answer.
                    # This catches the common hallucination where the model picks a plausible-sounding
                    # option as the correct answer but writes an explanation for a different concept.
                    explanation = raw.get("explanation", "")
                    if explanation:
                        answer_words     = set(raw_answer.lower().split())
                        explanation_words = set(explanation.lower().split())
                        # If answer and explanation share very few words it may be a mismatch.
                        # Use a loose overlap check — at least 1 meaningful word in common.
                        stopwords = {"the", "a", "an", "is", "are", "was", "were", "it", "they",
                                     "that", "this", "of", "in", "to", "and", "or", "for", "not",
                                     "be", "by", "at", "as", "on", "with", "its", "their", "have"}
                        answer_content      = answer_words - stopwords
                        explanation_content = explanation_words - stopwords
                        overlap = answer_content & explanation_content
                        if answer_content and explanation_content and len(overlap) == 0:
                            logger.warning(
                                "MCQ answer-explanation mismatch detected for '%s' — "
                                "answer '%s' shares no keywords with explanation. Retrying.",
                                topic, raw_answer[:60]
                            )
                            raise ValueError(
                                f"Answer '{raw_answer[:60]}' appears inconsistent with explanation. "
                                "Answer and explanation must describe the same concept."
                            )

                    # Combine correct answer and distractors then shuffle so answer position varies
                    distractors = raw.get("incorrect_answers", [])

                    # Distractor truthfulness check — warn if any distractor text appears
                    # verbatim or near-verbatim in the chunk content, which means it may
                    # actually be true. Log a warning so faculty can review these cases.
                    chunk_text_combined = " ".join(
                        normalize_whitespace(d.page_content) for d in docs
                    ).lower()
                    for dist in distractors:
                        if not isinstance(dist, dict):
                            continue
                        dist_text = dist.get("incorrect_answer", "").lower()
                        if not dist_text:
                            continue
                        # Check if the core words of the distractor appear in the chunks
                        dist_words = set(dist_text.split()) - {
                            "the", "a", "an", "is", "are", "was", "were", "it", "they",
                            "that", "this", "of", "in", "to", "and", "or", "for", "not",
                            "be", "by", "at", "as", "on", "with", "its", "their", "have",
                            "when", "from", "but", "which", "will", "can", "does",
                        }
                        if len(dist_words) >= 4:
                            # Count how many content words appear in the chunk text
                            found = sum(1 for w in dist_words if w in chunk_text_combined)
                            overlap_pct = found / len(dist_words)
                            if overlap_pct >= 0.7:
                                logger.warning(
                                    "[DISTRACTOR] '%s' shares %.0f%% words with chunk content "
                                    "— may be a true statement used as a wrong option. "
                                    "Review manually.",
                                    dist_text[:80], overlap_pct * 100
                                )
                    opts = [raw_answer] + [
                        d.get("incorrect_answer", "") if isinstance(d, dict) else str(d)
                        for d in distractors
                    ]
                    # Filter out any options that are empty or are themselves dicts
                    opts = [o for o in opts if isinstance(o, str) and o.strip()]
                    if len(opts) < 2:
                        raise ValueError("MCQ has fewer than 2 valid options after parsing")
                    random.shuffle(opts)
                    entry.update({
                        "question":    raw.get("question", ""),
                        "options":     opts,
                        "answer":      raw_answer,
                        "explanation": raw.get("explanation", ""),
                        "sources":     normalise_sources(raw.get("sources", [])),
                        "incorrect_answers": [
                            {
                                "answer":      d.get("incorrect_answer", "") if isinstance(d, dict) else str(d),
                                "explanation": d.get("explanation", "") if isinstance(d, dict) else "",
                                "sources":     normalise_sources(d.get("sources", []) if isinstance(d, dict) else []),
                            }
                            for d in distractors
                        ],
                    })

                elif question_type == "fill_blank":
                    r = await call_groq_json(
                        cfg, build_fill_blank_prompt(topic, docs, difficulty) + suffix
                    )
                    if isinstance(r, list):
                        raise ValueError("Unexpected list from Groq")
                    entry.update({
                        "question":    r.get("question", ""),
                        "answer":      r.get("answer", ""),
                        "explanation": r.get("explanation", ""),
                        "sources":     normalise_sources(r.get("sources", [])),
                    })

                elif question_type == "long_answer":
                    r = await call_groq_json(
                        cfg, build_long_answer_prompt(topic, docs, difficulty, style) + suffix
                    )
                    if isinstance(r, list):
                        raise ValueError("Unexpected list from Groq")
                    entry.update({
                        "question":     r.get("question", ""),
                        "model_answer": r.get("model_answer", ""),
                        "key_points":   r.get("key_points", []),
                        "sources":      normalise_sources(r.get("sources", [])),
                    })

                elif question_type == "true_false":
                    r = await call_groq_json(
                        cfg, build_true_false_prompt(topic, docs, difficulty) + suffix
                    )
                    if isinstance(r, list):
                        raise ValueError("Unexpected list from Groq")
                    entry.update({
                        "statement":   r.get("statement", ""),
                        "answer":      r.get("answer", ""),
                        "explanation": r.get("explanation", ""),
                        "sources":     normalise_sources(r.get("sources", [])),
                    })

                quiz_questions.append(entry)
                logger.info("[OK] %s %d/%d: %s", question_type, idx, len(topics), topic)
                success = True
                break

            except ValidationError as exc:
                logger.warning(
                    "Validation failure for '%s' (attempt %d): %s", topic, attempt + 1, exc
                )
            except Exception as exc:
                logger.warning(
                    "Generation failure for '%s' (attempt %d): %s", topic, attempt + 1, exc
                )
                suffix += "_retry"  # mutate suffix so the retry prompt gets a different cache key

        if not success:
            logger.warning("[SKIP] Skipping '%s' after 2 failed attempts.", topic)

    # Only raise if every single subtopic failed — a partial result is still usable
    if not quiz_questions:
        raise RuntimeError(
            "Quiz generation failed for all subtopics. "
            "This is usually caused by the model returning malformed JSON. "
            "Try running again — Groq responses can vary between calls."
        )

    # Deduplicate questions by both question text and correct answer.
    # The same concept asked in different words produces the same answer
    # and should be treated as a duplicate.
    seen_q_text:   set = set()
    seen_q_answer: set = set()
    unique_questions: List[Dict[str, Any]] = []
    for q in quiz_questions:
        q_text   = q.get("question") or q.get("statement") or ""
        q_answer = q.get("answer") or q.get("model_answer") or ""
        q_key    = re.sub(r"[^a-z0-9 ]", "", q_text.lower()).strip()[:60]
        a_key    = re.sub(r"[^a-z0-9 ]", "", q_answer.lower()).strip()[:60]

        is_dup_q = q_key and q_key in seen_q_text
        is_dup_a = a_key and a_key in seen_q_answer

        if is_dup_q:
            logger.warning("[DEDUP] Duplicate question text removed: '%s'", q_text[:80])
        elif is_dup_a:
            logger.warning("[DEDUP] Same-answer duplicate removed: '%s'", q_answer[:80])
        else:
            if q_key:
                seen_q_text.add(q_key)
            if a_key:
                seen_q_answer.add(a_key)
            unique_questions.append(q)

    if len(unique_questions) < len(quiz_questions):
        logger.warning(
            "%d duplicate question(s) removed — context may not have enough "
            "distinct concepts for %d unique questions on this topic.",
            len(quiz_questions) - len(unique_questions), num_questions
        )

    return {
        "quiz_id":       f"quiz_{utc_compact_ts()}",
        "difficulty":    difficulty,
        "style":         style,
        "question_type": question_type,
        "questions":     unique_questions,
        "run_metadata": {
            "generated_at_utc":    utc_compact_ts(),
            "model":               cfg.groq_model,
            "vector_table":        cfg.table_name,
            "retrieval_k":         retrieval_k,
            "max_docs":            max_docs,
            "requested_questions": num_questions,
            "generated_questions": len(unique_questions),
            "style":               style,
            "difficulty":          difficulty,
            "num_options":         num_options,
            "source_filter":       source_filter or "all",
            # Record which chunks were used so answers can be traced back to source PDFs.
            # content is stored so tests and the UI can verify answer grounding against
            # the actual text that was passed to Groq as context.
            "retrieved_chunks": [
                {
                    "chunk_id": (d.metadata or {}).get("chunk_id"),
                    "source":   (d.metadata or {}).get("source"),
                    "page":     (d.metadata or {}).get("page", (d.metadata or {}).get("page_number")),
                    "content":  d.page_content,
                }
                for d in docs
            ],
        },
    }


def write_json_to_gcs(cfg: AppConfig, object_path: str, payload: Any) -> str:
    """
    Serialise payload as JSON and upload it to the quiz output GCS bucket.

    Returns the full gs:// URI of the uploaded object.
    """
    client = storage.Client(project=cfg.project_id)
    blob   = client.bucket(QUIZ_OUTPUT_BUCKET).blob(object_path)
    blob.upload_from_string(
        json.dumps(payload, indent=2, ensure_ascii=False),
        content_type="application/json",
    )
    return f"gs://{QUIZ_OUTPUT_BUCKET}/{object_path}"


async def persist_quiz(cfg: AppConfig, quiz: Dict[str, Any]) -> str:
    """Write a completed quiz dict to GCS and return the gs:// URI."""
    quiz_id = quiz.get("quiz_id", f"quiz_{utc_compact_ts()}")
    return write_json_to_gcs(cfg, f"{cfg.gcs_out_prefix.rstrip('/')}/{quiz_id}.json", quiz)


def _banner(text: str) -> None:
    """Print a simple section banner to stdout."""
    print(f"\n{'─' * 50}\n  {text}\n{'─' * 50}")


def _ask(prompt: str, default: str = "") -> str:
    """Prompt the user for input, returning default if they press Enter."""
    hint = f" [{default}]" if default else ""
    raw  = input(f"{prompt}{hint}: ").strip()
    return raw if raw else default


def _choose(prompt: str, options: List[str], default: str = "") -> str:
    """Present a numbered list of options and return the user's selection."""
    print(f"\n{prompt}")
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}{' (default)' if o == default else ''}")
    while True:
        raw = input("Enter number or value: ").strip()
        if not raw and default:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        if raw in options:
            return raw
        print(f"  Enter a number 1-{len(options)}")


def _choose_many(prompt: str, options: List[str], defaults: List[str]) -> List[str]:
    """Present a numbered list and allow the user to select multiple options by number."""
    print(f"\n{prompt}")
    for i, o in enumerate(options, 1):
        print(f"  {i}. {QUESTION_TYPE_LABELS.get(o, o)}{' (selected)' if o in defaults else ''}")
    print(f"  Enter numbers separated by commas. Press Enter for defaults: {', '.join(defaults)}")
    while True:
        raw = input("Selection: ").strip()
        if not raw:
            return defaults
        parts          = [p.strip() for p in raw.split(",")]
        selected: List[str] = []
        valid          = True
        for p in parts:
            if p.isdigit() and 1 <= int(p) <= len(options):
                selected.append(options[int(p) - 1])
            elif p in options:
                selected.append(p)
            else:
                print(f"  Invalid: '{p}'. Try again.")
                valid = False
                break
        if valid and selected:
            return selected


def _collect_topics(
    cfg:              AppConfig,
    available_topics: Optional[List[str]],
    total_questions:  int,
) -> List[Dict]:
    """
    Interactively collect topic names and percentage weights from the user.

    Validates each custom topic against the knowledge base. Ensures weights
    sum to exactly 100 and question counts are proportionally distributed.
    Returns a list of topic dicts with keys: topic, weight, num_questions.
    """
    topics_config: List[Dict] = []
    total_weight  = 0

    while total_weight < 100:
        remaining = 100 - total_weight
        print(f"  Weight remaining: {remaining}%")

        if available_topics:
            while True:
                raw = input(
                    "  Pick topic number or type a custom topic (Enter to finish): "
                ).strip()
                if not raw and topics_config:
                    break
                if raw.isdigit():
                    idx = int(raw)
                    if 1 <= idx <= len(available_topics):
                        topic = available_topics[idx - 1]
                        break
                    print(f"  Invalid number. Enter 1-{len(available_topics)}.")
                elif len(raw) >= 3:
                    topic = raw
                    logger.info("Checking '%s' against knowledge base...", topic)
                    if asyncio.run(validate_topic_relevance(cfg, topic)):
                        break
                    print(f"  '{topic}' does not appear in the course knowledge base.")
                else:
                    print("  Topic must be at least 3 characters or a valid topic number.")
            if not raw and topics_config:
                break
        else:
            topic = _ask("  Topic name", "database systems")

        while True:
            w = input(f"  Weight for '{topic}' (remaining: {remaining}%): ").strip()
            if w.isdigit() and 1 <= int(w) <= remaining:
                weight = int(w)
                break
            print(f"  Enter a number between 1 and {remaining}.")

        # Distribute questions proportionally but ensure at least 1 per topic
        allocated = sum(t["num_questions"] for t in topics_config)
        num_q     = max(1, round(total_questions * weight / 100))
        num_q     = min(num_q, total_questions - allocated)
        num_q     = max(1, num_q)

        topics_config.append({"topic": topic, "weight": weight, "num_questions": num_q})
        total_weight += weight
        print(f"  Added: {topic} ({weight}%, {num_q} questions)\n")

        if total_weight == 100:
            print("  Total weight reached 100%. Topics locked in.")
            break

    return topics_config


def _suggest_mark_combinations(
    topics_config:      List[Dict],
    question_types:     List[str],
    total_marks_target: int,
    max_suggestions:    int = 5,
) -> List[Dict[str, int]]:
    """
    Find valid marks_per_type combinations that produce exactly total_marks_target.

    Searches marks 1-20 per type and returns up to max_suggestions valid combinations.
    Used to give faculty concrete examples when their input does not match the target.
    """
    suggestions: List[Dict[str, int]] = []

    def search(idx: int, current: Dict[str, int]) -> None:
        if len(suggestions) >= max_suggestions:
            return
        if idx == len(question_types):
            # Only keep this combination if it hits the target exactly
            if compute_total_marks(topics_config, current, question_types) == total_marks_target:
                suggestions.append(dict(current))
            return
        qt = question_types[idx]
        for m in range(1, 21):
            current[qt] = m
            search(idx + 1, current)
            if len(suggestions) >= max_suggestions:
                return

    search(0, {})
    return suggestions


def _collect_marks(
    question_types:     List[str],
    marks_per_type:     Dict[str, int],
    total_marks_target: int,
    topics_config:      List[Dict],
) -> Dict[str, int]:
    """
    Interactively collect marks per question type, looping until totals match the target.

    Shows valid combinations as hints before faculty start entering values and on each
    mismatch so they are never left guessing. Returns a validated marks_per_type dict.
    """
    # Compute and show valid combinations upfront so faculty know what will work
    suggestions = _suggest_mark_combinations(topics_config, question_types, total_marks_target)
    if suggestions:
        print(f"  Valid combinations that reach {total_marks_target} marks:")
        for s in suggestions:
            parts = ", ".join(
                f"{QUESTION_TYPE_LABELS.get(qt, qt)}: {v} pt{'s' if v != 1 else ''}"
                for qt, v in s.items()
            )
            print(f"    {parts}")
        print()
    else:
        print(f"  No valid combinations found for target {total_marks_target}.")
        print(f"  Try adjusting your question count or total marks target.\n")

    while True:
        per_type: Dict[str, int] = {}
        for qtype in question_types:
            label = QUESTION_TYPE_LABELS.get(qtype, qtype)
            while True:
                raw = _ask(f"  Marks for {label}", str(marks_per_type.get(qtype, 1)))
                if raw.isdigit() and int(raw) >= 1:
                    per_type[qtype] = int(raw)
                    break
                print("  Enter a number >= 1.")
        computed = compute_total_marks(topics_config, per_type, question_types)
        if computed == total_marks_target:
            print(f"\n  Total marks: {computed}")
            return per_type
        # Show clearly how far off the total is and remind faculty of working options
        diff      = total_marks_target - computed
        direction = f"{abs(diff)} marks short" if diff > 0 else f"{abs(diff)} marks over"
        print(f"\n  Total would be {computed} ({direction}). Target is {total_marks_target}.")
        if suggestions:
            print(f"  Valid combinations: {', '.join(str(s) for s in suggestions[:3])}")
        print()


def _ask_total_marks() -> int:
    """
    Ask faculty for a total marks target that is a multiple of 5 (minimum 5).

    Suggests the nearest valid values when input is invalid.
    Returns the validated integer target.
    """
    print("  Total marks must be a multiple of 5 (e.g. 10, 20, 25, 50, 100).")
    while True:
        raw = input("  Total marks for this quiz: ").strip()
        if raw.isdigit():
            val = int(raw)
            if val >= 5 and val % 5 == 0:
                return val
            # Suggest nearest multiples of 5 above and below
            lower = max(5, (val // 5) * 5)
            upper = lower + 5
            if val < 5:
                print(f"  Minimum is 5.")
            elif val % 5 != 0:
                print(f"  {val} is not a multiple of 5. Try {lower} or {upper}.")
        else:
            print("  Enter a whole number that is a multiple of 5.")


def _ask_total_questions(total_marks_target: int, marks_mode_key: str) -> int:
    """
    Ask faculty for the total number of questions (1-100).

    In equal marks mode, also validates that total_marks_target divides evenly
    across the questions and prompts faculty to adjust if not. Returns the
    final validated question count.
    """
    while True:
        raw = input("  Total number of questions: ").strip()
        if not raw.isdigit() or not (1 <= int(raw) <= 100):
            print("  Enter a number between 1 and 100.")
            continue
        total_questions = int(raw)

        # In equal mode the marks must divide evenly — check immediately
        if marks_mode_key == "equal":
            if total_marks_target % total_questions != 0:
                marks_each = max(1, total_marks_target // total_questions)
                # Find the nearest question count that divides the target evenly
                lower_q = total_marks_target // (marks_each + 1) if marks_each + 1 > 0 else 1
                upper_q = total_marks_target // marks_each if marks_each > 0 else total_questions
                print(
                    f"  {total_marks_target} marks cannot be divided equally across "
                    f"{total_questions} questions ({total_marks_target / total_questions:.1f} each)."
                )
                print(f"  Try {lower_q} or {upper_q} questions for clean equal marks.")
                continue
            marks_each = total_marks_target // total_questions
            print(f"  Each question = {marks_each} mark(s)  →  Total: {total_marks_target} ✅")
        return total_questions


def build_config_interactively(
    cfg:              AppConfig,
    available_topics: Optional[List[str]] = None,
) -> Dict:
    """
    Guide the user through an interactive quiz configuration wizard.

    New order — marks are locked in first before any other customisation:
      1. Total marks (multiple of 5)
      2. Marks mode (equal or per type)
      3. Total questions
      4. Question types  [per type mode only: show distribution + valid combos here]
      5. Marks per type  [per type mode only]
      6. Topics and weightage
      7. Difficulty and style
      8. Summary and confirmation

    Returns a fully validated quiz configuration dict ready for generation.
    """
    _banner("Quiz configurator — faculty edition")
    print("Configure your quiz paper below.")
    print("Press Enter to accept the default shown in [brackets].\n")

    quiz_cfg = dict(DEFAULT_CONFIG)
    quiz_cfg["marks_per_type"] = dict(DEFAULT_CONFIG["marks_per_type"])

    if available_topics:
        print("Available topics in the knowledge base:")
        for i, t in enumerate(available_topics, 1):
            print(f"  {i}. {t}")
        print()
    else:
        print("(No topic list available — type topics manually)\n")

    # ── Step 0: Total marks — set before anything else ──────────────────────
    _banner("Step 0 — Total marks")
    print("  Set the total marks for this quiz first.")
    print("  All other settings will be configured around this target.\n")
    total_marks_target = _ask_total_marks()

    # ── Marks mode — chosen immediately after total marks ───────────────────
    marks_mode = _choose(
        "How should marks be distributed?",
        ["Equal marks per question", "Different marks per question type"],
        "Equal marks per question",
    )
    marks_mode_key = "equal" if "Equal" in marks_mode else "per_type"

    # ── Step 1: Total questions ──────────────────────────────────────────────
    _banner("Step 1 — Total questions")
    # In equal mode, validate divisibility immediately so faculty don't get
    # stuck later. In per type mode, divisibility is handled per type.
    print(f"  Target: {total_marks_target} marks  |  Mode: {'Equal per question' if marks_mode_key == 'equal' else 'Per question type'}\n")
    total_questions = _ask_total_questions(total_marks_target, marks_mode_key)

    # ── Step 2: Question types ───────────────────────────────────────────────
    _banner("Step 2 — Question types")
    quiz_cfg["question_types"] = _choose_many(
        "Which question types do you want?",
        list(QUESTION_TYPE_LABELS.keys()),
        quiz_cfg["question_types"],
    )

    # Validate that at least one type was selected — _choose_many can return
    # the defaults silently so guard against an empty list here
    if not quiz_cfg["question_types"]:
        print("  No question types selected. Defaulting to MCQ.")
        quiz_cfg["question_types"] = ["mcq"]

    # Show how questions will be distributed across types so faculty can
    # see the split before entering marks in the next step
    n_types = len(quiz_cfg["question_types"])
    base    = total_questions // n_types
    extra   = total_questions % n_types  # first 'extra' types get one more question
    print(f"\n  Distribution across {total_questions} questions:")
    type_counts: Dict[str, int] = {}
    for i, qt in enumerate(quiz_cfg["question_types"]):
        count = base + (1 if i < extra else 0)
        type_counts[qt] = count
        label = QUESTION_TYPE_LABELS.get(qt, qt)
        print(f"    {label:<30} {count} question{'s' if count != 1 else ''}")

    # ── Step 3: Marks per question type ─────────────────────────────────────
    _banner("Step 3 — Marks per question")
    if marks_mode_key == "equal":
        # Safe because _ask_total_questions already enforced divisibility
        marks_each = total_marks_target // total_questions
        quiz_cfg["marks_per_type"] = {qt: marks_each for qt in quiz_cfg["question_types"]}
        print(f"  Each question: {marks_each} mark(s)  →  Total: {total_marks_target} ✅\n")

    else:
        # Per type mode — compute achievable range using actual type distribution
        # so faculty know what is possible before they start typing
        min_m = sum(type_counts[qt] * 1 for qt in quiz_cfg["question_types"])
        max_m = sum(type_counts[qt] * 20 for qt in quiz_cfg["question_types"])
        print(f"  Target: {total_marks_target} marks")
        print(f"  Achievable range with this distribution: {min_m}–{max_m} marks\n")

        if total_marks_target < min_m:
            # Target is below the minimum — even 1 mark per question exceeds it
            print(
                f"  [ERROR] Target {total_marks_target} is below the minimum achievable "
                f"({min_m} marks with 1 mark per question)."
            )
            print(f"  Either reduce questions or increase your total marks target.")
            print(f"  Minimum valid target for this setup: {min_m} (next multiple of 5: "
                  f"{min_m if min_m % 5 == 0 else min_m + (5 - min_m % 5)})")
            sys.exit(1)

        if total_marks_target > max_m:
            # Target exceeds max even with 20 marks per question — impossible
            print(
                f"  [ERROR] Target {total_marks_target} exceeds the maximum achievable "
                f"({max_m} marks with 20 marks per question)."
            )
            print(f"  Either add more questions or reduce your total marks target.")
            sys.exit(1)

        # Build a temporary topics stub so _suggest_mark_combinations can simulate
        # distribution — we don't have real topics yet at this point in the wizard
        temp_topics = [{"num_questions": count} for count in type_counts.values()]

        suggestions = _suggest_mark_combinations(
            temp_topics, quiz_cfg["question_types"], total_marks_target
        )

        if suggestions:
            print(f"  Valid combinations that reach {total_marks_target} marks:")
            for s in suggestions:
                parts = []
                for qt, v in s.items():
                    count = type_counts[qt]
                    label = QUESTION_TYPE_LABELS.get(qt, qt)
                    parts.append(f"{label}: {v}pt × {count}q = {v * count}pts")
                print(f"    {' | '.join(parts)}")
            print()
        else:
            # No valid combinations exist — guide faculty to fix settings
            print(f"  No valid combinations found for target {total_marks_target}.")
            print(f"  Try adjusting your question count, number of types, or total marks.\n")

        # Collect marks per type with running total shown after each entry
        while True:
            per_type: Dict[str, int] = {}
            running  = 0
            print()
            for qt in quiz_cfg["question_types"]:
                label  = QUESTION_TYPE_LABELS.get(qt, qt)
                count  = type_counts[qt]
                remaining_types = [
                    t for t in quiz_cfg["question_types"] if t not in per_type and t != qt
                ]
                # Show remaining marks budget after this type is filled
                remaining_marks = total_marks_target - running
                while True:
                    raw = _ask(
                        f"  Marks for {label} ({count}q, need {remaining_marks} left for "
                        f"{len(remaining_types) + 1} type(s))",
                        str(quiz_cfg["marks_per_type"].get(qt, 1)),
                    )
                    if raw.isdigit() and int(raw) >= 1:
                        # Warn if this single type already exceeds the remaining budget
                        contribution = int(raw) * count
                        if contribution > remaining_marks and remaining_types:
                            print(
                                f"  {label} alone contributes {contribution} marks "
                                f"which exceeds the remaining budget of {remaining_marks}."
                            )
                            print(f"  Try a lower value.")
                            continue
                        per_type[qt] = int(raw)
                        running += contribution
                        print(f"    → {count} × {int(raw)} = {contribution}pts  |  running total: {running}/{total_marks_target}")
                        break
                    print("  Enter a number >= 1.")

            computed = compute_total_marks(
                # Use temp_topics for consistent distribution simulation
                temp_topics, per_type, quiz_cfg["question_types"]
            )
            if computed == total_marks_target:
                quiz_cfg["marks_per_type"] = per_type
                print(f"\n  Total marks: {computed} ✅")
                break
            diff      = total_marks_target - computed
            direction = f"{abs(diff)} marks short" if diff > 0 else f"{abs(diff)} marks over"
            print(f"\n  Total is {computed} ({direction}). Target is {total_marks_target}.")
            if suggestions:
                print(f"  Reminder — valid combinations:")
                for s in suggestions[:3]:
                    parts = ", ".join(
                        f"{QUESTION_TYPE_LABELS.get(qt, qt)}: {v}pt" for qt, v in s.items()
                    )
                    print(f"    {parts}")
            print()

    # Store target in config so it survives save/reload cycles
    quiz_cfg["total_marks_target"] = total_marks_target

    # ── Step 4: Topics and weightage ─────────────────────────────────────────
    _banner("Step 4 — Topics and weightage")
    print("Add topics and assign % weightage. Total must add up to exactly 100%.")
    print("Questions will be distributed proportionally across topics.\n")

    topics_config      = _collect_topics(cfg, available_topics, total_questions)
    quiz_cfg["topics"] = topics_config

    # Fix any rounding drift so question count matches exactly
    actual = sum(t["num_questions"] for t in quiz_cfg["topics"])
    diff   = total_questions - actual
    if diff != 0:
        largest = max(quiz_cfg["topics"], key=lambda t: t["num_questions"])
        largest["num_questions"] = max(1, largest["num_questions"] + diff)

    try:
        validate_topic_weights(quiz_cfg["topics"])
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    # ── Step 5: MCQ options ──────────────────────────────────────────────────
    if "mcq" in quiz_cfg["question_types"]:
        quiz_cfg["num_options"] = int(
            _choose("How many answer options per MCQ?", ["3", "4", "5"], str(quiz_cfg["num_options"]))
        )

    # ── Step 6: Difficulty and style ─────────────────────────────────────────
    _banner("Step 5 — Difficulty and style")
    quiz_cfg["difficulty"] = _choose(
        "Difficulty level?", ["easy", "medium", "hard"], quiz_cfg["difficulty"]
    )
    quiz_cfg["style"] = _choose(
        "Question style?", ["conceptual", "scenario", "definition"], quiz_cfg["style"]
    )

    raw = _ask("\nFilter to a specific PDF filename? (leave blank for all)", "")
    quiz_cfg["source_filter"] = raw if raw else None

    # ── Summary ──────────────────────────────────────────────────────────────
    total_marks  = compute_total_marks(
        quiz_cfg["topics"], quiz_cfg["marks_per_type"], quiz_cfg["question_types"]
    )
    actual_total = sum(t["num_questions"] for t in quiz_cfg["topics"])

    _banner("Quiz paper summary")
    print("  Topics:")
    for t in quiz_cfg["topics"]:
        print(f"    {t['topic']:<40} {t['weight']}%  {t['num_questions']} questions")
    print(f"  Types          : {', '.join(QUESTION_TYPE_LABELS.get(t, t) for t in quiz_cfg['question_types'])}")
    marks_summary = ", ".join(
        f"{QUESTION_TYPE_LABELS.get(k, k)}: {v}pt"
        for k, v in quiz_cfg["marks_per_type"].items()
        if k in quiz_cfg["question_types"]
    )
    print(f"  Marks per type : {marks_summary}")
    print(f"  Difficulty     : {quiz_cfg['difficulty']}")
    print(f"  Style          : {quiz_cfg['style']}")
    print(f"  Total questions: {actual_total}")
    print(f"  Total weight   : {sum(t['weight'] for t in quiz_cfg['topics'])}%")
    match = " ✅" if total_marks == total_marks_target else " ❌"
    print(f"  Total marks    : {total_marks} / {total_marks_target} target{match}")
    print(f"  Output bucket  : gs://{QUIZ_OUTPUT_BUCKET}")
    print(f"  Source filter  : {quiz_cfg['source_filter'] or 'all sources'}")

    if _ask("\nGenerate quiz with these settings? (yes/no)", "yes").lower() not in ("yes", "y"):
        print("Cancelled.")
        sys.exit(0)

    return quiz_cfg


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    p = argparse.ArgumentParser(description="Generate a quiz from Cloud SQL vector store.")
    p.add_argument("--interactive",         action="store_true")
    p.add_argument("--config",              metavar="FILE")
    p.add_argument("--save-config",         metavar="FILE")
    p.add_argument("--text",                type=str, default="Generate a database systems quiz.")
    p.add_argument("--num-questions",       type=int, default=None)
    p.add_argument("--questions-per-topic", type=int, default=3)
    p.add_argument("--difficulty",          type=str, default="medium",
                   choices=["easy", "medium", "hard"])
    p.add_argument("--style",               type=str, default="conceptual",
                   choices=["conceptual", "scenario", "definition"])
    p.add_argument("--num-options",         type=int, default=4, choices=[3, 4, 5])
    p.add_argument("--source-filter",       type=str, default=None)
    p.add_argument("--question-type",       type=str, default="mcq",
                   choices=["mcq", "fill_blank", "long_answer", "true_false"])
    p.add_argument("--retrieval-k",         type=int, default=6)
    p.add_argument("--max-docs",            type=int, default=12)
    p.add_argument("--tag-topics",          action="store_true")
    p.add_argument("--force-retag",         action="store_true")
    return p.parse_args()


def main() -> None:
    """Entry point for the Quiz Builder CLI."""
    app_cfg = load_config()
    args    = parse_args()

    # Tag topics mode runs KeyBERT tagging and exits — no quiz generation
    if args.tag_topics:
        asyncio.run(tag_topics_with_keybert(app_cfg, force=args.force_retag))
        return

    print("""
╔══════════════════════════════════════════════════════════╗
║           AI Teaching Assistant — Quiz Builder           ║
╚══════════════════════════════════════════════════════════╝

  MODES       --interactive  |  --config FILE  |  --save-config FILE
  QUESTIONS   --text "topic"  --num-questions 5 (total)
                              --questions-per-topic 3 (per topic, default)
              --question-type mcq|fill_blank|long_answer|true_false
  OPTIONS     --num-options 3-5  --difficulty easy|medium|hard
              --style conceptual|scenario|definition
  FILTER      --source-filter "file.pdf"
  OUTPUT      gs://quizzes-output/
""")

    if args.config:
        # Load a previously saved config JSON file directly
        with open(args.config) as f:
            quiz_cfg = json.load(f)
        logger.info("Loaded config from %s", args.config)

    elif args.interactive:
        # Fetch the topics that are covered in the knowledge base before launching the wizard
        logger.info("Fetching available topics from knowledge base...")
        try:
            available_topics = asyncio.run(fetch_available_topics(app_cfg))
        except Exception as e:
            logger.warning("Could not fetch topics: %s", e)
            available_topics = []
        quiz_cfg = build_config_interactively(app_cfg, available_topics=available_topics)
        if args.save_config:
            Path(args.save_config).write_text(json.dumps(quiz_cfg, indent=2))
            logger.info("Config saved to %s", args.save_config)
            return

    else:
        # Non-interactive CLI mode — build a minimal single-topic config from flags
        num_q = args.num_questions if args.num_questions is not None else args.questions_per_topic
        quiz_cfg = {
            "topics":         [{"topic": args.text.strip(), "weight": 100, "num_questions": num_q}],
            "question_types": [args.question_type],
            "difficulty":     args.difficulty,
            "style":          args.style,
            "num_options":    args.num_options,
            "marks_per_type": {"mcq": 1, "fill_blank": 1, "long_answer": 5, "true_false": 1},
            "source_filter":  args.source_filter,
            "retrieval_k":    args.retrieval_k,
            "max_docs":       args.max_docs,
        }

    try:
        # Weights must sum to exactly 100% and marks must be >= 1 for every active type
        validate_topic_weights(quiz_cfg["topics"])
        validate_marks_per_type(quiz_cfg.get("marks_per_type", {}), quiz_cfg["question_types"])

        # If faculty set a total marks target in the config, validate it is achievable
        # and that marks_per_type produces exactly that total given the question distribution.
        # This field is optional — omitting it skips all target checks.
        total_marks_target = quiz_cfg.get("total_marks_target")
        if total_marks_target is not None:
            validate_total_marks_target(
                target=int(total_marks_target),
                topics=quiz_cfg["topics"],
                marks_per_type=quiz_cfg.get("marks_per_type", {}),
                question_types=quiz_cfg["question_types"],
            )

    except ValueError as e:
        print(f"\n[ERROR] Invalid quiz config: {e}")
        sys.exit(1)

    total_q     = sum(t["num_questions"] for t in quiz_cfg["topics"])
    total_marks = compute_total_marks(
        quiz_cfg["topics"], quiz_cfg.get("marks_per_type", {}), quiz_cfg["question_types"]
    )

    logger.info("Topics      : %s", ", ".join(t["topic"] for t in quiz_cfg["topics"]))
    logger.info("Types       : %s (randomized)", ", ".join(quiz_cfg["question_types"]))
    logger.info("Difficulty  : %s", quiz_cfg["difficulty"])
    logger.info("Style       : %s", quiz_cfg["style"])
    logger.info("Total Q     : %d", total_q)
    logger.info("Total weight: %d%%", sum(t["weight"] for t in quiz_cfg["topics"]))
    logger.info("Total marks : %d", total_marks)
    logger.info("Output      : gs://%s/", QUIZ_OUTPUT_BUCKET)
    logger.info("Filter      : %s", quiz_cfg.get("source_filter") or "all sources")

    async def run() -> None:
        all_questions  = []
        marks_per_type = quiz_cfg.get(
            "marks_per_type",
            {"mcq": 1, "fill_blank": 1, "long_answer": 5, "true_false": 1},
        )

        logger.info("Connecting to vector store...")
        shared_vs = await get_vector_store(app_cfg)  # reuse one connection across all topics
        q_index   = 0

        for topic_cfg in quiz_cfg["topics"]:
            n     = topic_cfg["num_questions"]
            types = quiz_cfg["question_types"]
            # Tile the type list and slice to n — guarantees exact question count
            assigned_types = (types * (n // len(types) + 1))[:n]
            random.shuffle(assigned_types)  # randomise order so types don't follow a predictable pattern

            logger.info("Generating %d questions for '%s'...", n, topic_cfg["topic"])
            logger.info("Types: %s", [QUESTION_TYPE_LABELS.get(t, t) for t in assigned_types])

            for qtype in assigned_types:
                try:
                    quiz = await build_quiz(
                        cfg=app_cfg,
                        user_text=topic_cfg["topic"],
                        num_questions=1,
                        difficulty=quiz_cfg["difficulty"],
                        retrieval_k=quiz_cfg.get("retrieval_k", 6),
                        max_docs=quiz_cfg.get("max_docs", 12),
                        style=quiz_cfg["style"],
                        num_options=quiz_cfg.get("num_options", 4),
                        source_filter=quiz_cfg.get("source_filter"),
                        question_type=qtype,
                        vs=shared_vs,
                        question_index=q_index,
                    )
                    q_index += 1
                    # Attach marks and topic weight to each question before collecting
                    for q in quiz["questions"]:
                        q["marks"]        = marks_per_type.get(qtype, 1)
                        q["topic_weight"] = topic_cfg["weight"]
                    all_questions.extend(quiz["questions"])
                except RuntimeError as exc:
                    # A single topic/type combination failing should not crash the whole quiz
                    # Log and continue so the rest of the questions are still generated
                    logger.warning(
                        "Skipping %s question for '%s': %s",
                        qtype, topic_cfg["topic"], exc
                    )
                    q_index += 1

        # If every topic/type combination failed, stop cleanly instead of saving an empty quiz
        if not all_questions:
            print("\n[ERROR] No questions were generated. "
                  "All subtopics failed after retries. Try running again.")
            return

        # Final cross-quiz deduplication — catches identical questions that slipped through
        # when the same subtopic was generated for multiple build_quiz calls. Each call only
        # sees its own questions, so this is the only place we can deduplicate across all of them.
        # We deduplicate on BOTH question text and correct answer independently:
        #   - Same question text → obvious duplicate
        #   - Same correct answer → same concept asked in different words (e.g. Q1/Q3 above)
        seen_q_text:   set = set()
        seen_q_answer: set = set()
        deduped: List[Dict[str, Any]] = []
        for q in all_questions:
            q_text   = q.get("question") or q.get("statement") or ""
            q_answer = q.get("answer") or q.get("model_answer") or ""
            q_key    = re.sub(r"[^a-z0-9 ]", "", q_text.lower()).strip()[:60]
            a_key    = re.sub(r"[^a-z0-9 ]", "", q_answer.lower()).strip()[:60]

            is_dup_q = q_key and q_key in seen_q_text
            is_dup_a = a_key and a_key in seen_q_answer

            if is_dup_q:
                logger.warning("[DEDUP] Removed duplicate question text: '%s'", q_text[:80])
            elif is_dup_a:
                logger.warning("[DEDUP] Removed same-answer duplicate: answer='%s'", q_answer[:80])
            else:
                if q_key:
                    seen_q_text.add(q_key)
                if a_key:
                    seen_q_answer.add(a_key)
                deduped.append(q)

        if len(deduped) < len(all_questions):
            logger.warning(
                "%d duplicate question(s) removed from final quiz. "
                "Try increasing retrieval_k or max_docs to get more varied context.",
                len(all_questions) - len(deduped)
            )
        all_questions = deduped

        total_marks = sum(q["marks"] for q in all_questions)
        merged = {
            "quiz_id":        f"quiz_{utc_compact_ts()}",
            "config":         quiz_cfg,
            "question_types": quiz_cfg["question_types"],
            "questions":      all_questions,
            "run_metadata": {
                "generated_at_utc": utc_compact_ts(),
                "model":            app_cfg.groq_model,
                "total_generated":  len(all_questions),
                "total_marks":      total_marks,
                "total_weight_pct": sum(t["weight"] for t in quiz_cfg["topics"]),
                "types":            quiz_cfg["question_types"],
                "marks_per_type":   marks_per_type,
                "output_bucket":    QUIZ_OUTPUT_BUCKET,
            },
        }

        gcs_uri = await persist_quiz(app_cfg, merged)
        print(f"\nQuiz saved to      : {gcs_uri}")
        print(f"Total questions    : {len(all_questions)}")
        print(f"Total marks        : {total_marks}")
        print(f"Total weight       : {merged['run_metadata']['total_weight_pct']}%")
        print_budget_summary()

    asyncio.run(run())


if __name__ == "__main__":
    main()