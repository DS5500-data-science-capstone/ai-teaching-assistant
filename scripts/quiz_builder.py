#!/usr/bin/env python3
"""
quiz_builder.py

GCP quiz builder — Cloud SQL + OpenAI embeddings + Groq generation.

Usage:
    python scripts/quiz_builder.py --interactive
    python scripts/quiz_builder.py --text "database indexing" --num-questions 3
    python scripts/quiz_builder.py --config my_quiz.json
    python scripts/quiz_builder.py --tag-topics
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if creds:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds

OUTPUT_PATH    = Path(os.getenv("OUTPUT_PATH", str(PROJECT_ROOT / "output")))
TAG_CACHE_PATH = "knowledge_base/tags/chunk_tags.json"

DEFAULT_CONFIG = {
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

QUESTION_TYPE_LABELS = {
    "mcq":         "Multiple choice (MCQ)",
    "fill_blank":  "Fill in the blank",
    "long_answer": "Long answer / essay",
    "true_false":  "True or false",
}

SYLLABUS_TOPICS = [
    "Relational Model & Algebra",
    "Modern SQL",
    "Database Storage I",
    "Memory Management",
    "Database Storage II",
    "Storage Models & Compression",
    "Hash Tables",
    "Indexes & Filters I",
    "Indexes & Filters II",
    "Index Concurrency Control",
    "Sorting & Aggregations Algorithms",
    "Joins Algorithms",
    "Query Execution I",
    "Query Execution II",
    "Query Planning & Optimization",
    "Concurrency Control Theory",
    "Two-Phase Locking Concurrency Control",
    "Timestamp Ordering Concurrency Control",
    "Multi-Version Concurrency Control",
    "Database Logging",
    "Database Recovery",
    "Introduction to Distributed Databases",
    "Distributed OLTP Database Systems",
    "Distributed OLAP Database Systems",
]

COST_PER_INPUT_TOKEN  = 0.05 / 1_000_000
COST_PER_OUTPUT_TOKEN = 0.08 / 1_000_000
BUDGET_LIMIT_USD      = 10.0

GROQ_MODEL_LIMITS = {
    "llama-3.3-70b-versatile": {"req_per_min": 500, "tokens_per_min": 100_000},
    "llama-3.1-8b-instant":    {"req_per_min": 500, "tokens_per_min": 100_000},
}

_groq_cache: Dict[str, Any] = {}
_budget = {
    "requests_this_minute": 0,
    "tokens_this_minute":   0,
    "minute_start":         0.0,
    "total_requests":       0,
    "total_tokens":         0,
    "total_cost_usd":       0.0,
}


def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def _reset_minute_budget() -> None:
    import time
    now = time.time()
    if now - _budget["minute_start"] >= 60:
        _budget["requests_this_minute"] = 0
        _budget["tokens_this_minute"]   = 0
        _budget["minute_start"]         = now


def _record_usage(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    _reset_minute_budget()
    total = prompt_tokens + completion_tokens
    cost  = (prompt_tokens * COST_PER_INPUT_TOKEN) + (completion_tokens * COST_PER_OUTPUT_TOKEN)
    _budget["requests_this_minute"] += 1
    _budget["tokens_this_minute"]   += total
    _budget["total_requests"]       += 1
    _budget["total_tokens"]         += total
    _budget["total_cost_usd"]       += cost

    limits  = GROQ_MODEL_LIMITS.get(model, {"req_per_min": 30, "tokens_per_min": 6_000})
    req_pct = (_budget["requests_this_minute"] / limits["req_per_min"])    * 100
    tok_pct = (_budget["tokens_this_minute"]   / limits["tokens_per_min"]) * 100

    print(f"[BUDGET] ${_budget['total_cost_usd']:.4f} / ${BUDGET_LIMIT_USD:.2f} | "
          f"{_budget['total_requests']} req | {_budget['total_tokens']} tokens")

    if req_pct >= 80 or tok_pct >= 80:
        print(f"[BUDGET] Warning: {req_pct:.0f}% req/min, {tok_pct:.0f}% token/min used")

    if _budget["total_cost_usd"] >= BUDGET_LIMIT_USD:
        raise RuntimeError(f"[BUDGET] Hard stop: reached ${BUDGET_LIMIT_USD:.2f} spending limit.")


def print_budget_summary() -> None:
    print(f"\n[BUDGET] Session total: {_budget['total_requests']} requests, "
          f"{_budget['total_tokens']} tokens, ${_budget['total_cost_usd']:.4f} spent")


@dataclass(frozen=True)
class AppConfig:
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
        groq_base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions"),
        openai_api_key=must("OPENAI_API_KEY"),
        table_name=os.getenv("VECTOR_TABLE_NAME", "course_embeddings"),
        vector_size=int(os.getenv("VECTOR_SIZE", "1536")),
    )


class TopicItem(BaseModel):
    topic: str = Field(description="A short and precise subtopic.")

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Topic too short.")
        return v


class TopicList(BaseModel):
    topics: List[TopicItem]

    @field_validator("topics")
    @classmethod
    def validate_topics(cls, v):
        if not v:
            raise ValueError("At least one topic required.")
        return v


class QuestionItem(BaseModel):
    question: str
    sources: List[str]

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 10:  raise ValueError("Question too short.")
        if len(v) > 500: raise ValueError("Question too long.")
        return v

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v):
        if not v: raise ValueError("Must include at least one source.")
        return v[:3]


class AnswerItem(BaseModel):
    answer: str
    explanation: str
    sources: List[str]

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, v: str) -> str:
        v = v.strip()
        if not v: raise ValueError("Answer missing.")
        return v

    @field_validator("explanation")
    @classmethod
    def validate_explanation(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 5: raise ValueError("Explanation too short.")
        return v

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v):
        if not v: raise ValueError("Must include at least one source.")
        return v[:3]


class DistractorItem(BaseModel):
    incorrect_answer: str
    explanation: str
    sources: List[str]

    @field_validator("incorrect_answer")
    @classmethod
    def validate_incorrect_answer(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2: raise ValueError("Incorrect answer too short.")
        return v

    @field_validator("explanation")
    @classmethod
    def validate_explanation(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 5: raise ValueError("Explanation too short.")
        return v

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v):
        if not v: raise ValueError("Must include at least one source.")
        return v[:3]


def utc_compact_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    return text.replace("\x00", " ").strip()


def safe_json_extract(text: str) -> Any:
    obj = re.search(r"(\{\s*[\s\S]*\s*\})", text)
    if obj:
        return json.loads(obj.group(1))
    arr = re.search(r"(\[\s*[\s\S]*\s*\])", text)
    if arr:
        return json.loads(arr.group(1))
    raise ValueError("No JSON found in model output.")


def dedupe_documents(documents: Sequence[Document], max_docs: int) -> List[Document]:
    seen, results = set(), []
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
    blocks = []
    for i, doc in enumerate(documents, 1):
        m        = doc.metadata or {}
        src      = m.get("source", "")
        page     = m.get("page", m.get("page_number", ""))
        chunk_id = m.get("chunk_id", "")
        blocks.append(f"[CHUNK {i}] chunk_id={chunk_id} source={src} page={page}\n{doc.page_content}")
    return "\n\n".join(blocks)


async def get_vector_store(cfg: AppConfig) -> PostgresVectorStore:
    engine = await PostgresEngine.afrom_instance(
        project_id=cfg.project_id,
        region=cfg.region,
        instance=cfg.instance,
        database=cfg.database,
        user=cfg.db_user,
        password=cfg.db_pass,
    )
    os.environ["OPENAI_API_KEY"] = cfg.openai_api_key
    return await PostgresVectorStore.create(
        engine,
        embedding_service=OpenAIEmbeddings(),
        table_name=cfg.table_name,
    )


async def retrieve_context(cfg: AppConfig, query_text: str, retrieval_k: int,
                            max_docs: int, source_filter: str = None) -> List[Document]:
    vs = await get_vector_store(cfg)
    queries = [query_text, f"{query_text} database systems", f"{query_text} CMU DBMS"]
    all_docs = []
    for q in queries:
        docs = await vs.asimilarity_search(q, k=retrieval_k)
        if source_filter:
            docs = [d for d in docs if source_filter.lower() in
                    str(d.metadata.get("source", "")).lower()]
        all_docs.extend(docs)
    return dedupe_documents(all_docs, max_docs=max_docs)


def load_tag_cache(cfg: AppConfig) -> Dict[str, Dict]:
    try:
        client = storage.Client(project=cfg.project_id)
        blob   = client.bucket(cfg.gcs_bucket).blob(TAG_CACHE_PATH)
        if not blob.exists():
            return {}
        print(f"[INFO] Loaded tag cache from gs://{cfg.gcs_bucket}/{TAG_CACHE_PATH}")
        return json.loads(blob.download_as_text())
    except Exception as e:
        print(f"[WARN] Could not load tag cache: {e}")
        return {}


def save_tag_cache(cfg: AppConfig, cache: Dict[str, Dict]) -> None:
    try:
        client = storage.Client(project=cfg.project_id)
        blob   = client.bucket(cfg.gcs_bucket).blob(TAG_CACHE_PATH)
        blob.upload_from_string(
            json.dumps(cache, indent=2, ensure_ascii=False),
            content_type="application/json",
        )
        print(f"[INFO] Tag cache saved to gs://{cfg.gcs_bucket}/{TAG_CACHE_PATH}")
    except Exception as e:
        print(f"[WARN] Could not save tag cache: {e}")


async def tag_topics_with_keybert(cfg: AppConfig, force: bool = False) -> None:
    try:
        from keybert import KeyBERT
        from sklearn.metrics.pairwise import cosine_similarity
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError("Run: pip install keybert sentence-transformers scikit-learn")

    cache = {} if force else load_tag_cache(cfg)
    if cache and not force:
        print(f"[INFO] Tag cache has {len(cache)} entries. Only tagging new chunks.")

    print("[INFO] Loading sentence transformer model...")
    model    = SentenceTransformer("all-MiniLM-L6-v2")
    kw_model = KeyBERT(model=model)

    print("[INFO] Encoding syllabus topics...")
    topic_embeddings = model.encode(SYLLABUS_TOPICS)

    vs = await get_vector_store(cfg)
    print("[INFO] Fetching all chunks from Cloud SQL...")
    results = await vs.asimilarity_search("database", k=500)
    print(f"[INFO] Found {len(results)} chunks.")

    new_tags = 0
    for doc in results:
        key = hashlib.md5(doc.page_content[:200].encode()).hexdigest()
        if key in cache and not force:
            continue

        if new_tags % 100 == 0 and new_tags > 0:
            print(f"[INFO]   Tagged {new_tags} new chunks...")

        try:
            kws     = kw_model.extract_keywords(doc.page_content, keyphrase_ngram_range=(1, 2),
                                                 stop_words="english", top_n=5)
            kw_text = " ".join(kw for kw, _ in kws) if kws else doc.page_content[:200]
            sims    = cosine_similarity(model.encode([kw_text]), topic_embeddings)[0]
            best    = int(sims.argmax())
            topic   = SYLLABUS_TOPICS[best]
            conf    = float(sims[best])
        except Exception:
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
        print("[INFO] All chunks already tagged. Use --force-retag to re-run.")
        return

    save_tag_cache(cfg, cache)
    print(f"\n[INFO] Done — {new_tags} chunks tagged.")
    print("\n[INFO] Topic distribution:")
    for topic, count in sorted(Counter(v["topic"] for v in cache.values()).items(), key=lambda x: -x[1]):
        print(f"  {topic:<50} {count} chunks")


async def fetch_available_topics(cfg: AppConfig) -> List[str]:
    try:
        cache = load_tag_cache(cfg)
        if cache:
            covered = {v["topic"] for v in cache.values()}
            return [t for t in SYLLABUS_TOPICS if t in covered]
    except Exception as e:
        print(f"[WARN] Could not load tag cache: {e}")
    return list(SYLLABUS_TOPICS)


def build_mcq_combined_prompt(subtopic: str, documents: List[Document],
                               difficulty: str = "medium", style: str = "conceptual",
                               num_options: int = 4) -> str:
    ctx = format_context_blocks(documents)
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
    n = num_options - 1
    return f"""
You are generating a complete multiple choice question for a Database Systems course.
Subtopic: {subtopic}
Style: {styles[style]}
Difficulty: {diffs[difficulty]}
Generate the question, correct answer, and exactly {n} incorrect answers in ONE response.
Return ONLY a single JSON object with keys: "question", "answer", "explanation", "incorrect_answers", "sources"
"incorrect_answers" must be a list of {n} objects each with keys: "incorrect_answer", "explanation", "sources"
Example:
{{
  "question": "What is the primary advantage of a B+ tree index?",
  "answer": "It allows efficient range queries by maintaining sorted keys.",
  "explanation": "B+ trees keep all data in leaf nodes linked together.",
  "sources": ["CHUNK 1"],
  "incorrect_answers": [{{"incorrect_answer": "It reduces storage by compressing duplicate keys.", "explanation": "B+ trees do not compress keys.", "sources": ["CHUNK 1"]}}]
}}
Context:
{ctx}
""".strip()


def build_topics_prompt(user_text: str, documents: List[Document], n: int) -> str:
    ctx = format_context_blocks(documents)
    return f"""
You are generating quiz subtopics for a Database Systems course.
User request: {user_text}
Using only the context below, produce exactly {n} distinct subtopics.
Output JSON only: {{"topics": [{{"topic": "string"}}]}}
Context:
{ctx}
""".strip()


def build_fill_blank_prompt(subtopic: str, documents: List[Document],
                             difficulty: str = "medium") -> str:
    ctx = format_context_blocks(documents)
    diffs = {
        "easy":   "The blank should be a single well-known keyword.",
        "medium": "The blank should be a key term that requires understanding.",
        "hard":   "The blank should require deep knowledge to fill correctly.",
    }
    return f"""
You are generating a fill-in-the-blank question for a Database Systems course.
Subtopic: {subtopic}
Difficulty: {diffs[difficulty]}
Return ONLY a single JSON object with keys "question", "answer", "explanation", "sources".
The question must contain exactly one blank as _____.
Example: {{"question": "A _____ index contains an entry for every search-key value.", "answer": "dense", "explanation": "A dense index has one entry per search-key value.", "sources": ["CHUNK 1"]}}
Context:
{ctx}
""".strip()


def build_long_answer_prompt(subtopic: str, documents: List[Document],
                              difficulty: str = "medium", style: str = "conceptual") -> str:
    ctx = format_context_blocks(documents)
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
    return f"""
You are generating a long-answer question for a Database Systems course.
Subtopic: {subtopic}
Style: {styles[style]}
Difficulty: {diffs[difficulty]}
Return ONLY a single JSON object with keys "question", "model_answer", "key_points", "sources".
Example: {{"question": "Explain clustered vs non-clustered indexes.", "model_answer": "A clustered index determines physical order...", "key_points": ["Physical vs logical ordering"], "sources": ["CHUNK 1"]}}
Context:
{ctx}
""".strip()


def build_true_false_prompt(subtopic: str, documents: List[Document],
                             difficulty: str = "medium") -> str:
    ctx = format_context_blocks(documents)
    diffs = {
        "easy":   "The statement should be clearly true or false.",
        "medium": "The statement should be plausible and require knowledge to judge.",
        "hard":   "The statement should be subtle — a common misconception or edge case.",
    }
    return f"""
You are generating a true/false question for a Database Systems course.
Subtopic: {subtopic}
Difficulty: {diffs[difficulty]}
Return ONLY a single JSON object with keys "statement", "answer", "explanation", "sources".
"answer" must be exactly "True" or "False".
Example: {{"statement": "A sparse index always has fewer entries than a dense index.", "answer": "True", "explanation": "Sparse indexes store entries for only some search-key values.", "sources": ["CHUNK 2"]}}
Context:
{ctx}
""".strip()


async def call_groq_json(cfg: AppConfig, prompt: str, max_retries: int = 15) -> Any:
    key = _cache_key(prompt)
    if key in _groq_cache:
        print("[BUDGET] Cache hit — skipping API call")
        return _groq_cache[key]

    _reset_minute_budget()
    headers = {
        "Authorization": f"Bearer {cfg.groq_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg.groq_model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "You are a precise quiz-generation assistant. Return valid JSON only."},
            {"role": "user",   "content": prompt},
        ],
    }

    for attempt in range(max_retries):
        async with aiohttp.ClientSession() as session:
            async with session.post(cfg.groq_base_url, headers=headers, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=180)) as resp:
                if resp.status == 429:
                    wait = min(2 ** attempt, 60)
                    print(f"[BUDGET] Rate limit — waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                body = await resp.json()

        usage = body.get("usage", {})
        _record_usage(cfg.groq_model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        result = safe_json_extract(body["choices"][0]["message"]["content"].strip())
        _groq_cache[key] = result
        return result

    raise RuntimeError("Groq rate limit exceeded after max retries.")


async def generate_topics(cfg, text, docs, n) -> List[str]:
    payload = await call_groq_json(cfg, build_topics_prompt(text, docs, n))
    if isinstance(payload, list):
        if all(isinstance(t, str) for t in payload):
            return [t.strip() for t in payload[:n]]
        payload = {"topics": payload}
    if isinstance(payload, dict) and "topics" in payload:
        result = []
        for t in payload["topics"][:n]:
            if isinstance(t, str):
                result.append(t.strip())
            elif isinstance(t, dict):
                result.append(t.get("topic", "").strip())
        return [r for r in result if r]
    return []


async def build_quiz(cfg: AppConfig, user_text: str, num_questions: int, difficulty: str,
                     retrieval_k: int, max_docs: int, style: str = "conceptual",
                     num_options: int = 4, source_filter: str = None,
                     question_type: str = "mcq") -> Dict[str, Any]:

    docs = await retrieve_context(cfg, user_text, retrieval_k, max_docs, source_filter)
    if not docs:
        raise RuntimeError("No documents retrieved from Cloud SQL. Run create_database.py first.")

    topics = await generate_topics(cfg, user_text, docs, num_questions)
    quiz_questions: List[Dict[str, Any]] = []

    for idx, topic in enumerate(topics, 1):
        try:
            entry: Dict[str, Any] = {"id": f"q_{idx}", "type": question_type, "topic": topic}

            if question_type == "mcq":
                raw = await call_groq_json(cfg, build_mcq_combined_prompt(topic, docs, difficulty, style, num_options))
                if isinstance(raw, list):
                    raise ValueError("Unexpected list from Groq")
                opts = [raw.get("answer", "")] + [d.get("incorrect_answer", "") for d in raw.get("incorrect_answers", [])]
                random.shuffle(opts)
                entry.update({
                    "question":    raw.get("question", ""),
                    "options":     opts,
                    "answer":      raw.get("answer", ""),
                    "explanation": raw.get("explanation", ""),
                    "sources":     sorted(set(raw.get("sources", []))),
                    "incorrect_answers": [{"answer": d.get("incorrect_answer", ""),
                                           "explanation": d.get("explanation", ""),
                                           "sources": d.get("sources", [])}
                                          for d in raw.get("incorrect_answers", [])],
                })

            elif question_type == "fill_blank":
                r = await call_groq_json(cfg, build_fill_blank_prompt(topic, docs, difficulty))
                if isinstance(r, list): raise ValueError("Unexpected list from Groq")
                entry.update({"question": r.get("question", ""), "answer": r.get("answer", ""),
                               "explanation": r.get("explanation", ""), "sources": r.get("sources", [])})

            elif question_type == "long_answer":
                r = await call_groq_json(cfg, build_long_answer_prompt(topic, docs, difficulty, style))
                if isinstance(r, list): raise ValueError("Unexpected list from Groq")
                entry.update({"question": r.get("question", ""), "model_answer": r.get("model_answer", ""),
                               "key_points": r.get("key_points", []), "sources": r.get("sources", [])})

            elif question_type == "true_false":
                r = await call_groq_json(cfg, build_true_false_prompt(topic, docs, difficulty))
                if isinstance(r, list): raise ValueError("Unexpected list from Groq")
                entry.update({"statement": r.get("statement", ""), "answer": r.get("answer", ""),
                               "explanation": r.get("explanation", ""), "sources": r.get("sources", [])})

            quiz_questions.append(entry)
            print(f"[OK] {question_type} {idx}/{len(topics)}: {topic}")

        except ValidationError as exc:
            print(f"[WARN] Validation failure for '{topic}': {exc}")
        except Exception as exc:
            print(f"[WARN] Generation failure for '{topic}': {exc}")

    if not quiz_questions:
        raise RuntimeError("Quiz generation failed for all topics.")

    return {
        "quiz_id":       f"quiz_{utc_compact_ts()}",
        "difficulty":    difficulty,
        "style":         style,
        "question_type": question_type,
        "questions":     quiz_questions,
        "run_metadata": {
            "generated_at_utc":    utc_compact_ts(),
            "model":               cfg.groq_model,
            "vector_table":        cfg.table_name,
            "retrieval_k":         retrieval_k,
            "max_docs":            max_docs,
            "requested_questions": num_questions,
            "generated_questions": len(quiz_questions),
            "style":               style,
            "difficulty":          difficulty,
            "num_options":         num_options,
            "source_filter":       source_filter or "all",
            "retrieved_chunks": [
                {"chunk_id": (d.metadata or {}).get("chunk_id"),
                 "source":   (d.metadata or {}).get("source"),
                 "page":     (d.metadata or {}).get("page", (d.metadata or {}).get("page_number"))}
                for d in docs
            ],
        },
    }

def write_json_to_gcs(cfg: AppConfig, object_path: str, payload: Any) -> str:
    client = storage.Client(project=cfg.project_id)
    blob   = client.bucket(cfg.gcs_bucket).blob(object_path)
    blob.upload_from_string(json.dumps(payload, indent=2, ensure_ascii=False),
                            content_type="application/json")
    return f"gs://{cfg.gcs_bucket}/{object_path}"


async def persist_quiz(cfg: AppConfig, quiz: Dict[str, Any]) -> str:
    quiz_id = quiz.get("quiz_id", f"quiz_{utc_compact_ts()}")
    return write_json_to_gcs(cfg, f"{cfg.gcs_out_prefix.rstrip('/')}/{quiz_id}.json", quiz)


def _banner(text: str) -> None:
    print(f"\n{'─' * 50}\n  {text}\n{'─' * 50}")


def _ask(prompt: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    raw  = input(f"{prompt}{hint}: ").strip()
    return raw if raw else default


def _choose(prompt: str, options: List[str], default: str = "") -> str:
    print(f"\n{prompt}")
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}{' (default)' if o == default else ''}")
    while True:
        raw = input("Enter number or value: ").strip()
        if not raw and default: return default
        if raw.isdigit() and 1 <= int(raw) <= len(options): return options[int(raw)-1]
        if raw in options: return raw
        print(f"  Enter a number 1-{len(options)}")


def _choose_many(prompt: str, options: List[str], defaults: List[str]) -> List[str]:
    print(f"\n{prompt}")
    for i, o in enumerate(options, 1):
        print(f"  {i}. {QUESTION_TYPE_LABELS.get(o, o)}{' (selected)' if o in defaults else ''}")
    print(f"  Enter numbers separated by commas. Press Enter for defaults: {', '.join(defaults)}")
    while True:
        raw = input("Selection: ").strip()
        if not raw: return defaults
        parts = [p.strip() for p in raw.split(",")]
        selected, valid = [], True
        for p in parts:
            if p.isdigit() and 1 <= int(p) <= len(options):
                selected.append(options[int(p)-1])
            elif p in options:
                selected.append(p)
            else:
                print(f"  Invalid: '{p}'. Try again.")
                valid = False
                break
        if valid and selected: return selected


def build_config_interactively(available_topics: List[str] = None) -> Dict:
    _banner("Quiz configurator — faculty edition")
    print("Configure your quiz paper below.")
    print("Press Enter to accept the default shown in [brackets].\n")

    cfg = dict(DEFAULT_CONFIG)
    cfg["marks_per_type"] = dict(DEFAULT_CONFIG["marks_per_type"])

    if available_topics:
        print("Available topics in the knowledge base:")
        for i, t in enumerate(available_topics, 1):
            print(f"  {i}. {t}")
        print()
    else:
        print("(No topic list available — type topics manually)\n")

    _banner("Step 1 — Topics and weightage")
    print("Add topics and assign % weightage. Total must add up to 100%.\n")

    topics_config, total_weight = [], 0
    while total_weight < 100:
        remaining = 100 - total_weight
        print(f"  Weight remaining: {remaining}%")

        if available_topics:
            raw = input("  Pick topic number or type a custom topic (Enter to finish): ").strip()
            if not raw and topics_config:
                break
            topic = available_topics[int(raw)-1] if raw.isdigit() and 1 <= int(raw) <= len(available_topics) else (raw or "database systems")
        else:
            topic = _ask("  Topic name", "database systems")

        while True:
            w = input(f"  Weight for '{topic}' (remaining: {remaining}%): ").strip()
            if w.isdigit() and 1 <= int(w) <= remaining:
                weight = int(w); break
            print(f"  Enter a number between 1 and {remaining}.")

        while True:
            n = input(f"  Number of questions for '{topic}': ").strip()
            if n.isdigit() and 1 <= int(n) <= 20:
                num_q = int(n); break
            print("  Enter a number between 1 and 20.")

        topics_config.append({"topic": topic, "weight": weight, "num_questions": num_q})
        total_weight += weight
        print(f"  Added: {topic} ({weight}%, {num_q} questions)\n")
        if total_weight == 100:
            break

    cfg["topics"] = topics_config

    _banner("Step 2 — Question types")
    cfg["question_types"] = _choose_many("Which question types do you want?",
                                          list(QUESTION_TYPE_LABELS.keys()), cfg["question_types"])

    _banner("Step 3 — Marks per question")
    print("Set how many marks each question type is worth.\n")
    for qtype in cfg["question_types"]:
        label = QUESTION_TYPE_LABELS.get(qtype, qtype)
        while True:
            raw = _ask(f"  Marks for {label}", str(cfg["marks_per_type"].get(qtype, 1)))
            if raw.isdigit() and int(raw) >= 1:
                cfg["marks_per_type"][qtype] = int(raw); break
            print("  Enter a number >= 1.")

    if "mcq" in cfg["question_types"]:
        cfg["num_options"] = int(_choose("How many answer options per MCQ?",
                                          ["3", "4", "5"], str(cfg["num_options"])))

    _banner("Step 4 — Difficulty and style")
    cfg["difficulty"] = _choose("Difficulty level?", ["easy", "medium", "hard"], cfg["difficulty"])
    cfg["style"]      = _choose("Question style?", ["conceptual", "scenario", "definition"], cfg["style"])

    raw = _ask("\nFilter to a specific PDF filename? (leave blank for all)", "")
    cfg["source_filter"] = raw if raw else None

    _banner("Quiz paper summary")
    total_q     = sum(t["num_questions"] for t in cfg["topics"])
    total_marks = sum(
        t["num_questions"] * (
            sum(cfg["marks_per_type"].get(qt, 1) for qt in cfg["question_types"]) // len(cfg["question_types"])
        )
        for t in cfg["topics"]
    )
    print("  Topics:")
    for t in cfg["topics"]:
        print(f"    {t['topic']:<40} {t['weight']}%  {t['num_questions']} questions")
    print(f"  Types          : {', '.join(QUESTION_TYPE_LABELS.get(t, t) for t in cfg['question_types'])}")
    marks_summary = ", ".join(f"{QUESTION_TYPE_LABELS.get(k, k)}: {v}"
                               for k, v in cfg["marks_per_type"].items() if k in cfg["question_types"])
    print(f"  Marks per type : {marks_summary}")
    print(f"  Difficulty     : {cfg['difficulty']}")
    print(f"  Style          : {cfg['style']}")
    print(f"  Total questions: {total_q}")
    print(f"  Total marks    : ~{total_marks} (varies by random type assignment)")
    print(f"  Source filter  : {cfg['source_filter'] or 'all sources'}")

    if _ask("\nGenerate quiz with these settings? (yes/no)", "yes").lower() not in ("yes", "y"):
        print("Cancelled.")
        sys.exit(0)

    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a quiz from Cloud SQL vector store.")
    p.add_argument("--interactive",   action="store_true")
    p.add_argument("--config",        metavar="FILE")
    p.add_argument("--save-config",   metavar="FILE")
    p.add_argument("--text",          type=str, default="Generate a database systems quiz.")
    p.add_argument("--num-questions", type=int, default=3)
    p.add_argument("--difficulty",    type=str, default="medium", choices=["easy", "medium", "hard"])
    p.add_argument("--style",         type=str, default="conceptual", choices=["conceptual", "scenario", "definition"])
    p.add_argument("--num-options",   type=int, default=4, choices=[3, 4, 5])
    p.add_argument("--source-filter", type=str, default=None)
    p.add_argument("--question-type", type=str, default="mcq",
                   choices=["mcq", "fill_blank", "long_answer", "true_false"])
    p.add_argument("--retrieval-k",   type=int, default=6)
    p.add_argument("--max-docs",      type=int, default=12)
    p.add_argument("--tag-topics",    action="store_true", help="Tag chunks with syllabus topics (run once).")
    p.add_argument("--force-retag",   action="store_true", help="Force re-tagging all chunks.")
    return p.parse_args()


def main() -> None:
    app_cfg = load_config()
    args    = parse_args()

    if args.tag_topics:
        asyncio.run(tag_topics_with_keybert(app_cfg, force=args.force_retag))
        return

    print("""
╔══════════════════════════════════════════════════════════╗
║           AI Teaching Assistant — Quiz Builder           ║
╚══════════════════════════════════════════════════════════╝

  MODES       --interactive  |  --config FILE  |  --save-config FILE
  QUESTIONS   --text "topic"  --num-questions 1-20  --question-type mcq|fill_blank|long_answer|true_false
  OPTIONS     --num-options 3-5  --difficulty easy|medium|hard  --style conceptual|scenario|definition
  FILTER      --source-filter "file.pdf"
  TAGGING     --tag-topics  |  --force-retag
""")

    if args.config:
        with open(args.config) as f:
            quiz_cfg = json.load(f)
        print(f"[INFO] Loaded config from {args.config}")

    elif args.interactive:
        print("[INFO] Fetching available topics from knowledge base...")
        try:
            available_topics = asyncio.run(fetch_available_topics(app_cfg))
        except Exception as e:
            print(f"[WARN] Could not fetch topics: {e}")
            available_topics = []

        if available_topics:
            print("\nAvailable topics in your knowledge base:")
            for i, t in enumerate(available_topics, 1):
                print(f"  {i}. {t}")
            print()
        else:
            print("[INFO] Could not load topics — type them manually.\n")

        quiz_cfg = build_config_interactively(available_topics=available_topics)

        if args.save_config:
            Path(args.save_config).write_text(json.dumps(quiz_cfg, indent=2))
            print(f"[INFO] Config saved to {args.save_config}")
            return

    else:
        quiz_cfg = {
            "topics":         [{"topic": args.text.strip(), "weight": 100, "num_questions": args.num_questions}],
            "question_types": [args.question_type],
            "difficulty":     args.difficulty,
            "style":          args.style,
            "num_options":    args.num_options,
            "marks_per_type": {"mcq": 1, "fill_blank": 1, "long_answer": 5, "true_false": 1},
            "source_filter":  args.source_filter,
            "retrieval_k":    args.retrieval_k,
            "max_docs":       args.max_docs,
        }

    total_q = sum(t["num_questions"] for t in quiz_cfg["topics"])
    print(f"\n[CONFIG] Topics    : {', '.join(t['topic'] for t in quiz_cfg['topics'])}")
    print(f"[CONFIG] Types     : {', '.join(quiz_cfg['question_types'])} (randomized)")
    print(f"[CONFIG] Difficulty: {quiz_cfg['difficulty']}")
    print(f"[CONFIG] Style     : {quiz_cfg['style']}")
    print(f"[CONFIG] Total Q   : {total_q}")
    print(f"[CONFIG] Filter    : {quiz_cfg.get('source_filter') or 'all sources'}\n")

    async def run() -> None:
        all_questions  = []
        marks_per_type = quiz_cfg.get("marks_per_type", {"mcq": 1, "fill_blank": 1, "long_answer": 5, "true_false": 1})

        for topic_cfg in quiz_cfg["topics"]:
            n     = topic_cfg["num_questions"]
            types = quiz_cfg["question_types"]

            # distribute types evenly across n questions then shuffle for randomness
            assigned_types = (types * (n // len(types) + 1))[:n]
            random.shuffle(assigned_types)

            print(f"\n[INFO] Generating {n} questions for '{topic_cfg['topic']}'...")
            print(f"[INFO] Type order: {[QUESTION_TYPE_LABELS.get(t, t) for t in assigned_types]}")

            for qtype in assigned_types:
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
                )
                for q in quiz["questions"]:
                    q["marks"]        = marks_per_type.get(qtype, 1)
                    q["topic_weight"] = topic_cfg["weight"]
                all_questions.extend(quiz["questions"])

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
                "types":            quiz_cfg["question_types"],
                "marks_per_type":   marks_per_type,
            },
        }

        gcs_uri    = await persist_quiz(app_cfg, merged)
        print(f"Quiz saved to GCS  : {gcs_uri}")
        print(f"Total questions    : {len(all_questions)}")
        print(f"Total marks        : {total_marks}")
        print_budget_summary()

    asyncio.run(run())


if __name__ == "__main__":
    main()
