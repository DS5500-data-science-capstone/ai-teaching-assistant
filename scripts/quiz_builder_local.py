#!/usr/bin/env python3
"""
quiz_builder_local.py

Local quiz builder with interactive configurator built in.
- Loads PDFs from the local data/ folder
- Uses Chroma as the local vector store
- Uses HuggingFace MiniLM for embeddings (free, runs locally)
- Uses Groq for quiz generation
- Supports MCQ, fill-in-the-blank, long answer, and true/false
- Saves the quiz as a local JSON file

Usage:
    # Interactive mode (recommended)
    python scripts/quiz_builder_local.py --interactive

    # Save config without running
    python scripts/quiz_builder_local.py --interactive --save-config my_quiz.json

    # Run from a saved config
    python scripts/quiz_builder_local.py --config my_quiz.json

    # Direct CLI (original style)
    python scripts/quiz_builder_local.py --text "database indexing" --num-questions 3

    # List available topics
    python scripts/quiz_builder_local.py --list-topics

    # Force re-embed all PDFs
    python scripts/quiz_builder_local.py --interactive --reindex
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import hashlib
import aiohttp
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from keybert import KeyBERT
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, ValidationError, field_validator

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH   = Path(os.getenv("DATA_PATH",   str(PROJECT_ROOT / "data/knowledge_base")))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "chroma_db")))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", str(PROJECT_ROOT / "output")))

# ---------------------------------------------------------------------------
# Config defaults (used by both interactive mode and --config file)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "topic":          "database indexing",
    "num_questions":  3,
    "question_types": ["mcq"],
    "difficulty":     "medium",
    "style":          "conceptual",
    "num_options":    4,
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

# ---------------------------------------------------------------------------
# Groq config
# ---------------------------------------------------------------------------
def load_config() -> Dict[str, str]:
    def must(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise RuntimeError(f"Missing required env var: {name}")
        return value
    return {
        "groq_api_key": must("GROQ_API_KEY"),
        "groq_model":   os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "groq_base_url": os.getenv("GROQ_BASE_URL",
                                   "https://api.groq.com/openai/v1/chat/completions"),
    }


# ---------------------------------------------------------------------------
# Pydantic validation models
# ---------------------------------------------------------------------------

class TopicItem(BaseModel):
    topic: str = Field(description="A short and precise subtopic.")

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:   raise ValueError("Topic too short.")
        if len(v.split()) > 6: raise ValueError("Topic too long.")
        return v


class TopicList(BaseModel):
    topics: List[TopicItem]

    @field_validator("topics")
    @classmethod
    def validate_topics(cls, v):
        if not v: raise ValueError("At least one topic required.")
        return v


class QuestionItem(BaseModel):
    question: str
    sources: List[str]

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 10:  raise ValueError("Question too short.")
        if len(v) > 300: raise ValueError("Question too long.")
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


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def utc_compact_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def clean_text(text: str) -> str:
    return text.replace("\x00", "").strip()

def safe_json_extract(text: str) -> Any:
    """Extract first valid JSON object (tried before array) from LLM output."""
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
        if key in seen: continue
        seen.add(key)
        results.append(doc)
        if len(results) >= max_docs: break
    return results

def format_context_blocks(documents: Sequence[Document]) -> str:
    blocks = []
    for i, doc in enumerate(documents, 1):
        m = doc.metadata or {}
        src      = Path(m.get("source", "")).name
        page     = m.get("page", "")
        topic    = m.get("topic", "")
        keywords = m.get("keywords", "")
        blocks.append(
            f"[CHUNK {i}] source={src} page={page} topic={topic} keywords={keywords}\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def load_and_index_documents() -> Chroma:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    if CHROMA_PATH.exists():
        print(f"[INFO] Loading existing Chroma DB from {CHROMA_PATH}")
        return Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embedding_model)

    print(f"[INFO] Loading PDFs from {DATA_PATH}...")
    if not DATA_PATH.exists():
        raise RuntimeError(f"Data folder not found: {DATA_PATH}")

    loader = PyPDFDirectoryLoader(str(DATA_PATH))
    documents = loader.load()
    if not documents:
        raise RuntimeError(f"No PDFs found in {DATA_PATH}.")

    print(f"[INFO] Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=500,
        length_function=len, add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)
    chunks = [c for c in chunks if c.page_content]

    # Tag each chunk with topic from PDF filename + page number
    # then enrich with KeyBERT keyword tags (fully local, no API needed)
    print("[INFO] Tagging chunks with KeyBERT keywords (runs once during reindex)...")
    kw_model = KeyBERT(model="sentence-transformers/all-MiniLM-L6-v2")
    tagged = []
    for i, chunk in enumerate(chunks):
        if i % 500 == 0:
            print(f"[INFO]   Tagging chunk {i}/{len(chunks)}...")
        src   = Path(chunk.metadata.get("source", "")).stem
        page  = chunk.metadata.get("page", 0)
        topic = src.replace("-", " ").replace("_", " ").title()

        # Extract top 3 keywords/keyphrases from the chunk text
        try:
            kws = kw_model.extract_keywords(
                chunk.page_content,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=3,
            )
            keywords = ", ".join(kw for kw, _ in kws) if kws else "general"
        except Exception:
            keywords = "general"

        new_meta = dict(chunk.metadata)
        new_meta["topic"]      = topic
        new_meta["topic_page"] = f"{topic} — p.{page}"
        new_meta["keywords"]   = keywords
        chunk.metadata = new_meta
        tagged.append(chunk)
    chunks = tagged

    print(f"[INFO] Split into {len(chunks)} chunks. Embedding and indexing...")
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(CHROMA_PATH),
    )
    print(f"[INFO] Chroma DB saved to {CHROMA_PATH}")
    return vs


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_context(vs: Chroma, query: str, k: int, max_docs: int,
                     source_filter: str = None) -> List[Document]:
    queries = [query, f"{query} database systems", f"{query} CMU DBMS"]
    all_docs = []
    for q in queries:
        docs = vs.similarity_search(q, k=k)
        if source_filter:
            docs = [d for d in docs if source_filter.lower() in
                    Path(d.metadata.get("source", "")).name.lower()]
        all_docs.extend(docs)
    return dedupe_documents(all_docs, max_docs=max_docs)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_topics_prompt(user_text: str, documents: List[Document], n: int) -> str:
    ctx = format_context_blocks(documents)
    return f"""
You are generating quiz subtopics for a Database Systems course.

User request: {user_text}

Using only the context below, produce exactly {n} distinct subtopics.
Each must be concise, specific, and useful for writing one quiz question.

Output JSON only:
{{
  "topics": [{{"topic": "string"}}]
}}

Context:
{ctx}
""".strip()


def build_question_prompt(subtopic: str, documents: List[Document],
                           difficulty: str = "medium", style: str = "conceptual") -> str:
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
    return f"""
You are generating a single quiz question for a Database Systems course.

Subtopic: {subtopic}
Style: {styles[style]}
Difficulty: {diffs[difficulty]}

Return ONLY a single JSON object with keys "question" and "sources".
Do NOT return a list or add text outside the JSON.

Example:
{{
  "question": "What is the purpose of a B+ tree index?",
  "sources": ["CHUNK 1", "CHUNK 2"]
}}

Context:
{ctx}
""".strip()


def build_correct_answer_prompt(question: str, documents: List[Document]) -> str:
    ctx = format_context_blocks(documents)
    return f"""
You are generating the correct answer for a Database Systems quiz question.

Question: {question}

Return ONLY a single JSON object with keys "answer", "explanation", and "sources".

Example:
{{
  "answer": "A B+ tree index speeds up retrieval by maintaining sorted keys.",
  "explanation": "B+ trees allow efficient range queries and point lookups.",
  "sources": ["CHUNK 1"]
}}

Context:
{ctx}
""".strip()


def build_distractor_prompt(question: str, correct_answer: str,
                             previous_answers: List[str],
                             documents: List[Document]) -> str:
    ctx = format_context_blocks(documents)
    return f"""
You are generating one plausible but incorrect answer for a Database Systems quiz.

Question: {question}
Correct answer: {correct_answer}
Previous incorrect answers (do not repeat): {json.dumps(previous_answers)}

Return ONLY a single JSON object with keys "incorrect_answer", "explanation", and "sources".

Example:
{{
  "incorrect_answer": "A hash index is better for range queries than a B+ tree.",
  "explanation": "Hash indexes do not support range queries; B+ trees do.",
  "sources": ["CHUNK 1"]
}}

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

Example:
{{
  "question": "A _____ index contains an entry for every search-key value in the file.",
  "answer": "dense",
  "explanation": "A dense index has one entry per search-key value.",
  "sources": ["CHUNK 1"]
}}

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

Example:
{{
  "question": "Explain the difference between clustered and non-clustered indexes.",
  "model_answer": "A clustered index determines the physical order of data...",
  "key_points": ["Physical vs logical ordering", "One clustered per table"],
  "sources": ["CHUNK 1"]
}}

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

Example:
{{
  "statement": "A sparse index always has fewer entries than a dense index.",
  "answer": "True",
  "explanation": "Sparse indexes store entries for only some search-key values.",
  "sources": ["CHUNK 2"]
}}

Context:
{ctx}
""".strip()


# ---------------------------------------------------------------------------
# Groq response cache (in-memory, lives for the duration of the run)
# Keyed by SHA256 of the prompt — same prompt = same response, no API call
# ---------------------------------------------------------------------------
_groq_cache: Dict[str, Any] = {}

def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Groq API call
# ---------------------------------------------------------------------------

async def call_groq_json(cfg: Dict[str, str], prompt: str, max_retries: int = 15) -> Any:
    # Check cache first
    key = _cache_key(prompt)
    if key in _groq_cache:
        return _groq_cache[key]

    headers = {
        "Authorization": f"Bearer {cfg['groq_api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg["groq_model"],
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "You are a precise quiz-generation assistant. Return valid JSON only."},
            {"role": "user",   "content": prompt},
        ],
    }
    # Proactive delay: stay under 30 req/min free tier (2s = ~15 req/min, safe headroom)
    await asyncio.sleep(2)

    for attempt in range(max_retries):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                cfg["groq_base_url"], headers=headers, json=payload,
                timeout=aiohttp.ClientTimeout(total=180),
            ) as resp:
                if resp.status == 429:
                    await asyncio.sleep(min(2 ** attempt, 60))
                    continue
                resp.raise_for_status()
                body = await resp.json()

        raw = body["choices"][0]["message"]["content"].strip()
        result = safe_json_extract(raw)
        _groq_cache[key] = result  # store in cache
        return result

    raise RuntimeError("Groq rate limit exceeded after max retries.")


# ---------------------------------------------------------------------------
# Generation steps
# ---------------------------------------------------------------------------

async def generate_topics(cfg, text, docs, n) -> List[str]:
    payload = await call_groq_json(cfg, build_topics_prompt(text, docs, n))
    if isinstance(payload, list):
        payload = {"topics": payload}
    validated = TopicList.model_validate(payload)
    return [item.topic for item in validated.topics[:n]]


async def generate_question_for_topic(cfg, subtopic, docs,
                                       difficulty="medium", style="conceptual") -> QuestionItem:
    payload = await call_groq_json(cfg, build_question_prompt(subtopic, docs, difficulty, style))
    if isinstance(payload, list):
        raise ValueError(f"Unexpected list from Groq: {payload}")
    return QuestionItem.model_validate(payload)


async def generate_correct_answer(cfg, question, docs) -> AnswerItem:
    payload = await call_groq_json(cfg, build_correct_answer_prompt(question, docs))
    if isinstance(payload, list):
        raise ValueError(f"Unexpected list from Groq: {payload}")
    return AnswerItem.model_validate(payload)


async def generate_distractors(cfg, question, correct_answer, docs,
                                num_distractors=3) -> List[DistractorItem]:
    """Generate all distractors in parallel for speed."""
    async def _one(previous: List[str]) -> DistractorItem:
        payload = await call_groq_json(cfg, build_distractor_prompt(
            question, correct_answer, previous, docs))
        if isinstance(payload, list):
            raise ValueError(f"Unexpected list from Groq: {payload}")
        return DistractorItem.model_validate(payload)

    # Fire all distractor requests concurrently
    tasks = [_one([]) for _ in range(num_distractors)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    distractors, seen = [], set()
    for r in results:
        if isinstance(r, Exception):
            print(f"[WARN] Distractor generation error: {r}")
            continue
        if r.incorrect_answer in seen or r.incorrect_answer == correct_answer:
            continue
        seen.add(r.incorrect_answer)
        distractors.append(r)
    return distractors


async def generate_fill_blank(cfg, subtopic, docs, difficulty="medium") -> Dict:
    payload = await call_groq_json(cfg, build_fill_blank_prompt(subtopic, docs, difficulty))
    if isinstance(payload, list):
        raise ValueError(f"Unexpected list from Groq: {payload}")
    return payload


async def generate_long_answer(cfg, subtopic, docs,
                                difficulty="medium", style="conceptual") -> Dict:
    payload = await call_groq_json(cfg, build_long_answer_prompt(subtopic, docs, difficulty, style))
    if isinstance(payload, list):
        raise ValueError(f"Unexpected list from Groq: {payload}")
    return payload


async def generate_true_false(cfg, subtopic, docs, difficulty="medium") -> Dict:
    payload = await call_groq_json(cfg, build_true_false_prompt(subtopic, docs, difficulty))
    if isinstance(payload, list):
        raise ValueError(f"Unexpected list from Groq: {payload}")
    return payload


# ---------------------------------------------------------------------------
# Main quiz builder
# ---------------------------------------------------------------------------

async def build_quiz(
    cfg: Dict[str, str],
    vector_store: Chroma,
    user_text: str,
    num_questions: int,
    difficulty: str,
    retrieval_k: int,
    max_docs: int,
    style: str = "conceptual",
    num_options: int = 4,
    source_filter: str = None,
    question_type: str = "mcq",
) -> Dict[str, Any]:

    docs = retrieve_context(vector_store, user_text, retrieval_k, max_docs, source_filter)
    if not docs:
        raise RuntimeError(f"No documents retrieved. Make sure PDFs are in {DATA_PATH}")

    topics = await generate_topics(cfg, user_text, docs, num_questions)
    quiz_questions: List[Dict[str, Any]] = []

    for idx, topic in enumerate(topics, 1):
        try:
            entry: Dict[str, Any] = {"id": f"q_{idx}", "type": question_type, "topic": topic}

            if question_type == "mcq":
                # Step 1: generate question
                q = await generate_question_for_topic(cfg, topic, docs, difficulty, style)
                # Step 2: generate answer + distractors in parallel (both need the question text)
                ans, dist = await asyncio.gather(
                    generate_correct_answer(cfg, q.question, docs),
                    generate_distractors(cfg, q.question, "", docs, num_options - 1),
                )
                # Re-run distractors now that we have the correct answer for deduplication
                dist = await generate_distractors(cfg, q.question, ans.answer, docs, num_options - 1)
                opts = [ans.answer] + [d.incorrect_answer for d in dist]
                random.shuffle(opts)
                entry.update({
                    "question":    q.question,
                    "options":     opts,
                    "answer":      ans.answer,
                    "explanation": ans.explanation,
                    "sources":     sorted({*q.sources, *ans.sources,
                                           *[s for d in dist for s in d.sources]}),
                    "incorrect_answers": [
                        {"answer": d.incorrect_answer, "explanation": d.explanation,
                         "sources": d.sources} for d in dist
                    ],
                })

            elif question_type == "fill_blank":
                r = await generate_fill_blank(cfg, topic, docs, difficulty)
                entry.update({
                    "question":    r.get("question", ""),
                    "answer":      r.get("answer", ""),
                    "explanation": r.get("explanation", ""),
                    "sources":     r.get("sources", []),
                })

            elif question_type == "long_answer":
                r = await generate_long_answer(cfg, topic, docs, difficulty, style)
                entry.update({
                    "question":     r.get("question", ""),
                    "model_answer": r.get("model_answer", ""),
                    "key_points":   r.get("key_points", []),
                    "sources":      r.get("sources", []),
                })

            elif question_type == "true_false":
                r = await generate_true_false(cfg, topic, docs, difficulty)
                entry.update({
                    "statement":   r.get("statement", ""),
                    "answer":      r.get("answer", ""),
                    "explanation": r.get("explanation", ""),
                    "sources":     r.get("sources", []),
                })

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
            "generated_at_utc":  utc_compact_ts(),
            "model":             cfg["groq_model"],
            "requested_questions": num_questions,
            "generated_questions": len(quiz_questions),
            "style":         style,
            "difficulty":    difficulty,
            "num_options":   num_options,
            "source_filter": source_filter or "all",
        },
    }


# ---------------------------------------------------------------------------
# Save quiz
# ---------------------------------------------------------------------------

def save_quiz_locally(quiz: Dict[str, Any]) -> Path:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_PATH / f"{quiz['quiz_id']}.json"
    out.write_text(json.dumps(quiz, indent=2, ensure_ascii=False))
    return out


# ---------------------------------------------------------------------------
# List topics
# ---------------------------------------------------------------------------

def list_topics(vs: Chroma) -> None:
    data = vs.get(include=["metadatas"])
    counts: Dict[str, int] = {}
    sample_keywords: Dict[str, list] = {}
    for m in data["metadatas"]:
        t = m.get("topic", "Unknown")
        counts[t] = counts.get(t, 0) + 1
        # Collect up to 5 sample keywords per topic
        kw = m.get("keywords", "")
        if kw and kw != "general" and t not in sample_keywords:
            sample_keywords[t] = []
        if kw and kw != "general" and len(sample_keywords.get(t, [])) < 5:
            sample_keywords.setdefault(t, []).append(kw)

    print("\n=== TOPICS IN CHROMA DB ===\n")
    for t, c in sorted(counts.items()):
        print(f"  {t:<45} {c} chunks")
        samples = sample_keywords.get(t, [])
        if samples:
            print(f"  {'':45} e.g. {' | '.join(samples[:3])}")
    print(f"\nTotal: {sum(counts.values())} chunks across {len(counts)} sources")
    print("\nTip: use --source-filter \"lectures.pdf\" to target a source\n")


# ---------------------------------------------------------------------------
# Interactive configurator
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    print(f"\n{'─' * 50}\n  {text}\n{'─' * 50}")

def _ask(prompt: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    raw = input(f"{prompt}{hint}: ").strip()
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
        label = QUESTION_TYPE_LABELS.get(o, o)
        print(f"  {i}. {label}{' (selected)' if o in defaults else ''}")
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
        if valid and selected:
            return selected

def build_config_interactively(vector_store: Chroma = None) -> Dict:
    _banner("Quiz configurator")
    print("Answer each question to configure your quiz.")
    print("Press Enter to accept the default shown in [brackets].\n")

    cfg = dict(DEFAULT_CONFIG)

    # Topic
    cfg["topic"] = _ask("What topic should the quiz cover?", cfg["topic"])

    # Number of questions
    while True:
        raw = _ask("How many questions?", str(cfg["num_questions"]))
        if raw.isdigit() and 1 <= int(raw) <= 20:
            cfg["num_questions"] = int(raw); break
        print("  Enter a number between 1 and 20.")

    # Question types
    cfg["question_types"] = _choose_many(
        "Which question types do you want?",
        list(QUESTION_TYPE_LABELS.keys()),
        cfg["question_types"],
    )

    # MCQ options
    if "mcq" in cfg["question_types"]:
        cfg["num_options"] = int(_choose(
            "How many answer options per MCQ?", ["3", "4", "5"], str(cfg["num_options"])
        ))

    # Difficulty
    cfg["difficulty"] = _choose("Difficulty level?",
                                 ["easy", "medium", "hard"], cfg["difficulty"])

    # Style
    cfg["style"] = _choose("Question style?",
                            ["conceptual", "scenario", "definition"], cfg["style"])

    # Source filter
    if vector_store:
        data = vector_store.get(include=["metadatas"])
        sources = sorted({Path(m.get("source","")).name
                          for m in data["metadatas"] if m.get("source")})
        if sources:
            print("\nAvailable PDF sources:")
            for i, s in enumerate(sources, 1):
                print(f"  {i}. {s}")
            print("  0. All sources (default)")
            raw = input("Filter to a specific PDF? Enter number or 0: ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(sources):
                cfg["source_filter"] = sources[int(raw)-1]
            else:
                cfg["source_filter"] = None
    else:
        raw = _ask("Filter to a specific PDF filename? (leave blank for all)", "")
        cfg["source_filter"] = raw if raw else None

    # Summary
    _banner("Configuration summary")
    print(f"  Topic          : {cfg['topic']}")
    print(f"  Questions      : {cfg['num_questions']} per type")
    print(f"  Types          : {', '.join(QUESTION_TYPE_LABELS.get(t,t) for t in cfg['question_types'])}")
    print(f"  Difficulty     : {cfg['difficulty']}")
    print(f"  Style          : {cfg['style']}")
    if "mcq" in cfg["question_types"]:
        print(f"  MCQ options    : {cfg['num_options']}")
    print(f"  Source filter  : {cfg['source_filter'] or 'all sources'}")

    confirm = _ask("\nGenerate quiz with these settings? (yes/no)", "yes").lower()
    if confirm not in ("yes", "y"):
        print("Cancelled.")
        sys.exit(0)

    return cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a quiz locally from PDF files.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  Interactive mode (recommended):
    python scripts/quiz_builder_local.py --interactive

  Save config without running:
    python scripts/quiz_builder_local.py --interactive --save-config my_quiz.json

  Run from a saved config:
    python scripts/quiz_builder_local.py --config my_quiz.json

  Direct CLI:
    python scripts/quiz_builder_local.py --text "indexing" --num-questions 3 --difficulty hard

  List available topics:
    python scripts/quiz_builder_local.py --list-topics

  Force re-embed all PDFs:
    python scripts/quiz_builder_local.py --interactive --reindex
        """
    )
    p.add_argument("--interactive",   action="store_true",
                   help="Ask configuration questions one at a time.")
    p.add_argument("--config",        metavar="FILE",
                   help="Load config from a JSON file and run.")
    p.add_argument("--save-config",   metavar="FILE",
                   help="Save interactive config to file without running.")
    p.add_argument("--list-topics",   action="store_true",
                   help="Print all topics in the Chroma DB and exit.")
    p.add_argument("--reindex",       action="store_true",
                   help="Force re-embedding of all PDFs.")
    # Direct CLI flags (used when not in --interactive or --config mode)
    p.add_argument("--text",          type=str, default="Generate a database systems quiz.")
    p.add_argument("--num-questions", type=int, default=3)
    p.add_argument("--difficulty",    type=str, default="medium",
                   choices=["easy", "medium", "hard"])
    p.add_argument("--style",         type=str, default="conceptual",
                   choices=["conceptual", "scenario", "definition"])
    p.add_argument("--num-options",   type=int, default=4, choices=[3, 4, 5])
    p.add_argument("--source-filter", type=str, default=None)
    p.add_argument("--question-type", type=str, default="mcq",
                   choices=["mcq", "fill_blank", "long_answer", "true_false"])
    p.add_argument("--retrieval-k",   type=int, default=6)
    p.add_argument("--max-docs",      type=int, default=12)
    return p.parse_args()


def main() -> None:
    groq_cfg = load_config()
    args     = parse_args()

    # Uncomment below to wipe and rebuild the Chroma DB from scratch
    # WARNING: this deletes all existing tags and chunks — use only if needed
    # if args.reindex and CHROMA_PATH.exists():
    #     import shutil
    #     shutil.rmtree(CHROMA_PATH)
    #     print(f"[INFO] Cleared Chroma DB at {CHROMA_PATH}")

    vs = load_and_index_documents()

    if args.list_topics:
        list_topics(vs); return

    # ── Determine quiz config ─────────────────────────────────────────────
    if args.config:
        with open(args.config) as f:
            quiz_cfg = json.load(f)
        print(f"[INFO] Loaded config from {args.config}")

    elif args.interactive:
        quiz_cfg = build_config_interactively(vector_store=vs)
        if args.save_config:
            Path(args.save_config).write_text(json.dumps(quiz_cfg, indent=2))
            print(f"[INFO] Config saved to {args.save_config}")
            return

    else:
        # Direct CLI mode — single question type
        quiz_cfg = {
            "topic":          args.text.strip(),
            "num_questions":  args.num_questions,
            "question_types": [args.question_type],
            "difficulty":     args.difficulty,
            "style":          args.style,
            "num_options":    args.num_options,
            "source_filter":  args.source_filter,
            "retrieval_k":    args.retrieval_k,
            "max_docs":       args.max_docs,
        }

    # ── Print config summary ──────────────────────────────────────────────
    print(f"\n[CONFIG] Topic     : {quiz_cfg['topic']}")
    print(f"[CONFIG] Questions : {quiz_cfg['num_questions']} per type")
    print(f"[CONFIG] Types     : {', '.join(quiz_cfg['question_types'])}")
    print(f"[CONFIG] Difficulty: {quiz_cfg['difficulty']}")
    print(f"[CONFIG] Style     : {quiz_cfg['style']}")
    print(f"[CONFIG] Filter    : {quiz_cfg.get('source_filter') or 'all sources'}\n")

    # ── Generate ──────────────────────────────────────────────────────────
    async def run() -> None:
        all_questions = []

        for qtype in quiz_cfg["question_types"]:
            print(f"\n[INFO] Generating {QUESTION_TYPE_LABELS.get(qtype, qtype)} questions...")
            quiz = await build_quiz(
                cfg=groq_cfg,
                vector_store=vs,
                user_text=quiz_cfg["topic"],
                num_questions=quiz_cfg["num_questions"],
                difficulty=quiz_cfg["difficulty"],
                retrieval_k=quiz_cfg.get("retrieval_k", 6),
                max_docs=quiz_cfg.get("max_docs", 12),
                style=quiz_cfg["style"],
                num_options=quiz_cfg.get("num_options", 4),
                source_filter=quiz_cfg.get("source_filter"),
                question_type=qtype,
            )
            all_questions.extend(quiz["questions"])

        merged = {
            "quiz_id":       f"quiz_{utc_compact_ts()}",
            "config":        quiz_cfg,
            "question_types": quiz_cfg["question_types"],
            "questions":     all_questions,
            "run_metadata": {
                "generated_at_utc":    utc_compact_ts(),
                "model":               groq_cfg["groq_model"],
                "requested_per_type":  quiz_cfg["num_questions"],
                "total_generated":     len(all_questions),
                "types":               quiz_cfg["question_types"],
            },
        }

        out = save_quiz_locally(merged)
        print(f"\nQuiz saved to  : {out}")
        print(f"Total questions: {len(all_questions)}")
        print(f"Types          : {', '.join(quiz_cfg['question_types'])}")

    asyncio.run(run())


if __name__ == "__main__":
    main()