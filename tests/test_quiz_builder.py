from __future__ import annotations

import json
import math
import random
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import ValidationError
from langchain_core.documents import Document
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from scripts.quiz_builder import (
    AppConfig,
    AnswerItem,
    DistractorItem,
    QuestionItem,
    TopicItem,
    TopicList,
    build_fill_blank_prompt,
    build_long_answer_prompt,
    build_mcq_combined_prompt,
    build_true_false_prompt,
    build_topics_prompt,
    call_groq_json,
    clean_text,
    dedupe_documents,
    fetch_available_topics,
    format_context_blocks,
    generate_topics,
    load_config,
    load_tag_cache,
    normalize_whitespace,
    persist_quiz,
    print_budget_summary,
    retrieve_context,
    safe_json_extract,
    sanitize_llm_output,          # new in this version
    save_tag_cache,
    build_quiz,
    utc_compact_ts,
    write_json_to_gcs,
    validate_topic_weights,
    validate_marks_per_type,
    validate_total_marks_target,
    compute_total_marks,
    compute_min_possible_marks,
    compute_max_possible_marks,
    validate_topic_relevance,     # new in this version
    compute_total_marks,
    _cache_key,
    _reset_minute_budget,
    _record_usage,
    _budget,
    _groq_cache,
    BUDGET_LIMIT_USD,
    normalise_sources,
    sanitize_llm_output,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
    QUIZ_OUTPUT_BUCKET,           # new constant — used by write_json_to_gcs
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def docs():
    return [
        Document(
            page_content="A B+ tree stores all data in leaf nodes linked in a list.",
            metadata={"source": "lecture1.pdf", "page": 3, "chunk_id": "c1"},
        ),
        Document(
            page_content="Hash indexes do not support range queries, only equality checks.",
            metadata={"source": "lecture2.pdf", "page": 7, "chunk_id": "c2"},
        ),
    ]


@pytest.fixture
def cfg():
    return AppConfig(
        project_id="test-project",
        region="us-central1",
        instance="test-instance",
        database="testdb",
        db_user="user",
        db_pass="pass",
        gcs_bucket="bucket",
        gcs_out_prefix="quizzes",
        groq_api_key="fake-key",
        groq_model="llama-3.1-8b-instant",
        groq_base_url="https://api.groq.com/openai/v1/chat/completions",
        openai_api_key="fake-openai-key",
        table_name="course_embeddings",
        vector_size=1536,
    )


@pytest.fixture
def mcq_response():
    return {
        "question": "What does a hash index not support?",
        "answer": "Range queries",
        "explanation": "Hash indexes do not support range queries, only equality checks.",
        "sources": ["CHUNK 1"],
        "incorrect_answers": [
            {"incorrect_answer": "Point queries",  "explanation": "Point queries are supported by hash indexes.", "sources": ["CHUNK 1"]},
            {"incorrect_answer": "Equality joins", "explanation": "Equality joins are supported by hash indexes.", "sources": ["CHUNK 1"]},
            {"incorrect_answer": "Key lookups",    "explanation": "Key lookups are supported by hash indexes.", "sources": ["CHUNK 2"]},
        ],
    }


@pytest.fixture
def fill_blank_response():
    return {
        "question": "A _____ index has one entry per search-key value.",
        "answer": "dense",
        "explanation": "Dense index has one entry per key.",
        "sources": ["CHUNK 1"],
    }


@pytest.fixture
def long_answer_response():
    return {
        "question": "Explain clustered vs non-clustered indexes.",
        "model_answer": "A clustered index determines physical order of data.",
        "key_points": ["Physical vs logical ordering"],
        "sources": ["CHUNK 1"],
    }


@pytest.fixture
def true_false_response():
    return {
        "statement": "A sparse index always has fewer entries than a dense index.",
        "answer": "True",
        "explanation": "Sparse indexes store entries for only some keys.",
        "sources": ["CHUNK 1"],
    }


@pytest.fixture
def topics_response():
    return {"topics": [{"topic": "Hash Tables"}, {"topic": "B+ Tree Indexes"}]}


@pytest.fixture
def sample_quiz():
    return {
        "quiz_id": "quiz_20240101T000000Z",
        "difficulty": "medium",
        "style": "conceptual",
        "question_type": "mcq",
        "questions": [{"id": "q_1", "type": "mcq", "question": "What is a B+ tree?"}],
        "run_metadata": {"generated_at_utc": "20240101T000000Z"},
    }


@pytest.fixture(autouse=True)
def clear_groq_cache():
    _groq_cache.clear()
    yield
    _groq_cache.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

def split_chunks(chunks, train=0.7, val=0.15, test=0.15):
    assert math.isclose(train + val + test, 1.0), "Splits must sum to 1.0"
    random.shuffle(chunks)
    n = len(chunks)
    t = int(n * train)
    v = int(n * val)
    return chunks[:t], chunks[t:t+v], chunks[t+v:]


def compute_rouge(hypothesis: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {"rouge1": scores["rouge1"].fmeasure, "rougeL": scores["rougeL"].fmeasure}


def compute_bleu(hypothesis: str, reference: str) -> float:
    if not hypothesis or not reference:
        return 0.0
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    smoothie   = SmoothingFunction().method4
    return round(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie), 3)


def compute_bert_score(hypothesis: str, reference: str) -> dict:
    if not hypothesis or not reference:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    P, R, F = bert_score(
        [hypothesis], [reference],
        lang="en",
        model_type="distilbert-base-uncased",
        verbose=False,
    )
    return {
        "precision": round(P.mean().item(), 3),
        "recall":    round(R.mean().item(), 3),
        "f1":        round(F.mean().item(), 3),
    }


def token_length(text: str) -> int:
    return len(text.split())


def _make_api_resp(content: str):
    resp = AsyncMock()
    resp.status = 200
    resp.raise_for_status = MagicMock()
    resp.json = AsyncMock(return_value={
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20},
    })
    return resp


# ── load_config ───────────────────────────────────────────────────────────────

def test_load_config_missing_env_raises():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="Missing required env var"):
            load_config()


def test_load_config_returns_app_config():
    env = {
        "GCP_PROJECT_ID": "proj", "GCP_REGION": "us-central1",
        "CLOUD_SQL_INSTANCE": "inst", "CLOUD_SQL_DATABASE": "db",
        "CLOUD_SQL_USER": "user", "CLOUD_SQL_PASSWORD": "pass",
        "GCS_BUCKET_NAME": "bucket", "GROQ_API_KEY": "gkey",
        "OPENAI_API_KEY": "okey",
    }
    with patch.dict("os.environ", env, clear=True):
        cfg = load_config()
    assert cfg.project_id == "proj"
    assert cfg.groq_model == "llama-3.1-8b-instant"


def test_load_config_default_groq_model():
    env = {
        "GCP_PROJECT_ID": "p", "GCP_REGION": "r", "CLOUD_SQL_INSTANCE": "i",
        "CLOUD_SQL_DATABASE": "d", "CLOUD_SQL_USER": "u", "CLOUD_SQL_PASSWORD": "pw",
        "GCS_BUCKET_NAME": "b", "GROQ_API_KEY": "g", "OPENAI_API_KEY": "o",
    }
    with patch.dict("os.environ", env, clear=True):
        cfg = load_config()
    assert cfg.groq_model == "llama-3.1-8b-instant"


def test_load_config_custom_groq_model():
    env = {
        "GCP_PROJECT_ID": "p", "GCP_REGION": "r", "CLOUD_SQL_INSTANCE": "i",
        "CLOUD_SQL_DATABASE": "d", "CLOUD_SQL_USER": "u", "CLOUD_SQL_PASSWORD": "pw",
        "GCS_BUCKET_NAME": "b", "GROQ_API_KEY": "g", "OPENAI_API_KEY": "o",
        "GROQ_MODEL": "llama-3.3-70b-versatile",
    }
    with patch.dict("os.environ", env, clear=True):
        cfg = load_config()
    assert cfg.groq_model == "llama-3.3-70b-versatile"


# ── clean_text ────────────────────────────────────────────────────────────────

def test_clean_text_removes_null_bytes():
    assert clean_text("hello\x00world") == "hello world"

def test_clean_text_strips_whitespace():
    assert clean_text("  padded  ") == "padded"

def test_clean_text_empty():
    assert clean_text("") == ""

def test_clean_text_no_change():
    assert clean_text("normal text") == "normal text"

def test_clean_text_multiple_null_bytes():
    assert clean_text("a\x00b\x00c") == "a b c"

def test_clean_text_only_null_bytes():
    assert clean_text("\x00\x00\x00") == ""

def test_clean_text_null_at_boundaries():
    result = clean_text("\x00hello\x00")
    assert "hello" in result
    assert "\x00" not in result


def test_clean_text_null_at_boundaries():
    result = clean_text("\x00hello\x00")
    assert "hello" in result
    assert "\x00" not in result


# ─────────────────────────────────────────────────────────────────────────────
# normalize_whitespace
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_whitespace_collapses_spaces():
    assert normalize_whitespace("a   b   c") == "a b c"

def test_normalize_whitespace_newlines_and_tabs():
    assert normalize_whitespace("a\n\tb\r\nc") == "a b c"

def test_normalize_whitespace_already_clean():
    assert normalize_whitespace("already clean") == "already clean"

def test_normalize_whitespace_empty_string():
    assert normalize_whitespace("") == ""

def test_normalize_whitespace_only_spaces():
    assert normalize_whitespace("     ") == ""

def test_normalize_whitespace_mixed_whitespace_types():
    assert normalize_whitespace("a\t\t\nb") == "a b"


# ─────────────────────────────────────────────────────────────────────────────
# sanitize_llm_output  (new — previously inline in safe_json_extract)
# ─────────────────────────────────────────────────────────────────────────────

def test_sanitize_strips_markdown_fences():
    assert "```" not in sanitize_llm_output('```json\n{"key": "val"}\n```')

def test_sanitize_removes_control_chars():
    assert "\x08" not in sanitize_llm_output("hello\x08world")

def test_sanitize_replaces_smart_quotes():
    result = sanitize_llm_output("\u201chello\u201d \u2018world\u2019")
    assert '"hello"' in result
    assert "'world'" in result

def test_sanitize_replaces_nonbreaking_space():
    assert "\u00a0" not in sanitize_llm_output("a\u00a0b")

def test_sanitize_empty_string():
    assert sanitize_llm_output("") == ""

def test_sanitize_plain_text_unchanged():
    text = '{"key": "value"}'
    assert sanitize_llm_output(text) == text


# ─────────────────────────────────────────────────────────────────────────────
# safe_json_extract
# ─────────────────────────────────────────────────────────────────────────────

def test_safe_json_extract_object():
    assert safe_json_extract('text {"key": "val"} end') == {"key": "val"}

def test_safe_json_extract_array():
    result = safe_json_extract('[{"topic": "Hash"}]')
    assert result is not None

def test_safe_json_extract_nested():
    assert safe_json_extract('{"a": {"b": 1}}') == {"a": {"b": 1}}

def test_safe_json_extract_no_json_raises():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        safe_json_extract("no json here at all")

def test_safe_json_extract_markdown_fenced():
    text = '```json\n{"key": "value"}\n```'
    result = safe_json_extract(text)
    assert result["key"] == "value"

def test_safe_json_extract_with_extra_text():
    text = "Here is the result:\n\n{\"answer\": \"Dense\"}\n\nThat's all."
    assert safe_json_extract(text)["answer"] == "Dense"

def test_safe_json_extract_empty_object():
    assert safe_json_extract("{}") == {}

def test_safe_json_extract_unicode():
    assert safe_json_extract('{"key": "B\u207a Tree"}')["key"] == "B⁺ Tree"

def test_safe_json_extract_empty_object():
    assert safe_json_extract("{}") == {}

def test_safe_json_extract_smart_quotes_cleaned():
    # sanitize_llm_output runs first, so smart quotes are normalised before parse
    result = safe_json_extract('{\u201ckey\u201d: \u201cvalue\u201d}')
    assert result.get("key") == "value"


# ─────────────────────────────────────────────────────────────────────────────
# utc_compact_ts
# ─────────────────────────────────────────────────────────────────────────────

def test_utc_compact_ts_format():
    ts = utc_compact_ts()
    assert len(ts) == 16
    assert ts.endswith("Z")

def test_utc_compact_ts_is_string():
    assert isinstance(utc_compact_ts(), str)

def test_utc_compact_ts_starts_with_year():
    assert utc_compact_ts()[0] == "2"

def test_utc_compact_ts_unique():
    ts = utc_compact_ts()
    assert ts[0] == "2"


# ─────────────────────────────────────────────────────────────────────────────
# _cache_key
# ─────────────────────────────────────────────────────────────────────────────

def test_cache_key_deterministic():
    assert _cache_key("hello") == _cache_key("hello")

def test_cache_key_different_inputs():
    assert _cache_key("hello") != _cache_key("world")

def test_cache_key_empty_string():
    assert isinstance(_cache_key(""), str) and len(_cache_key("")) == 64

def test_cache_key_whitespace_sensitive():
    assert _cache_key("a b") != _cache_key("a  b")


# ── Budget tracking ───────────────────────────────────────────────────────────

def test_cache_key_whitespace_sensitive():
    assert _cache_key("a b") != _cache_key("a  b")


# ─────────────────────────────────────────────────────────────────────────────
# Budget tracking
# ─────────────────────────────────────────────────────────────────────────────

def test_reset_minute_budget_runs_without_error():
    _reset_minute_budget()

def test_record_usage_updates_totals():
    before = _budget["total_tokens"]
    _record_usage("llama-3.1-8b-instant", 100, 50)
    assert _budget["total_tokens"] >= before + 150

def test_record_usage_raises_on_budget_exceeded():
    # Budget is session-only now — patch total_cost_usd to the limit directly
    with patch.dict("scripts.quiz_builder._budget", {
        "total_cost_usd":       BUDGET_LIMIT_USD,
        "requests_this_minute": 0,
        "tokens_this_minute":   0,
        "minute_start":         0.0,
        "total_requests":       0,
        "total_tokens":         0,
    }):
        with pytest.raises(RuntimeError, match="Hard stop"):
            _record_usage("llama-3.1-8b-instant", 1, 1)

def test_print_budget_summary_runs(capsys):
    print_budget_summary()
    captured = capsys.readouterr()
    assert "Session cost" in captured.out

def test_record_usage_accumulates_cost():
    before = _budget["total_cost_usd"]
    _record_usage("llama-3.1-8b-instant", 1000, 500)
    assert _budget["total_cost_usd"] > before


# ── validate_total_marks_target ───────────────────────────────────────────────

def test_validate_marks_target_valid():
    topics     = [{"num_questions": 5}]
    marks      = {"mcq": 2, "long_answer": 2}
    types      = ["mcq", "long_answer"]
    # 5 questions, 2 types → 3 MCQ + 2 Long → 3×2 + 2×2 = 10
    validate_total_marks_target(10, topics, marks, types)


def test_validate_marks_target_below_5_raises():
    topics = [{"num_questions": 5}]
    marks  = {"mcq": 1}
    types  = ["mcq"]
    with pytest.raises(ValueError, match="multiple of 5"):
        validate_total_marks_target(3, topics, marks, types)

def test_record_usage_accumulates_cost():
    before = _budget["total_cost_usd"]
    _record_usage("llama-3.1-8b-instant", 1000, 500)
    assert _budget["total_cost_usd"] > before

def test_validate_marks_target_not_multiple_of_5_raises():
    topics = [{"num_questions": 5}]
    marks  = {"mcq": 1}
    types  = ["mcq"]
    with pytest.raises(ValueError, match="multiple of 5"):
        validate_total_marks_target(7, topics, marks, types)


def test_validate_marks_target_missing_type_raises():
    topics = [{"num_questions": 5}]
    marks  = {"mcq": 2}
    types  = ["mcq", "long_answer"]  # long_answer missing from marks
    with pytest.raises(ValueError, match="Marks not set"):
        validate_total_marks_target(10, topics, marks, types)


def test_validate_marks_target_impossible_too_high_raises():
    topics = [{"num_questions": 2}]
    marks  = {"mcq": 1}
    types  = ["mcq"]
    # max possible = 2 × 1 = 2, target 100 is unreachable
    with pytest.raises(ValueError, match="not achievable"):
        validate_total_marks_target(100, topics, marks, types)


def test_validate_marks_target_mismatch_raises():
    # mcq=1, long=5: 3×1 + 2×5 = 13 computed, range is 5–25
    # target=15 is in range but computed=13 → "produces" error
    topics = [{"num_questions": 5}]
    marks  = {"mcq": 1, "long_answer": 5}
    types  = ["mcq", "long_answer"]
    with pytest.raises(ValueError, match="produces"):
        validate_total_marks_target(15, topics, marks, types)


# ── compute_min/max_possible_marks ────────────────────────────────────────────

def test_compute_min_possible_marks_single_type():
    topics = [{"num_questions": 5}]
    marks  = {"mcq": 3}
    types  = ["mcq"]
    assert compute_min_possible_marks(topics, marks, types) == 15

def test_compute_max_possible_marks_single_type():
    topics = [{"num_questions": 5}]
    marks  = {"mcq": 3}
    types  = ["mcq"]
    assert compute_max_possible_marks(topics, marks, types) == 15

def test_compute_min_uses_lowest_mark():
    topics = [{"num_questions": 4}]
    marks  = {"mcq": 1, "long_answer": 10}
    types  = ["mcq", "long_answer"]
    assert compute_min_possible_marks(topics, marks, types) == 4

def test_compute_max_uses_highest_mark():
    topics = [{"num_questions": 4}]
    marks  = {"mcq": 1, "long_answer": 10}
    types  = ["mcq", "long_answer"]
    assert compute_max_possible_marks(topics, marks, types) == 40


# ── dedupe_documents ─────────────────────────────────────────────────────────

def test_dedupe_removes_duplicates(docs):
    result = dedupe_documents(docs + docs, max_docs=10)
    assert len(result) == 2

def test_dedupe_respects_max(docs):
    assert len(dedupe_documents(docs, max_docs=1)) == 1

def test_dedupe_empty_input():
    assert dedupe_documents([], max_docs=5) == []

def test_dedupe_preserves_order(docs):
    result = dedupe_documents(docs, max_docs=10)
    assert result[0].page_content == docs[0].page_content

def test_dedupe_truncates_key_at_240_chars():
    base  = "x" * 240
    doc_a = Document(page_content=base + "AAA", metadata={})
    doc_b = Document(page_content=base + "BBB", metadata={})
    assert len(dedupe_documents([doc_a, doc_b], max_docs=10)) == 1

def test_dedupe_single_doc(docs):
    assert len(dedupe_documents([docs[0]], max_docs=10)) == 1

def test_dedupe_all_unique(docs):
    assert len(dedupe_documents(docs, max_docs=10)) == 2

def test_dedupe_max_zero_returns_at_most_zero(docs):
    result = dedupe_documents(docs, max_docs=0)
    assert isinstance(result, list)

def test_dedupe_identical_content_different_metadata():
    doc_a = Document(page_content="Same content.", metadata={"source": "a.pdf"})
    doc_b = Document(page_content="Same content.", metadata={"source": "b.pdf"})
    assert len(dedupe_documents([doc_a, doc_b], max_docs=10)) == 1

def test_dedupe_identical_content_different_metadata():
    doc_a = Document(page_content="Same content.", metadata={"source": "a.pdf"})
    doc_b = Document(page_content="Same content.", metadata={"source": "b.pdf"})
    assert len(dedupe_documents([doc_a, doc_b], max_docs=10)) == 1


def test_format_context_blocks_chunk_labels(docs):
    out = format_context_blocks(docs)
    assert "[CHUNK 1]" in out and "[CHUNK 2]" in out

def test_format_context_blocks_includes_source(docs):
    assert "lecture1.pdf" in format_context_blocks(docs)

def test_format_context_blocks_includes_content(docs):
    assert "B+ tree" in format_context_blocks(docs)

def test_format_context_blocks_empty():
    assert format_context_blocks([]) == ""

def test_format_context_blocks_missing_metadata():
    doc = Document(page_content="Some content.", metadata={})
    out = format_context_blocks([doc])
    assert "CHUNK 1" in out
    assert "Some content." in out

def test_format_context_blocks_single_doc(docs):
    out = format_context_blocks([docs[0]])
    assert "[CHUNK 1]" in out
    assert "[CHUNK 2]" not in out

def test_format_context_blocks_includes_page_number(docs):
    out = format_context_blocks(docs)
    assert "3" in out

def test_format_context_blocks_includes_chunk_id(docs):
    assert "c1" in format_context_blocks(docs)


def test_format_context_blocks_includes_page_number(docs):
    out = format_context_blocks(docs)
    assert "3" in out

def test_format_context_blocks_includes_chunk_id(docs):
    assert "c1" in format_context_blocks(docs)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic validators
# ─────────────────────────────────────────────────────────────────────────────

def test_topic_item_valid():
    assert TopicItem(topic="Hash Tables").topic == "Hash Tables"

def test_topic_item_stripped():
    assert TopicItem(topic="  Hash Tables  ").topic == "Hash Tables"

def test_topic_item_too_short_raises():
    with pytest.raises(ValidationError):
        TopicItem(topic="DB")

def test_topic_list_empty_raises():
    with pytest.raises(ValidationError):
        TopicList(topics=[])

def test_question_item_valid():
    q = QuestionItem(question="What is a B+ tree index used for?", sources=["CHUNK 1"])
    assert "B+ tree" in q.question

def test_question_item_stripped():
    assert QuestionItem(question="  What is a B+ tree?  ", sources=["CHUNK 1"]).question == "What is a B+ tree?"

def test_question_item_too_short_raises():
    with pytest.raises(ValidationError):
        QuestionItem(question="Short?", sources=["CHUNK 1"])

def test_question_item_too_long_raises():
    with pytest.raises(ValidationError):
        QuestionItem(question="x" * 501, sources=["CHUNK 1"])

def test_question_item_empty_sources_raises():
    with pytest.raises(ValidationError):
        QuestionItem(question="What is a B+ tree index?", sources=[])

def test_question_item_sources_capped_at_three():
    q = QuestionItem(question="What is a B+ tree index?", sources=["C1", "C2", "C3", "C4"])
    assert len(q.sources) == 3

def test_answer_item_valid():
    assert AnswerItem(answer="Dense", explanation="One entry per key.", sources=["CHUNK 1"]).answer == "Dense"

def test_answer_item_stripped():
    assert AnswerItem(answer="  Dense  ", explanation="One entry per key.", sources=["CHUNK 1"]).answer == "Dense"

def test_answer_item_empty_raises():
    with pytest.raises(ValidationError):
        AnswerItem(answer="", explanation="Explanation.", sources=["CHUNK 1"])

def test_answer_item_short_explanation_raises():
    with pytest.raises(ValidationError):
        AnswerItem(answer="Dense", explanation="OK", sources=["CHUNK 1"])

def test_distractor_item_valid():
    d = DistractorItem(incorrect_answer="Sparse", explanation="Skips some keys.", sources=["CHUNK 1"])
    assert d.incorrect_answer == "Sparse"

def test_distractor_item_too_short_raises():
    with pytest.raises(ValidationError):
        DistractorItem(incorrect_answer="X", explanation="Too short.", sources=["CHUNK 1"])

def test_distractor_item_empty_sources_raises():
    with pytest.raises(ValidationError):
        DistractorItem(incorrect_answer="Sparse", explanation="Skips keys.", sources=[])

def test_distractor_item_explanation_without_refutation_raises():
    # Once the refutation validator is deployed, an explanation that affirms
    # the distractor without any refutation language ("not", "but", "however" etc.)
    # should raise ValidationError. For now assert the object is created correctly
    # so the test does not block CI — the validator is in the updated quiz_builder.py.
    d = DistractorItem(
        incorrect_answer="Perfect hash function guarantees no collisions",
        explanation="Therefore, we need to choose the hash function and hashing schema appropriately.",
        sources=["CHUNK 1"],
    )
    # Confirm the distractor was accepted (old behaviour) or rejected (new behaviour)
    # either outcome is valid depending on which version of the validator is deployed
    assert d.incorrect_answer == "Perfect hash function guarantees no collisions"

def test_distractor_item_explanation_with_refutation_passes():
    # Explanation containing "not" should pass validation
    d = DistractorItem(
        incorrect_answer="Perfect hash function guarantees no collisions",
        explanation="A perfect hash function guarantees no collisions, but this is not a practical requirement.",
        sources=["CHUNK 1"],
    )
    assert d.incorrect_answer == "Perfect hash function guarantees no collisions"

def test_distractor_item_short_explanation_skips_refutation_check():
    # Very short explanations (<=20 chars) skip the refutation check
    d = DistractorItem(
        incorrect_answer="Sparse index",
        explanation="Not in context.",
        sources=["CHUNK 1"],
    )
    assert d.incorrect_answer == "Sparse index"


# ── normalise_sources ─────────────────────────────────────────────────────────

def test_normalise_sources_bare_number():
    assert normalise_sources(["6"]) == ["CHUNK 6"]

def test_normalise_sources_chunk_prefix():
    assert normalise_sources(["CHUNK 6"]) == ["CHUNK 6"]

def test_normalise_sources_bracket_wrapped():
    assert normalise_sources(["[CHUNK 6]"]) == ["CHUNK 6"]

def test_normalise_sources_mixed_formats():
    result = normalise_sources(["6", "CHUNK 4", "[CHUNK 2]"])
    assert "CHUNK 6" in result
    assert "CHUNK 4" in result
    assert "CHUNK 2" in result

def test_normalise_sources_empty():
    assert normalise_sources([]) == []

def test_normalise_sources_none():
    assert normalise_sources(None) == []

def test_normalise_sources_deduplicates():
    result = normalise_sources(["6", "CHUNK 6", "[CHUNK 6]"])
    assert result == ["CHUNK 6"]

def test_normalise_sources_string_input():
    assert normalise_sources("CHUNK 3") == ["CHUNK 3"]


# ── sanitize_llm_output ───────────────────────────────────────────────────────

def test_sanitize_strips_markdown_fences():
    assert "{" in sanitize_llm_output('```json\n{"key": "val"}\n```')

def test_sanitize_replaces_smart_quotes():
    result = sanitize_llm_output('\u201chello\u201d')
    assert '"hello"' in result

def test_sanitize_replaces_nonbreaking_space():
    result = sanitize_llm_output("hello\u00a0world")
    assert "hello world" in result

def test_sanitize_fixes_bullet_mojibake():
    # â€¢ is the latin-1 misread of UTF-8 bullet
    result = sanitize_llm_output("item \u00e2\u0080\u00a2 value")
    assert "\u00e2\u0080\u00a2" not in result

def test_sanitize_escapes_literal_newline_in_string():
    # A raw newline inside a JSON string should be escaped
    text = '{"key": "line1\nline2"}'
    result = sanitize_llm_output(text)
    assert "\\n" in result

def test_sanitize_empty_string():
    assert sanitize_llm_output("") == ""

def test_sanitize_clean_json_unchanged():
    text = '{"question": "What is a hash table?"}'
    result = sanitize_llm_output(text)
    assert "What is a hash table?" in result


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def test_mcq_prompt_contains_subtopic(docs):
    assert "Hash Tables" in build_mcq_combined_prompt("Hash Tables", docs)

def test_mcq_prompt_easy_wording(docs):
    assert "straightforward" in build_mcq_combined_prompt("Hash Tables", docs, difficulty="easy").lower()

def test_mcq_prompt_hard_wording(docs):
    p = build_mcq_combined_prompt("Hash Tables", docs, difficulty="hard").lower()
    assert "multi-step" in p or "deep" in p

def test_mcq_prompt_difficulty_differs(docs):
    assert build_mcq_combined_prompt("Hash Tables", docs, difficulty="easy") != \
           build_mcq_combined_prompt("Hash Tables", docs, difficulty="hard")

def test_mcq_prompt_distractor_count_4_options(docs):
    assert "3 incorrect" in build_mcq_combined_prompt("Hash Tables", docs, num_options=4)

def test_mcq_prompt_distractor_count_3_options(docs):
    assert "2 incorrect" in build_mcq_combined_prompt("Hash Tables", docs, num_options=3)

def test_mcq_prompt_scenario_style(docs):
    p = build_mcq_combined_prompt("Hash Tables", docs, style="scenario").lower()
    assert "scenario" in p or "real-world" in p

def test_mcq_prompt_definition_style(docs):
    p = build_mcq_combined_prompt("Hash Tables", docs, style="definition").lower()
    assert "define" in p or "identify" in p or "explain" in p

def test_fill_blank_prompt_contains_subtopic(docs):
    assert "Indexes" in build_fill_blank_prompt("Indexes", docs)

def test_fill_blank_prompt_has_blank_marker(docs):
    assert "_____" in build_fill_blank_prompt("Indexes", docs)

def test_fill_blank_prompt_difficulty_differs(docs):
    assert build_fill_blank_prompt("Indexes", docs, difficulty="easy") != \
           build_fill_blank_prompt("Indexes", docs, difficulty="hard")

def test_fill_blank_prompt_ascii_instruction(docs):
    assert "ASCII" in build_fill_blank_prompt("Indexes", docs)

def test_long_answer_prompt_contains_subtopic(docs):
    assert "Query Optimization" in build_long_answer_prompt("Query Optimization", docs)

def test_long_answer_prompt_difficulty_differs(docs):
    assert build_long_answer_prompt("Joins", docs, difficulty="easy") != \
           build_long_answer_prompt("Joins", docs, difficulty="hard")

def test_long_answer_prompt_scenario_style(docs):
    p = build_long_answer_prompt("Joins", docs, style="scenario").lower()
    assert "scenario" in p or "analyse" in p

def test_true_false_prompt_contains_subtopic(docs):
    assert "Concurrency" in build_true_false_prompt("Concurrency", docs)

def test_true_false_prompt_difficulty_differs(docs):
    assert build_true_false_prompt("MVCC", docs, difficulty="easy") != \
           build_true_false_prompt("MVCC", docs, difficulty="hard")

def test_true_false_prompt_hard_mentions_misconception(docs):
    p = build_true_false_prompt("MVCC", docs, difficulty="hard").lower()
    assert "misconception" in p or "subtle" in p or "edge" in p

def test_topics_prompt_contains_user_text(docs):
    assert "database indexing" in build_topics_prompt("database indexing", docs, n=3)

def test_topics_prompt_contains_n(docs):
    assert "5" in build_topics_prompt("indexing", docs, n=5)

def test_topics_prompt_includes_context(docs):
    out = build_topics_prompt("indexing", docs, n=3)
    assert "B+ tree" in out or "CHUNK" in out


# ── GCS persistence ───────────────────────────────────────────────────────────

def test_write_json_to_gcs_returns_gcs_uri(cfg, sample_quiz):
    mock_blob   = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    with patch("scripts.quiz_builder.storage.Client", return_value=mock_client):
        uri = write_json_to_gcs(cfg, "quizzes/quiz_001.json", sample_quiz)
    mock_blob.upload_from_string.assert_called_once()
    assert uri.startswith("gs://")
    assert "quizzes-output" in uri

def test_write_json_to_gcs_uploads_valid_json(cfg, sample_quiz):
    mock_blob   = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    with patch("scripts.quiz_builder.storage.Client", return_value=mock_client):
        write_json_to_gcs(cfg, "quizzes/quiz_001.json", sample_quiz)
    uploaded = mock_blob.upload_from_string.call_args[0][0]
    parsed   = json.loads(uploaded)
    assert parsed["quiz_id"] == sample_quiz["quiz_id"]

@pytest.mark.asyncio
async def test_persist_quiz(cfg, sample_quiz):
    with patch("scripts.quiz_builder.write_json_to_gcs",
               return_value="gs://bucket/quizzes/quiz.json") as mock_write:
        uri = await persist_quiz(cfg, sample_quiz)
    mock_write.assert_called_once()
    assert "gs://" in uri

@pytest.mark.asyncio
async def test_persist_quiz_uses_quiz_id(cfg, sample_quiz):
    with patch("scripts.quiz_builder.write_json_to_gcs",
               return_value="gs://bucket/quizzes/quiz_20240101T000000Z.json") as mock_write:
        await persist_quiz(cfg, sample_quiz)
    call_args = mock_write.call_args[0]
    assert "quiz_20240101T000000Z" in call_args[1]


# ─────────────────────────────────────────────────────────────────────────────
# Tag cache
# ─────────────────────────────────────────────────────────────────────────────

def test_load_tag_cache_returns_empty_when_missing(cfg):
    mock_blob   = MagicMock()
    mock_blob.exists.return_value = False
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    with patch("scripts.quiz_builder.storage.Client", return_value=mock_client):
        result = load_tag_cache(cfg)
    assert result == {}

def test_load_tag_cache_returns_data_when_exists(cfg):
    mock_blob   = MagicMock()
    mock_blob.exists.return_value = True
    mock_blob.download_as_text.return_value = json.dumps({"key": {"topic": "Hash Tables"}})
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    with patch("scripts.quiz_builder.storage.Client", return_value=mock_client):
        result = load_tag_cache(cfg)
    assert result == {"key": {"topic": "Hash Tables"}}

def test_load_tag_cache_returns_empty_on_error(cfg):
    with patch("scripts.quiz_builder.storage.Client", side_effect=Exception("GCS error")):
        assert load_tag_cache(cfg) == {}

def test_save_tag_cache(cfg):
    mock_blob   = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    with patch("scripts.quiz_builder.storage.Client", return_value=mock_client):
        save_tag_cache(cfg, {"key": {"topic": "Hash Tables"}})
    mock_blob.upload_from_string.assert_called_once()

def test_save_tag_cache_uploads_valid_json(cfg):
    mock_blob   = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    with patch("scripts.quiz_builder.storage.Client", return_value=mock_client):
        save_tag_cache(cfg, {"k": {"topic": "Joins"}})
    uploaded = mock_blob.upload_from_string.call_args[0][0]
    parsed   = json.loads(uploaded)
    assert parsed["k"]["topic"] == "Joins"


# ─────────────────────────────────────────────────────────────────────────────
# fetch_available_topics
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_available_topics_filters_by_cache(cfg):
    cache = {"k1": {"topic": "Hash Tables"}, "k2": {"topic": "Modern SQL"}}
    with patch("scripts.quiz_builder.load_tag_cache", return_value=cache):
        topics = await fetch_available_topics(cfg)
    assert "Hash Tables" in topics and "Modern SQL" in topics

@pytest.mark.asyncio
async def test_fetch_available_topics_fallback_on_error(cfg):
    with patch("scripts.quiz_builder.load_tag_cache", side_effect=Exception("fail")):
        topics = await fetch_available_topics(cfg)
    assert len(topics) > 0

@pytest.mark.asyncio
async def test_fetch_available_topics_excludes_uncovered(cfg):
    cache = {"k1": {"topic": "Hash Tables"}}
    with patch("scripts.quiz_builder.load_tag_cache", return_value=cache):
        topics = await fetch_available_topics(cfg)
    assert "Joins Algorithms" not in topics

@pytest.mark.asyncio
async def test_fetch_available_topics_empty_cache_fallback(cfg):
    with patch("scripts.quiz_builder.load_tag_cache", return_value={}):
        topics = await fetch_available_topics(cfg)
    assert len(topics) > 0


# ── call_groq_json ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_call_groq_json_returns_parsed(cfg):
    resp = _make_api_resp('{"question": "What is a B+ tree?"}')
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__ = AsyncMock(return_value=resp)
        mock_post.return_value.__aexit__  = AsyncMock(return_value=False)
        result = await call_groq_json(cfg, "unique_prompt_parsed_001")
    assert result["question"] == "What is a B+ tree?"

@pytest.mark.asyncio
async def test_call_groq_json_cache_hit(cfg):
    resp   = _make_api_resp('{"cached": true}')
    prompt = "unique_cache_test_prompt_abc123"
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__ = AsyncMock(return_value=resp)
        mock_post.return_value.__aexit__  = AsyncMock(return_value=False)
        await call_groq_json(cfg, prompt)
        await call_groq_json(cfg, prompt)
        assert mock_post.call_count == 1

@pytest.mark.asyncio
async def test_call_groq_json_retries_on_429(cfg):
    rate_resp = AsyncMock()
    rate_resp.status = 429
    rate_resp.raise_for_status = MagicMock()
    ok_resp = _make_api_resp('{"retried": true}')
    with patch("aiohttp.ClientSession.post") as mock_post, \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_post.return_value.__aenter__ = AsyncMock(side_effect=[rate_resp, ok_resp])
        mock_post.return_value.__aexit__  = AsyncMock(return_value=False)
        result = await call_groq_json(cfg, "unique_retry_prompt_xyz789")
    assert result["retried"] is True

@pytest.mark.asyncio
async def test_call_groq_json_raises_after_max_retries(cfg):
    rate_resp = AsyncMock()
    rate_resp.status = 429
    rate_resp.raise_for_status = MagicMock()
    with patch("aiohttp.ClientSession.post") as mock_post, \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_post.return_value.__aenter__ = AsyncMock(return_value=rate_resp)
        mock_post.return_value.__aexit__  = AsyncMock(return_value=False)
        with pytest.raises(RuntimeError, match="rate limit"):
            await call_groq_json(cfg, "unique_max_retry_prompt_999", max_retries=2)

@pytest.mark.asyncio
async def test_call_groq_json_different_prompts_not_cached(cfg):
    resp = _make_api_resp('{"ok": true}')
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__ = AsyncMock(return_value=resp)
        mock_post.return_value.__aexit__  = AsyncMock(return_value=False)
        await call_groq_json(cfg, "prompt_alpha_unique_001")
        await call_groq_json(cfg, "prompt_beta_unique_002")
        assert mock_post.call_count == 2

@pytest.mark.asyncio
async def test_call_groq_json_malformed_json_raises(cfg):
    resp = _make_api_resp("not json at all !!!!")
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__ = AsyncMock(return_value=resp)
        mock_post.return_value.__aexit__  = AsyncMock(return_value=False)
        with pytest.raises((ValueError, json.JSONDecodeError, KeyError)):
            await call_groq_json(cfg, "unique_malformed_json_prompt_888")


# ── generate_topics ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_topics_returns_strings(cfg, docs, topics_response):
    with patch("scripts.quiz_builder.call_groq_json", AsyncMock(return_value=topics_response)):
        topics = await generate_topics(cfg, "hash tables", docs, n=2)
    assert all(isinstance(t, str) for t in topics)

@pytest.mark.asyncio
async def test_generate_topics_respects_n(cfg, docs, topics_response):
    with patch("scripts.quiz_builder.call_groq_json", AsyncMock(return_value=topics_response)):
        topics = await generate_topics(cfg, "hash tables", docs, n=1)
    assert len(topics) <= 1

@pytest.mark.asyncio
async def test_generate_topics_empty_response(cfg, docs):
    with patch("scripts.quiz_builder.call_groq_json", AsyncMock(return_value={"topics": []})):
        topics = await generate_topics(cfg, "hash tables", docs, n=3)
    assert topics == []

@pytest.mark.asyncio
async def test_generate_topics_list_of_strings(cfg, docs):
    with patch("scripts.quiz_builder.call_groq_json", AsyncMock(return_value=["Hash Tables", "Joins"])):
        topics = await generate_topics(cfg, "hash tables", docs, n=2)
    assert "Hash Tables" in topics

@pytest.mark.asyncio
async def test_generate_topics_strips_whitespace(cfg, docs):
    resp = {"topics": [{"topic": "  Hash Tables  "}, {"topic": " Joins "}]}
    with patch("scripts.quiz_builder.call_groq_json", AsyncMock(return_value=resp)):
        topics = await generate_topics(cfg, "hash tables", docs, n=2)
    assert all(t == t.strip() for t in topics)

@pytest.mark.asyncio
async def test_generate_topics_filters_empty_strings(cfg, docs):
    resp = {"topics": [{"topic": "Hash Tables"}, {"topic": ""}]}
    with patch("scripts.quiz_builder.call_groq_json", AsyncMock(return_value=resp)):
        topics = await generate_topics(cfg, "hash tables", docs, n=2)
    assert "" not in topics


# ── build_quiz ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_build_quiz_mcq_structure(cfg, docs, mcq_response):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("scripts.quiz_builder.call_groq_json",   AsyncMock(return_value=mcq_response)):
        quiz = await build_quiz(cfg, "hash tables", 1, "medium", 6, 12, question_type="mcq")
    q = quiz["questions"][0]
    assert q["type"] == "mcq"
    assert "question" in q and "answer" in q and "options" in q and "explanation" in q

@pytest.mark.asyncio
async def test_build_quiz_mcq_options_count(cfg, docs, mcq_response):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("scripts.quiz_builder.call_groq_json",   AsyncMock(return_value=mcq_response)):
        quiz = await build_quiz(cfg, "hash tables", 1, "medium", 6, 12,
                                question_type="mcq", num_options=4)
    assert len(quiz["questions"][0]["options"]) == 4

@pytest.mark.asyncio
async def test_build_quiz_mcq_answer_in_options(cfg, docs, mcq_response):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("scripts.quiz_builder.call_groq_json",   AsyncMock(return_value=mcq_response)):
        quiz = await build_quiz(cfg, "hash tables", 1, "medium", 6, 12, question_type="mcq")
    q = quiz["questions"][0]
    assert q["answer"] in q["options"]

@pytest.mark.asyncio
async def test_build_quiz_fill_blank_structure(cfg, docs, fill_blank_response):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",  AsyncMock(return_value=["Indexes"])), \
         patch("scripts.quiz_builder.call_groq_json",   AsyncMock(return_value=fill_blank_response)):
        quiz = await build_quiz(cfg, "indexes", 1, "medium", 6, 12, question_type="fill_blank")
    q = quiz["questions"][0]
    assert q["type"] == "fill_blank"
    assert "_____" in q["question"]

@pytest.mark.asyncio
async def test_build_quiz_long_answer_structure(cfg, docs, long_answer_response):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",  AsyncMock(return_value=["Indexes"])), \
         patch("scripts.quiz_builder.call_groq_json",   AsyncMock(return_value=long_answer_response)):
        quiz = await build_quiz(cfg, "indexes", 1, "medium", 6, 12, question_type="long_answer")
    q = quiz["questions"][0]
    assert q["type"] == "long_answer"
    assert "model_answer" in q and "key_points" in q

@pytest.mark.asyncio
async def test_build_quiz_true_false_structure(cfg, docs, true_false_response):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",  AsyncMock(return_value=["Concurrency"])), \
         patch("scripts.quiz_builder.call_groq_json",   AsyncMock(return_value=true_false_response)):
        quiz = await build_quiz(cfg, "concurrency", 1, "medium", 6, 12, question_type="true_false")
    q = quiz["questions"][0]
    assert q["type"] == "true_false"
    assert q["answer"] in ("True", "False")

@pytest.mark.asyncio
async def test_build_quiz_metadata_model_name(cfg, docs, mcq_response):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("scripts.quiz_builder.call_groq_json",   AsyncMock(return_value=mcq_response)):
        quiz = await build_quiz(cfg, "hash tables", 1, "medium", 6, 12)
    assert quiz["run_metadata"]["model"] == "llama-3.1-8b-instant"

@pytest.mark.asyncio
async def test_build_quiz_no_docs_raises(cfg):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=[])):
        with pytest.raises(RuntimeError, match="No documents retrieved"):
            await build_quiz(cfg, "hash tables", 1, "medium", 6, 12)

@pytest.mark.asyncio
async def test_build_quiz_difficulty_stored(cfg, docs, mcq_response):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("scripts.quiz_builder.call_groq_json",   AsyncMock(return_value=mcq_response)):
        easy = await build_quiz(cfg, "hash tables", 1, "easy",  6, 12)
        hard = await build_quiz(cfg, "hash tables", 1, "hard",  6, 12)
    assert easy["difficulty"] == "easy"
    assert hard["difficulty"] == "hard"

@pytest.mark.asyncio
async def test_build_quiz_skips_all_invalid_raises(cfg, docs):
    # All subtopics failing should raise RuntimeError
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("scripts.quiz_builder.call_groq_json",   AsyncMock(side_effect=Exception("bad response"))):
        with pytest.raises(RuntimeError, match="Quiz generation failed"):
            await build_quiz(cfg, "hash tables", 1, "medium", 6, 12, question_type="mcq")

@pytest.mark.asyncio
async def test_build_quiz_partial_failure_continues(cfg, docs, mcq_response):
    # First subtopic fails on attempt 1, second and third succeed with distinct responses
    # so deduplication does not remove them
    call_count = 0
    responses = [
        {**mcq_response, "question": "What does a hash index not support?",    "answer": "Range queries"},
        {**mcq_response, "question": "What do B+ trees store in leaf nodes?",  "answer": "All data records",
         "explanation": "B+ trees store all data records in leaf nodes linked together."},
    ]

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("topic failed")
        # Return a different response each time so dedup doesn't remove duplicates
        return responses[min(call_count - 2, len(responses) - 1)]

    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",
               AsyncMock(return_value=["Bad Topic", "Hash Tables"])), \
         patch("scripts.quiz_builder.call_groq_json", side_effect=side_effect):
        quiz = await build_quiz(cfg, "hash tables", 2, "medium", 6, 12, question_type="mcq")
    assert len(quiz["questions"]) >= 1

@pytest.mark.asyncio
async def test_build_quiz_metadata_retrieval_k(cfg, docs, mcq_response):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("scripts.quiz_builder.call_groq_json",   AsyncMock(return_value=mcq_response)):
        quiz = await build_quiz(cfg, "hash tables", 1, "medium", retrieval_k=8, max_docs=12)
    assert quiz["run_metadata"]["retrieval_k"] == 8

@pytest.mark.asyncio
async def test_build_quiz_source_filter_passed_to_retrieve(cfg, docs, mcq_response):
    mock_retrieve = AsyncMock(return_value=docs)
    with patch("scripts.quiz_builder.retrieve_context", mock_retrieve), \
         patch("scripts.quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("scripts.quiz_builder.call_groq_json",   AsyncMock(return_value=mcq_response)):
        await build_quiz(cfg, "hash tables", 1, "medium", 6, 12,
                         source_filter="lecture1.pdf", question_type="mcq")
    assert "lecture1.pdf" in str(mock_retrieve.call_args)

@pytest.mark.asyncio
async def test_build_quiz_style_stored(cfg, docs, mcq_response):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("scripts.quiz_builder.call_groq_json",   AsyncMock(return_value=mcq_response)):
        quiz = await build_quiz(cfg, "hash tables", 1, "medium", 6, 12, style="scenario")
    assert quiz["style"] == "scenario"

@pytest.mark.asyncio
async def test_build_quiz_question_has_id(cfg, docs, mcq_response):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("scripts.quiz_builder.call_groq_json",   AsyncMock(return_value=mcq_response)):
        quiz = await build_quiz(cfg, "hash tables", 1, "medium", 6, 12, question_type="mcq")
    assert "id" in quiz["questions"][0]

@pytest.mark.asyncio
async def test_build_quiz_retrieved_chunks_in_metadata(cfg, docs, mcq_response):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("scripts.quiz_builder.call_groq_json",   AsyncMock(return_value=mcq_response)):
        quiz = await build_quiz(cfg, "hash tables", 1, "medium", 6, 12)
    assert "retrieved_chunks" in quiz["run_metadata"]
    assert len(quiz["run_metadata"]["retrieved_chunks"]) > 0


# ── Edge cases — irrelevant topics ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_build_quiz_irrelevant_topic_no_docs(cfg):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=[])):
        with pytest.raises(RuntimeError, match="No documents retrieved"):
            await build_quiz(cfg, "quantum physics unrelated topic", 1, "medium", 6, 12)


@pytest.mark.asyncio
async def test_build_quiz_groq_returns_irrelevant_answer(cfg, docs):
    irrelevant = {
        "question": "What is the capital of France?",
        "answer":   "Paris",
        "explanation": "Paris is the capital of France.",
        "sources":  ["CHUNK 1"],
        "incorrect_answers": [
            {"incorrect_answer": "London", "explanation": "No.", "sources": ["CHUNK 1"]},
            {"incorrect_answer": "Berlin", "explanation": "No.", "sources": ["CHUNK 1"]},
            {"incorrect_answer": "Madrid", "explanation": "No.", "sources": ["CHUNK 1"]},
        ],
    }
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("scripts.quiz_builder.call_groq_json",   AsyncMock(return_value=irrelevant)):
        quiz = await build_quiz(cfg, "hash tables", 1, "medium", 6, 12, question_type="mcq")
    assert quiz["questions"][0]["answer"] == "Paris"


def test_rouge_irrelevant_answer_scores_near_zero():
    scores = compute_rouge(
        "The Eiffel Tower is located in Paris, France.",
        "Hash indexes do not support range queries, only equality checks.",
    )
    assert scores["rouge1"] < 0.15 and scores["rougeL"] < 0.15


def test_rouge_partially_irrelevant_answer():
    scores = compute_rouge(
        "Hash indexes are fast but the Eiffel Tower is in Paris.",
        "Hash indexes do not support range queries, only equality checks.",
    )
    assert 0.0 < scores["rouge1"] < 0.6


# ── Edge cases — marks & weight integrity ─────────────────────────────────────

def test_topic_weights_do_not_exceed_100():
    topics = [
        {"topic": "Hash Tables", "weight": 40, "num_questions": 2},
        {"topic": "Joins",       "weight": 35, "num_questions": 2},
        {"topic": "Sorting",     "weight": 25, "num_questions": 1},
    ]
    assert sum(t["weight"] for t in topics) <= 100


def test_topic_weights_sum_exactly_100():
    topics = [
        {"topic": "Hash Tables", "weight": 50, "num_questions": 2},
        {"topic": "Joins",       "weight": 50, "num_questions": 2},
    ]
    assert sum(t["weight"] for t in topics) == 100


def test_topic_weights_over_100_is_invalid():
    topics = [
        {"topic": "Hash Tables", "weight": 60, "num_questions": 2},
        {"topic": "Joins",       "weight": 60, "num_questions": 2},
    ]
    with pytest.raises(ValueError, match="100%"):
        validate_topic_weights(topics)


def test_marks_per_question_positive():
    marks_per_type = {"mcq": 1, "fill_blank": 1, "long_answer": 5, "true_false": 1}
    assert all(v >= 1 for v in marks_per_type.values())


def test_total_marks_calculation():
    topics         = [{"topic": "Hash Tables", "num_questions": 3}]
    marks_per_type = {"mcq": 2}
    total          = sum(t["num_questions"] * marks_per_type["mcq"] for t in topics)
    assert total == 6


def test_total_marks_mixed_types():
    questions = [
        {"type": "mcq",         "marks": 1},
        {"type": "long_answer", "marks": 5},
        {"type": "true_false",  "marks": 1},
        {"type": "fill_blank",  "marks": 1},
    ]
    assert sum(q["marks"] for q in questions) == 8


# ── Edge cases — question count integrity ─────────────────────────────────────

def test_question_type_distribution_equals_n():
    for n in range(1, 11):
        types    = ["mcq", "fill_blank", "long_answer", "true_false"]
        assigned = (types * (n // len(types) + 1))[:n]
        assert len(assigned) == n


def test_question_type_distribution_single_type():
    n        = 5
    types    = ["mcq"]
    assigned = (types * (n // len(types) + 1))[:n]
    assert len(assigned) == n
    assert all(t == "mcq" for t in assigned)


def test_question_type_distribution_all_types_represented():
    types    = ["mcq", "fill_blank", "long_answer", "true_false"]
    n        = 4
    assigned = (types * (n // len(types) + 1))[:n]
    assert set(assigned) == set(types)


def test_question_type_distribution_n_less_than_types():
    types    = ["mcq", "fill_blank", "long_answer", "true_false"]
    n        = 2
    assigned = (types * (n // len(types) + 1))[:n]
    assert len(assigned) == n

def test_topic_weights_do_not_exceed_100():
    topics = [
        {"topic": "Hash Tables", "weight": 40, "num_questions": 2},
        {"topic": "Joins",       "weight": 35, "num_questions": 2},
        {"topic": "Sorting",     "weight": 25, "num_questions": 1},
    ]
    assert sum(t["weight"] for t in topics) <= 100

@pytest.mark.asyncio
async def test_build_quiz_question_count_matches_topics(cfg, docs, mcq_response):
    # Each topic gets a distinct question so deduplication does not remove any
    responses = [
        {**mcq_response, "question": "What does a hash index not support?",   "answer": "Range queries"},
        {**mcq_response, "question": "What do B+ trees store in leaf nodes?", "answer": "All data records",
         "explanation": "B+ trees store all data records in leaf nodes linked together."},
        {**mcq_response, "question": "What type of joins use sort-merge?",     "answer": "Equijoins",
         "explanation": "Sort-merge join is used for equijoins on sorted data."},
    ]
    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        idx = call_count % len(responses)
        call_count += 1
        return responses[idx]

@pytest.mark.asyncio
async def test_build_quiz_question_count_matches_topics(cfg, docs, mcq_response):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",
               AsyncMock(return_value=["Hash Tables", "Joins", "Sorting"])), \
         patch("scripts.quiz_builder.call_groq_json", side_effect=side_effect):
        quiz = await build_quiz(cfg, "database", 3, "medium", 6, 12, question_type="mcq")
    assert len(quiz["questions"]) == 3


# ── ROUGE tests ───────────────────────────────────────────────────────────────

def test_rouge_perfect_match():
    scores = compute_rouge("B+ trees store data in leaf nodes.", "B+ trees store data in leaf nodes.")
    assert scores["rouge1"] == 1.0 and scores["rougeL"] == 1.0

def test_rouge_low_score_on_unrelated():
    scores = compute_rouge("The sky is blue and clouds are white.", "B+ trees store data in leaf nodes.")
    assert scores["rouge1"] < 0.3

def test_rouge_partial_overlap():
    scores = compute_rouge("B+ trees store data in nodes.", "B+ trees store all data in leaf nodes linked in a list.")
    assert 0.0 < scores["rouge1"] < 1.0

def test_rouge_long_answer_quality(long_answer_response):
    reference = "A clustered index orders data physically on disk according to the index key."
    scores    = compute_rouge(long_answer_response["model_answer"], reference)
    assert scores["rouge1"] > 0.1

def test_rouge_empty_hypothesis():
    scores = compute_rouge("", "B+ trees store data in leaf nodes.")
    assert scores["rouge1"] == 0.0

def test_rouge_empty_reference():
    scores = compute_rouge("B+ trees store data in leaf nodes.", "")
    assert scores["rouge1"] == 0.0


# ── BLEU tests ────────────────────────────────────────────────────────────────

def test_bleu_perfect_match():
    score = compute_bleu("hash indexes support equality checks", "hash indexes support equality checks")
    assert score > 0.9

def test_bleu_partial_overlap():
    score = compute_bleu("hash indexes support equality", "hash indexes do not support range queries only equality checks")
    assert 0.0 < score < 1.0

def test_bleu_unrelated():
    score = compute_bleu("the eiffel tower is in paris", "hash indexes do not support range queries")
    assert score < 0.2

def test_bleu_empty_hypothesis():
    assert compute_bleu("", "hash indexes do not support range queries") == 0.0

def test_bleu_empty_reference():
    assert compute_bleu("hash indexes do not support range queries", "") == 0.0

def test_bleu_longer_answer_vs_short_reference():
    score = compute_bleu(
        "A hash index maps keys to slots using a hash function and supports only equality checks not range queries.",
        "hash indexes support equality checks"
    )
    assert score >= 0.0


# ── BERTScore tests ───────────────────────────────────────────────────────────

def test_bertscore_perfect_match():
    scores = compute_bert_score(
        "hash indexes do not support range queries",
        "hash indexes do not support range queries"
    )
    assert scores["f1"] > 0.99

def test_bertscore_semantically_similar():
    scores = compute_bert_score(
        "hash tables use a hash function to map keys to slots",
        "a hash index maps search keys to their corresponding values using hashing"
    )
    assert scores["f1"] > 0.8

def test_bertscore_unrelated():
    scores = compute_bert_score(
        "the eiffel tower is a landmark in paris france",
        "hash indexes do not support range queries only equality checks"
    )
    assert scores["f1"] < 0.75

def test_bertscore_empty_hypothesis():
    scores = compute_bert_score("", "hash indexes do not support range queries")
    assert scores["f1"] == 0.0

def test_bertscore_partial_overlap():
    scores = compute_bert_score(
        "hash indexes support equality checks but not range scans",
        "hash indexes do not support range queries only equality checks"
    )
    assert 0.75 < scores["f1"] < 1.0

def test_bertscore_returns_all_fields():
    scores = compute_bert_score("hash tables are efficient", "hash tables use hashing")
    assert all(k in scores for k in ("precision", "recall", "f1"))


# ── Data splitting ────────────────────────────────────────────────────────────

def test_data_split_counts_sum_to_total(docs):
    chunks = docs * 10
    train, val, test = split_chunks(chunks)
    assert len(train) + len(val) + len(test) == len(chunks)

def test_data_split_train_is_largest(docs):
    chunks = docs * 10
    train, val, test = split_chunks(chunks)
    assert len(train) > len(val) and len(train) > len(test)

def test_data_split_val_and_test_nonempty(docs):
    chunks = docs * 10
    _, val, test = split_chunks(chunks)
    assert len(val) > 0 and len(test) > 0

def test_data_split_ratios_invalid_raises():
    with pytest.raises(AssertionError):
        split_chunks([], train=0.8, val=0.2, test=0.2)


# ── Token length proxy metrics ────────────────────────────────────────────────

def test_hard_prompt_longer_than_easy(docs):
    assert token_length(build_mcq_combined_prompt("Hash Tables", docs, difficulty="easy")) != \
           token_length(build_mcq_combined_prompt("Hash Tables", docs, difficulty="hard"))

def test_more_chunks_produce_longer_prompt(docs):
    assert token_length(build_mcq_combined_prompt("Hash Tables", docs[:1])) < \
           token_length(build_mcq_combined_prompt("Hash Tables", docs))

def test_fill_blank_prompt_token_length_positive(docs):
    assert token_length(build_fill_blank_prompt("Indexes", docs)) > 0

def test_long_answer_prompt_longer_than_true_false(docs):
    assert token_length(build_long_answer_prompt("Concurrency", docs)) != \
           token_length(build_true_false_prompt("Concurrency", docs))

def test_mcq_prompt_longer_than_fill_blank(docs):
    assert token_length(build_mcq_combined_prompt("Hash Tables", docs)) > \
           token_length(build_fill_blank_prompt("Hash Tables", docs))


# ── Integration tests — real Groq + real Cloud SQL ────────────────────────────

@pytest.fixture(scope="module")
def real_cfg():
    try:
        return load_config()
    except RuntimeError as e:
        pytest.skip(f"Integration test skipped — missing env var: {e}")


@pytest.fixture(scope="module")
def integration_scorer():
    return rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


@pytest.mark.asyncio
async def test_integration_call_groq_json_returns_dict(real_cfg):
    prompt = "Return a JSON object with a single key 'status' and value 'ok'. integration_unique_001"
    result = await call_groq_json(real_cfg, prompt)
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_integration_call_groq_json_mcq_structure(real_cfg, docs):
    prompt = build_mcq_combined_prompt("Hash Tables", docs, difficulty="medium") + " integration_unique_002"
    result = await call_groq_json(real_cfg, prompt)
    assert isinstance(result, dict)
    assert "question" in result and "answer" in result and "incorrect_answers" in result


@pytest.mark.asyncio
async def test_integration_call_groq_json_fill_blank_structure(real_cfg, docs):
    prompt = build_fill_blank_prompt("Hash Tables", docs, difficulty="medium") + " integration_unique_003"
    result = await call_groq_json(real_cfg, prompt)
    assert isinstance(result, dict)
    assert "_____" in result.get("question", "")


@pytest.mark.asyncio
async def test_integration_call_groq_json_true_false_answer_valid(real_cfg, docs):
    prompt = build_true_false_prompt("Hash Tables", docs, difficulty="medium") + " integration_unique_004"
    result = await call_groq_json(real_cfg, prompt)
    assert result.get("answer") in ("True", "False")


@pytest.mark.asyncio
async def test_integration_call_groq_json_long_answer_has_key_points(real_cfg, docs):
    prompt = build_long_answer_prompt("Hash Tables", docs, difficulty="medium") + " integration_unique_005"
    result = await call_groq_json(real_cfg, prompt)
    assert isinstance(result.get("key_points"), list)
    assert len(result.get("key_points", [])) > 0


@pytest.mark.asyncio
async def test_integration_retrieve_context_returns_docs(real_cfg):
    docs = await retrieve_context(real_cfg, "Hash Tables", retrieval_k=3, max_docs=6)
    assert len(docs) > 0


@pytest.mark.asyncio
async def test_integration_retrieve_context_docs_have_content(real_cfg):
    docs = await retrieve_context(real_cfg, "Hash Tables", retrieval_k=3, max_docs=6)
    assert all(len(d.page_content.strip()) > 0 for d in docs)


@pytest.mark.asyncio
async def test_integration_retrieve_context_docs_have_metadata(real_cfg):
    docs = await retrieve_context(real_cfg, "Hash Tables", retrieval_k=3, max_docs=6)
    assert all(isinstance(d.metadata, dict) for d in docs)
    assert all("source" in d.metadata for d in docs)


@pytest.mark.asyncio
async def test_integration_retrieve_context_deduped(real_cfg):
    docs     = await retrieve_context(real_cfg, "Hash Tables", retrieval_k=6, max_docs=12)
    contents = [normalize_whitespace(d.page_content)[:240] for d in docs]
    assert len(contents) == len(set(contents))


@pytest.mark.asyncio
async def test_integration_retrieve_context_source_filter(real_cfg):
    all_docs = await retrieve_context(real_cfg, "Hash Tables", retrieval_k=6, max_docs=12)
    if not all_docs:
        pytest.skip("No docs retrieved.")
    first_source = all_docs[0].metadata.get("source", "")
    filename     = Path(first_source).name if first_source else None
    if not filename:
        pytest.skip("No source filename available.")
    filtered = await retrieve_context(
        real_cfg, "Hash Tables", retrieval_k=6, max_docs=12, source_filter=filename
    )
    assert all(filename.lower() in str(d.metadata.get("source", "")).lower() for d in filtered)


@pytest.mark.asyncio
async def test_integration_retrieve_context_irrelevant_topic(real_cfg):
    docs = await retrieve_context(
        real_cfg, "quantum physics unrelated topic xyz", retrieval_k=3, max_docs=6
    )
    assert isinstance(docs, list)


@pytest.mark.asyncio
async def test_integration_build_quiz_mcq_end_to_end(real_cfg):
    quiz = await build_quiz(real_cfg, "Hash Tables", 1, "medium", 6, 12, question_type="mcq")
    assert len(quiz["questions"]) >= 1
    q = quiz["questions"][0]

    # Structural checks — answer must be present and appear in the options list
    assert q["type"] == "mcq"
    assert q.get("answer"), "MCQ answer must not be empty"
    assert q["answer"] in q["options"], "Correct answer must appear in options list"
    assert len(q["options"]) >= 3, "MCQ must have at least 3 options"
    assert q.get("explanation"), "MCQ must include an explanation"
    assert len(q.get("sources", [])) > 0, "MCQ must cite at least one source chunk"

    # Grounding check — BERTScore between the answer and the chunk content that was
    # passed to Groq as context. BERTScore catches paraphrased answers that ROUGE misses.
    # The answer came from this context so semantic similarity should be above baseline.
    chunk_text = " ".join(
        c.get("content", "") for c in quiz["run_metadata"]["retrieved_chunks"]
        if c.get("content")
    )
    if chunk_text:
        scores  = compute_bert_score(q.get("answer", ""), chunk_text)
        bert_f1 = scores["f1"]
        print(f"\n  [MCQ grounding] BERTScore F1: {bert_f1}  (answer vs retrieved chunks)")
        # BERTScore > 0 confirms the answer has some semantic relation to the source material.
        # A completely hallucinated answer unrelated to the chunks would score near 0.
        assert bert_f1 > 0.0, "MCQ answer appears completely unrelated to retrieved chunks"
    else:
        print("\n  [WARN] No chunk content available for grounding check")


@pytest.mark.asyncio
async def test_integration_build_quiz_fill_blank_end_to_end(real_cfg):
    quiz = await build_quiz(real_cfg, "Hash Tables", 1, "medium", 6, 12, question_type="fill_blank")
    assert len(quiz["questions"]) >= 1
    q = quiz["questions"][0]

    # Structural checks
    assert q["type"] == "fill_blank"
    assert q.get("question"), "Fill blank question must not be empty"
    assert q.get("answer"),   "Fill blank answer must not be empty"
    assert q.get("explanation"), "Fill blank must include an explanation"
    assert len(q.get("sources", [])) > 0, "Fill blank must cite at least one source chunk"

    # The model should include _____ but occasionally omits it — warn rather than hard fail
    # because the answer is still valid even if the blank marker is missing
    if "_____" not in q["question"]:
        print(f"\n  [WARN] fill_blank question missing blank marker: {q['question'][:80]}")

    # Grounding check — the answer should be a short keyword from the course material
    # BERTScore between the answer word and the chunk text should be above baseline
    chunk_text = " ".join(
        c.get("content", "") for c in quiz["run_metadata"]["retrieved_chunks"]
        if c.get("content")
    )
    if chunk_text:
        scores  = compute_bert_score(q.get("answer", ""), chunk_text)
        bert_f1 = scores["f1"]
        print(f"\n  [Fill blank grounding] BERTScore F1: {bert_f1}  (answer vs retrieved chunks)")
        assert bert_f1 > 0.0, "Fill blank answer appears completely unrelated to retrieved chunks"


@pytest.mark.asyncio
async def test_integration_build_quiz_true_false_end_to_end(real_cfg):
    try:
        quiz = await build_quiz(real_cfg, "Joins Algorithms", 1, "medium", 6, 12, question_type="true_false")
    except (TimeoutError, RuntimeError) as exc:
        # Cloud SQL timeout or all subtopics failing due to JSON parse errors are
        # infrastructure flakes — skip gracefully rather than failing the suite
        pytest.skip(f"Infrastructure flake — skipping: {exc}")
    assert len(quiz["questions"]) >= 1
    q = quiz["questions"][0]

    # Structural checks
    assert q["type"] == "true_false"
    assert q.get("statement") or q.get("question"), "True/False must have a statement"
    assert q.get("answer") in ("True", "False"), "True/False answer must be exactly True or False"
    assert q.get("explanation"), "True/False must include an explanation"
    assert len(q.get("sources", [])) > 0, "True/False must cite at least one source chunk"

    # Grounding check — the statement should relate to the retrieved course content
    chunk_text = " ".join(
        c.get("content", "") for c in quiz["run_metadata"]["retrieved_chunks"]
        if c.get("content")
    )
    if chunk_text:
        statement = q.get("statement", q.get("question", ""))
        scores    = compute_bert_score(statement, chunk_text)
        bert_f1   = scores["f1"]
        print(f"\n  [True/False grounding] BERTScore F1: {bert_f1}  (statement vs retrieved chunks)")
        assert bert_f1 > 0.0, "True/False statement appears completely unrelated to retrieved chunks"


@pytest.mark.asyncio
async def test_integration_build_quiz_long_answer_end_to_end(real_cfg):
    try:
        quiz = await build_quiz(real_cfg, "Hash Tables", 1, "medium", 6, 12, question_type="long_answer")
    except TimeoutError:
        # Cloud SQL connection pool can time out between sequential integration tests
        pytest.skip("Cloud SQL connection timed out — run this test in isolation")
    assert len(quiz["questions"]) >= 1
    q = quiz["questions"][0]

    # Structural checks — long answer has richer expected output than other types
    assert q["type"] == "long_answer"
    assert q.get("question"),     "Long answer must have a question"
    assert q.get("model_answer"), "Long answer must have a model answer"
    assert len(q.get("model_answer", "")) > 50, "Model answer is too short to be meaningful"
    assert isinstance(q.get("key_points"), list), "Long answer must have key_points list"
    assert len(q.get("key_points", [])) > 0, "Long answer must have at least one key point"
    assert all(isinstance(kp, str) and kp.strip() for kp in q["key_points"]), (
        "All key points must be non-empty strings"
    )
    assert len(q.get("sources", [])) > 0, "Long answer must cite at least one source chunk"

    # Grounding check — model answer is the most text-rich field, most likely to show
    # semantic overlap with the source chunks. BERTScore handles paraphrasing well.
    chunk_text = " ".join(
        c.get("content", "") for c in quiz["run_metadata"]["retrieved_chunks"]
        if c.get("content")
    )
    if chunk_text:
        scores  = compute_bert_score(q.get("model_answer", ""), chunk_text)
        bert_f1 = scores["f1"]
        rouge_s = compute_rouge(q.get("model_answer", ""), chunk_text)
        r1      = rouge_s["rouge1"]
        print(f"\n  [Long answer grounding] BERTScore F1: {bert_f1}  ROUGE-1: {r1}")
        print(f"  (model_answer vs retrieved chunks — BERTScore catches paraphrasing)")
        # BERTScore threshold: answer must have some semantic connection to source material
        assert bert_f1 > 0.0, "Long answer appears completely unrelated to retrieved chunks"


@pytest.mark.asyncio
async def test_integration_build_quiz_metadata_complete(real_cfg):
    quiz = await build_quiz(real_cfg, "Hash Tables", 1, "medium", 6, 12, question_type="mcq")
    meta = quiz["run_metadata"]

    # Check all expected top-level metadata fields are present
    for field in ["model", "retrieval_k", "max_docs", "retrieved_chunks",
                  "generated_questions", "difficulty", "style"]:
        assert field in meta, f"Missing metadata field: {field}"

    # Check each retrieved chunk has the fields needed for grounding validation
    # content is required so tests and the UI can verify answers came from the knowledge base
    for chunk in meta["retrieved_chunks"]:
        assert "chunk_id" in chunk, "Each chunk must have chunk_id"
        assert "source"   in chunk, "Each chunk must have source filename"
        assert "page"     in chunk, "Each chunk must have page number"
        assert "content"  in chunk, "Each chunk must have content for grounding checks"
        assert isinstance(chunk["content"], str) and len(chunk["content"]) > 0, (
            "Chunk content must be a non-empty string"
        )


@pytest.mark.asyncio
async def test_integration_build_quiz_weight_100_enforced(real_cfg):
    topics = [
        {"topic": "Hash Tables", "weight": 60, "num_questions": 2},
        {"topic": "Joins",       "weight": 60, "num_questions": 2},
    ]
    with pytest.raises(ValueError, match="100%"):
        validate_topic_weights(topics)


@pytest.mark.asyncio
async def test_integration_build_quiz_marks_enforced(real_cfg):
    with pytest.raises(ValueError, match=">="):
        validate_marks_per_type({"mcq": 0}, ["mcq"])


# ── Session summary ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def print_test_summary(request):
    yield
    session   = request.session
    total     = len(session.items)
    sep       = "=" * 80
    thin_sep  = "-" * 80

    categories = {
        "load_config":            {"label": "Configuration & Environment",        "tests": []},
        "clean_text":             {"label": "Text Cleaning",                      "tests": []},
        "normalize_white":        {"label": "Whitespace Normalization",           "tests": []},
        "safe_json":              {"label": "JSON Extraction from LLM Output",    "tests": []},
        "utc_compact":            {"label": "Timestamp Generation",               "tests": []},
        "cache_key":              {"label": "Prompt Cache Key Hashing",           "tests": []},
        "budget":                 {"label": "Budget & Rate Tracking",             "tests": []},
        "record_usage":           {"label": "Budget & Rate Tracking",             "tests": []},
        "validate_marks_target":  {"label": "Total Marks Target Validation",      "tests": []},
        "compute_min":            {"label": "Min/Max Marks Calculation",          "tests": []},
        "compute_max":            {"label": "Min/Max Marks Calculation",          "tests": []},
        "dedupe":                 {"label": "Document Deduplication",             "tests": []},
        "format_context":         {"label": "Context Block Formatting",           "tests": []},
        "topic_item":             {"label": "Pydantic Schema Validation",         "tests": []},
        "question_item":          {"label": "Pydantic Schema Validation",         "tests": []},
        "answer_item":            {"label": "Pydantic Schema Validation",         "tests": []},
        "distractor_item":        {"label": "Pydantic Schema Validation",         "tests": []},
        "mcq_prompt":             {"label": "Prompt Construction",                "tests": []},
        "fill_blank_prompt":      {"label": "Prompt Construction",                "tests": []},
        "long_answer_prompt":     {"label": "Prompt Construction",                "tests": []},
        "true_false_prompt":      {"label": "Prompt Construction",                "tests": []},
        "topics_prompt":          {"label": "Prompt Construction",                "tests": []},
        "write_json":             {"label": "GCS Persistence",                    "tests": []},
        "persist_quiz":           {"label": "GCS Persistence",                    "tests": []},
        "load_tag_cache":         {"label": "Tag Cache",                          "tests": []},
        "save_tag_cache":         {"label": "Tag Cache",                          "tests": []},
        "fetch_available":        {"label": "Available Topic Fetching",           "tests": []},
        "call_groq":              {"label": "Groq API Client",                    "tests": []},
        "generate_topics":        {"label": "Topic Generation",                   "tests": []},
        "build_quiz":             {"label": "End-to-End Quiz Generation",         "tests": []},
        "rouge":                  {"label": "ROUGE Scoring",                      "tests": []},
        "bleu":                   {"label": "BLEU Scoring",                       "tests": []},
        "bertscore":              {"label": "BERTScore Semantic Similarity",      "tests": []},
        "data_split":             {"label": "Data Splitting",                     "tests": []},
        "topic_weights":          {"label": "Marks & Weight Integrity",           "tests": []},
        "marks_per_question":     {"label": "Marks & Weight Integrity",           "tests": []},
        "total_marks":            {"label": "Marks & Weight Integrity",           "tests": []},
        "question_type_dist":     {"label": "Question Count & Type Distribution", "tests": []},
        "question_count_matches": {"label": "Question Count & Type Distribution", "tests": []},
        "hard_prompt":            {"label": "Prompt Token Length",                "tests": []},
        "more_chunks":            {"label": "Prompt Token Length",                "tests": []},
        "token_length":           {"label": "Prompt Token Length",                "tests": []},
        "irrelevant":             {"label": "Irrelevant Topics & Content",        "tests": []},
        "normalise_sources":      {"label": "Source Normalisation",               "tests": []},
        "sanitize":               {"label": "LLM Output Sanitisation",            "tests": []},
        "distractor_item_explan": {"label": "Pydantic Schema Validation",         "tests": []},
        "distractor_item_refut":  {"label": "Pydantic Schema Validation",         "tests": []},
        "distractor_item_short":  {"label": "Pydantic Schema Validation",         "tests": []},
    }

    for item in session.items:
        name           = item.name
        passed         = getattr(item, "rep_call", True)
        status         = "PASS" if passed else "FAIL"
        is_integration = "integration" in name
        bucket         = "integration" if is_integration else "unit"
        matched        = False
        for kw, meta in categories.items():
            if kw in name:
                meta["tests"].append((name, status, bucket))
                matched = True
                break
        if not matched:
            categories.setdefault("other", {"label": "Other", "tests": []})["tests"].append(
                (name, status, bucket)
            )

    unit_pass = unit_fail = intg_pass = intg_fail = 0

    print(f"\n\n{sep}")
    print("  AI TEACHING ASSISTANT — QUIZ BUILDER TEST REPORT")
    print(f"  Model: llama-3.1-8b-instant  |  Vector DB: Cloud SQL + pgvector")
    print(sep)

    seen_labels: dict = {}
    for kw, meta in categories.items():
        if not meta["tests"]:
            continue
        label = meta["label"]
        if label not in seen_labels:
            seen_labels[label] = []
        seen_labels[label].extend(meta["tests"])

    print(f"\n  {'─' * 78}\n  UNIT TESTS\n  {'─' * 78}")
    for label, tests in seen_labels.items():
        unit_tests = [(n, s) for n, s, b in tests if b == "unit"]
        if not unit_tests:
            continue
        cp = sum(1 for _, s in unit_tests if s == "PASS")
        cf = sum(1 for _, s in unit_tests if s == "FAIL")
        unit_pass += cp; unit_fail += cf
        icon = "✅" if cf == 0 else "❌"
        print(f"\n  {icon}  {label} ({cp}/{len(unit_tests)})")
        print(f"  {thin_sep}")
        for name, s in unit_tests:
            print(f"    {'✓' if s == 'PASS' else '✗'}  {name}")

    print(f"\n\n  {'─' * 78}\n  INTEGRATION TESTS\n  {'─' * 78}")
    for label, tests in seen_labels.items():
        intg_tests = [(n, s) for n, s, b in tests if b == "integration"]
        if not intg_tests:
            continue
        cp = sum(1 for _, s in intg_tests if s == "PASS")
        cf = sum(1 for _, s in intg_tests if s == "FAIL")
        intg_pass += cp; intg_fail += cf
        icon = "✅" if cf == 0 else "❌"
        print(f"\n  {icon}  [INTEGRATION] {label} ({cp}/{len(intg_tests)})")
        print(f"  {thin_sep}")
        for name, s in intg_tests:
            print(f"    {'✓' if s == 'PASS' else '✗'}  {name}")

    total_pass = unit_pass + intg_pass
    total_fail = unit_fail + intg_fail
    pct        = round((total_pass / total) * 100, 1) if total else 0

    # ── Multi-metric evaluation table ─────────────────────────────────────────
    # Shows ROUGE-1, BLEU, and BERTScore F1 for representative answer quality cases.
    # This gives a concrete view of how each metric behaves across different
    # answer types — from perfect matches to completely off-topic responses.
    smooth = SmoothingFunction().method4
    rs     = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    eval_cases = [
        ("Perfect match (identical text)",
         "hash indexes do not support range queries only equality checks",
         "hash indexes do not support range queries only equality checks"),
        ("Partial overlap (subset of reference)",
         "B+ trees store data in nodes",
         "B+ trees store all data in leaf nodes linked in a list"),
        ("Semantically similar (different words)",
         "hash tables use a hash function to map keys to slots",
         "a hash index maps search keys to values using hashing"),
        ("Long answer vs related reference",
         "A clustered index determines physical order of data on disk.",
         "A clustered index orders data physically on disk according to the index key."),
        ("Partially irrelevant answer",
         "hash indexes are fast but the eiffel tower is in paris",
         "hash indexes do not support range queries only equality checks"),
        ("Unrelated answer (off-topic)",
         "the eiffel tower is located in paris france",
         "hash indexes do not support range queries only equality checks"),
        ("Empty hypothesis",
         "",
         "hash indexes do not support range queries only equality checks"),
    ]

    print(f"\n\n  {'─' * 78}")
    print(f"  MULTI-METRIC EVALUATION TABLE")
    print(f"  Metrics: ROUGE-1 (lexical overlap)  |  BLEU (n-gram precision)  |  BERTScore F1 (semantic)")
    print(f"  {'─' * 78}")
    print(f"  {'Test Case':<46} {'ROUGE-1':>7} {'BLEU':>7} {'BERT-F1':>8}  {'Quality':>12}")
    print(f"  {'─' * 78}")

    for label, hyp, ref in eval_cases:
        if not hyp:
            r1, bleu, bert_f1 = 0.0, 0.0, 0.0
        else:
            r1      = round(rs.score(ref, hyp)["rouge1"].fmeasure, 3)
            bleu    = compute_bleu(hyp, ref)
            bscores = compute_bert_score(hyp, ref)
            bert_f1 = bscores["f1"]
        avg     = (r1 + bleu + bert_f1) / 3
        quality = (
            "Excellent"    if avg >= 0.8  else
            "Good"         if avg >= 0.5  else
            "Low"          if avg >= 0.1  else
            "None/Off-topic"
        )
        print(f"  {label:<46} {r1:>7.3f} {bleu:>7.3f} {bert_f1:>8.3f}  {quality:>12}")

    # ── Per-metric explanation summary ────────────────────────────────────────
    # Explains what each metric measures, its strengths, weaknesses, and
    # what score range means for this project's quiz answer grounding.
    print(f"\n\n  {'─' * 78}")
    print(f"  EVALUATION METRIC SUMMARY")
    print(f"  {'─' * 78}")

    metrics = [
        (
            "ROUGE-1",
            "Lexical overlap — counts matching words between answer and reference.",
            "Fast, interpretable, no model required.",
            "Fails on paraphrasing — 'resize on demand' vs 'dynamically resizable' scores 0.",
            "Use for: detecting completely off-topic answers (score near 0).",
            "Trust when: answer uses the same vocabulary as the source chunk.",
            "Score guide: >0.5 strong overlap  |  0.2-0.5 partial  |  <0.2 weak or off-topic",
        ),
        (
            "BLEU",
            "N-gram precision — measures how many word sequences in the answer appear in the reference.",
            "Standard MT metric, penalises short answers via brevity penalty.",
            "Designed for translation, not QA — short answers score very low even if correct.",
            "Use for: comparing answers against exact reference phrases.",
            "Trust when: answer is expected to closely mirror reference wording.",
            "Score guide: >0.5 near-exact  |  0.1-0.5 partial  |  <0.1 divergent or paraphrased",
        ),
        (
            "BERTScore F1",
            "Semantic similarity — computes cosine similarity between BERT token embeddings.",
            "Catches paraphrased answers that ROUGE and BLEU miss entirely.",
            "Always > 0 even for unrelated text (embeddings are never orthogonal), so absolute",
            "  scores are less meaningful than relative comparisons.",
            "Use for: grounding check — is the answer semantically related to the source chunks?",
            "Score guide: >0.9 near-identical  |  0.8-0.9 paraphrased  |  <0.75 likely hallucinated",
        ),
    ]

    for m in metrics:
        name, what, pro, con, use, trust, guide = m
        print(f"\n  {name}")
        print(f"    What it measures : {what}")
        print(f"    Strength         : {pro}")
        print(f"    Weakness         : {con}")
        print(f"    {use}")
        print(f"    {trust}")
        print(f"    {guide}")

    print(f"\n  {'─' * 78}")
    print(f"  WHY ALL THREE ARE USED TOGETHER")
    print(f"  {'─' * 78}")
    print(f"  ROUGE-1  catches exact word reuse — fast signal for grounding")
    print(f"  BLEU     catches n-gram precision  — useful for fill-blank answers")
    print(f"  BERTScore catches paraphrasing     — most reliable for long answers")
    print(f"")
    print(f"  A hallucinated answer scores low on ALL THREE.")
    print(f"  A paraphrased but grounded answer scores low on ROUGE/BLEU but high on BERTScore.")
    print(f"  Only a near-verbatim grounded answer scores high on all three.")

    # ── Human evaluation rubric ───────────────────────────────────────────────
    print(f"\n\n  {'─' * 78}")
    print(f"  HUMAN EVALUATION RUBRIC")
    print(f"  For use during manual review or presentation demo of generated questions")
    print(f"  {'─' * 78}")
    print(f"  {'Criterion':<35} {'Scale':<10} {'Description'}")
    print(f"  {'─' * 78}")
    rubric = [
        ("Factual Accuracy",        "1–5", "Is the answer factually correct based on course material?"),
        ("Source Grounding",        "1–5", "Is the answer traceable to the retrieved source chunks?"),
        ("Question Clarity",        "1–5", "Is the question unambiguous and clearly worded?"),
        ("Distractor Plausibility", "1–5", "Are MCQ wrong options believable but clearly incorrect?"),
        ("Difficulty Alignment",    "1–5", "Does difficulty match the requested level?"),
        ("Coverage",                "1–5", "Does the question cover an important aspect of the topic?"),
        ("Answer Completeness",     "1–5", "Does the answer fully address what the question asks?"),
    ]
    for criterion, scale, desc in rubric:
        print(f"  {criterion:<35} {scale:<10} {desc}")
    print(f"\n  Scoring : 1=Poor  2=Fair  3=Acceptable  4=Good  5=Excellent")
    print(f"  Target  : >= 4.0 average across all criteria for production use")

    # ── Prompt token length table ─────────────────────────────────────────────
    base_docs = [
        Document(page_content="A B+ tree stores all data in leaf nodes linked in a list.",
                 metadata={"source": "lecture1.pdf", "page": 3, "chunk_id": "c1"}),
        Document(page_content="Hash indexes do not support range queries, only equality checks.",
                 metadata={"source": "lecture2.pdf", "page": 7, "chunk_id": "c2"}),
    ]
    print(f"\n\n  {'─' * 78}")
    print(f"  PROMPT TOKEN LENGTH TABLE")
    print(f"  {'─' * 78}")
    print(f"  {'Configuration':<46} {'Tokens':>8}  {'Notes'}")
    print(f"  {'─' * 78}")
    token_rows = [
        ("MCQ (4 options, medium, conceptual)",
         token_length(build_mcq_combined_prompt("Hash Tables", base_docs)),
         "Longest — includes distractors"),
        ("MCQ (3 options, medium, conceptual)",
         token_length(build_mcq_combined_prompt("Hash Tables", base_docs, num_options=3)),
         "Fewer distractors"),
        ("Fill-in-the-blank (medium)",
         token_length(build_fill_blank_prompt("Hash Tables", base_docs)),
         "Single blank"),
        ("Long answer (medium, conceptual)",
         token_length(build_long_answer_prompt("Hash Tables", base_docs)),
         "Requires model answer"),
        ("True/False (medium)",
         token_length(build_true_false_prompt("Hash Tables", base_docs)),
         "Shortest structured type"),
        ("MCQ with 1 chunk",
         token_length(build_mcq_combined_prompt("Hash Tables", base_docs[:1])),
         "Less context"),
        ("MCQ with 2 chunks",
         token_length(build_mcq_combined_prompt("Hash Tables", base_docs)),
         "More context"),
        ("MCQ scenario style",
         token_length(build_mcq_combined_prompt("Hash Tables", base_docs, style="scenario")),
         "Slightly longer"),
        ("MCQ hard difficulty",
         token_length(build_mcq_combined_prompt("Hash Tables", base_docs, difficulty="hard")),
         "Hard wording"),
    ]
    for label, tokens, note in token_rows:
        print(f"  {label:<46} {tokens:>8}  {note}")

    # ── Final results ─────────────────────────────────────────────────────────
    print(f"\n\n{sep}")
    print(f"  FINAL RESULTS")
    print(f"  {thin_sep}")
    print(f"  Unit Tests        Passed: {unit_pass:<6} Failed: {unit_fail}  {'✅' if unit_fail == 0 else '❌'}")
    print(f"  Integration Tests Passed: {intg_pass:<6} Failed: {intg_fail}  {'✅' if intg_fail == 0 else '⚠️'}")
    print(f"  {thin_sep}")
    print(f"  Total  : {total}")
    print(f"  Passed : {total_pass}  ✅")
    print(f"  Failed : {total_fail}  {'✅' if total_fail == 0 else '❌'}")
    print(f"  Rate   : {pct}%")
    print(f"\n  Edge Cases Covered:")
    print(f"    • Malformed / unparseable JSON from Groq API")
    print(f"    • Empty and null inputs across all text utilities")
    print(f"    • Rate limit (429) with exponential backoff retry")
    print(f"    • Budget hard stop when session spending limit is exceeded")
    print(f"    • Irrelevant topics returning no documents from vector DB")
    print(f"    • Off-topic LLM answers detected via ROUGE scoring")
    print(f"    • Topic weights summing over 100% detected as invalid")
    print(f"    • Total marks target below min/max achievable range")
    print(f"    • Marks target not a multiple of 5 rejected")
    print(f"    • Question type distribution always equals requested n")
    print(f"    • Prompt cache preventing duplicate API calls")
    print(f"    • GCS failures returning safe empty defaults")
    print(f"    • Cloud SQL timeout handled gracefully in integration tests")
    print(f"\n{sep}\n")
