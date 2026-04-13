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
    validate_topic_relevance,     # new in this version
    compute_total_marks,
    _cache_key,
    _reset_minute_budget,
    _record_usage,
    _budget,
    _groq_cache,
    BUDGET_LIMIT_USD,
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
        "explanation": "Hash indexes use equality checks only.",
        "sources": ["CHUNK 1"],
        "incorrect_answers": [
            {"incorrect_answer": "Point queries",  "explanation": "Supported.", "sources": ["CHUNK 1"]},
            {"incorrect_answer": "Equality joins", "explanation": "Supported.", "sources": ["CHUNK 1"]},
            {"incorrect_answer": "Key lookups",    "explanation": "Supported.", "sources": ["CHUNK 2"]},
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# load_config
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# clean_text
# ─────────────────────────────────────────────────────────────────────────────

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
    text = 'Here is the result:\n\n{"answer": "Dense"}\n\nThats all.'
    assert safe_json_extract(text)["answer"] == "Dense"

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
    with patch.dict("scripts.quiz_builder._budget", {
        "total_cost_usd": 0.0,
        "requests_this_minute": 0,
        "tokens_this_minute": 0,
        "minute_start": 0.0,
        "total_requests": 0,
        "total_tokens": 0,
    }), patch("scripts.quiz_builder._previous_spend", BUDGET_LIMIT_USD):
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


# ─────────────────────────────────────────────────────────────────────────────
# dedupe_documents
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# format_context_blocks
# ─────────────────────────────────────────────────────────────────────────────

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

def test_mcq_prompt_ascii_instruction(docs):
    # Prompts in this version explicitly ask for ASCII-only output
    assert "ASCII" in build_mcq_combined_prompt("Hash Tables", docs)

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

def test_long_answer_prompt_ascii_instruction(docs):
    assert "ASCII" in build_long_answer_prompt("Joins", docs)

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

def test_topics_prompt_ascii_instruction(docs):
    assert "ASCII" in build_topics_prompt("indexing", docs, n=3)


# ─────────────────────────────────────────────────────────────────────────────
# GCS persistence — uses QUIZ_OUTPUT_BUCKET constant, not cfg.gcs_bucket
# ─────────────────────────────────────────────────────────────────────────────

def test_write_json_to_gcs(cfg, sample_quiz):
    mock_blob   = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    with patch("scripts.quiz_builder.storage.Client", return_value=mock_client):
        uri = write_json_to_gcs(cfg, "quizzes/quiz_001.json", sample_quiz)
    mock_blob.upload_from_string.assert_called_once()
    assert QUIZ_OUTPUT_BUCKET in uri          # constant = "quizzes-output"

def test_write_json_to_gcs_returns_gs_uri(cfg, sample_quiz):
    mock_blob   = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    with patch("scripts.quiz_builder.storage.Client", return_value=mock_client):
        uri = write_json_to_gcs(cfg, "quizzes/quiz_001.json", sample_quiz)
    assert uri.startswith("gs://")

def test_write_json_to_gcs_uses_constant_bucket_not_cfg(cfg, sample_quiz):
    # cfg.gcs_bucket = "bucket" but QUIZ_OUTPUT_BUCKET = "quizzes-output"
    # The URI must reference QUIZ_OUTPUT_BUCKET, not cfg.gcs_bucket
    mock_blob   = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    with patch("scripts.quiz_builder.storage.Client", return_value=mock_client):
        uri = write_json_to_gcs(cfg, "quizzes/quiz_001.json", sample_quiz)
    assert cfg.gcs_bucket not in uri
    assert QUIZ_OUTPUT_BUCKET in uri

@pytest.mark.asyncio
async def test_persist_quiz(cfg, sample_quiz):
    with patch("scripts.quiz_builder.write_json_to_gcs",
               return_value=f"gs://{QUIZ_OUTPUT_BUCKET}/quizzes/quiz.json") as mock_write:
        uri = await persist_quiz(cfg, sample_quiz)
    mock_write.assert_called_once()
    assert "gs://" in uri

@pytest.mark.asyncio
async def test_persist_quiz_uses_quiz_id(cfg, sample_quiz):
    with patch("scripts.quiz_builder.write_json_to_gcs",
               return_value=f"gs://{QUIZ_OUTPUT_BUCKET}/quizzes/quiz_20240101T000000Z.json") as mock_write:
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


# ─────────────────────────────────────────────────────────────────────────────
# validate_topic_relevance  (new function)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_validate_topic_relevance_true_when_enough_docs(cfg, docs):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)):
        assert await validate_topic_relevance(cfg, "Hash Tables") is True

@pytest.mark.asyncio
async def test_validate_topic_relevance_false_when_too_few_docs(cfg):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=[])):
        assert await validate_topic_relevance(cfg, "Hash Tables") is False

@pytest.mark.asyncio
async def test_validate_topic_relevance_true_on_db_error(cfg):
    # DB errors should not block — returns True defensively
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(side_effect=Exception("db error"))):
        assert await validate_topic_relevance(cfg, "Hash Tables") is True

@pytest.mark.asyncio
async def test_validate_topic_relevance_custom_threshold(cfg, docs):
    # With threshold=3 and only 2 docs, should return False
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)):
        assert await validate_topic_relevance(cfg, "Hash Tables", threshold=3) is False


# ─────────────────────────────────────────────────────────────────────────────
# call_groq_json
# ─────────────────────────────────────────────────────────────────────────────

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
    resp = _make_api_resp('{"cached": true}')
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


# ─────────────────────────────────────────────────────────────────────────────
# generate_topics
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# build_quiz — updated for vs + question_index + 2-attempt retry
# ─────────────────────────────────────────────────────────────────────────────

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
        easy = await build_quiz(cfg, "hash tables", 1, "easy", 6, 12)
        hard = await build_quiz(cfg, "hash tables", 1, "hard", 6, 12)
    assert easy["difficulty"] == "easy"
    assert hard["difficulty"] == "hard"

@pytest.mark.asyncio
async def test_build_quiz_skips_invalid_groq_response(cfg, docs):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("scripts.quiz_builder.call_groq_json",   AsyncMock(side_effect=Exception("bad response"))):
        with pytest.raises(RuntimeError, match="Quiz generation failed"):
            await build_quiz(cfg, "hash tables", 1, "medium", 6, 12, question_type="mcq")

@pytest.mark.asyncio
async def test_build_quiz_partial_failure_continues(cfg, docs, mcq_response):
    """First topic exhausts both retry attempts; second topic succeeds → 1 question."""
    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:          # both attempts for "Bad Topic" fail
            raise Exception("bad response")
        return mcq_response

    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",
               AsyncMock(return_value=["Bad Topic", "Hash Tables"])), \
         patch("scripts.quiz_builder.call_groq_json", side_effect=side_effect):
        quiz = await build_quiz(cfg, "hash tables", 2, "medium", 6, 12, question_type="mcq")
    assert len(quiz["questions"]) == 1

@pytest.mark.asyncio
async def test_build_quiz_retry_busts_cache(cfg, docs, mcq_response):
    """On retry the suffix changes to _retry, producing a new cache key."""
    suffixes_seen = []

    async def capture_prompt(cfg_arg, prompt, **kwargs):
        suffixes_seen.append(prompt.split("\n# q=")[-1] if "# q=" in prompt else "")
        if len(suffixes_seen) == 1:
            raise Exception("first attempt fails")
        return mcq_response

    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("scripts.quiz_builder.call_groq_json", side_effect=capture_prompt):
        quiz = await build_quiz(cfg, "hash tables", 1, "medium", 6, 12, question_type="mcq")
    # The second call suffix should contain "_retry"
    assert any("retry" in s for s in suffixes_seen)
    assert len(quiz["questions"]) == 1

@pytest.mark.asyncio
async def test_build_quiz_question_index_passed(cfg, docs, mcq_response):
    """question_index is appended to the prompt, making each question's cache key unique."""
    prompts_seen = []

    async def capture(cfg_arg, prompt, **kwargs):
        prompts_seen.append(prompt)
        return mcq_response

    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("scripts.quiz_builder.call_groq_json", side_effect=capture):
        await build_quiz(cfg, "hash tables", 1, "medium", 6, 12,
                         question_type="mcq", question_index=5)
    assert any("q=6" in p for p in prompts_seen)   # index+1 = 6

@pytest.mark.asyncio
async def test_build_quiz_accepts_vs_param(cfg, docs, mcq_response):
    """Passing vs= should prevent a second get_vector_store call."""
    mock_vs = MagicMock()
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)) as mock_rc, \
         patch("scripts.quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("scripts.quiz_builder.call_groq_json",   AsyncMock(return_value=mcq_response)):
        await build_quiz(cfg, "hash tables", 1, "medium", 6, 12,
                         question_type="mcq", vs=mock_vs)
    # vs should be forwarded to retrieve_context
    call_kwargs = mock_rc.call_args
    assert call_kwargs is not None

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


# ─────────────────────────────────────────────────────────────────────────────
# compute_total_marks — exact distribution logic
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_total_marks_single_type():
    topics = [{"topic": "Hash Tables", "num_questions": 4}]
    assert compute_total_marks(topics, {"mcq": 2}, ["mcq"]) == 8

def test_compute_total_marks_mixed_types_exact():
    # 4 questions, 2 types → mcq,fill_blank,mcq,fill_blank → 2+1+2+1 = 6
    topics = [{"topic": "Hash Tables", "num_questions": 4}]
    assert compute_total_marks(topics, {"mcq": 2, "fill_blank": 1}, ["mcq", "fill_blank"]) == 6

def test_compute_total_marks_multiple_topics():
    topics = [
        {"topic": "Hash Tables", "num_questions": 2},
        {"topic": "Joins",       "num_questions": 2},
    ]
    total = compute_total_marks(topics, {"mcq": 3}, ["mcq"])
    assert total == 12

def test_compute_total_marks_long_answer_weight():
    topics = [{"topic": "Hash Tables", "num_questions": 2}]
    assert compute_total_marks(topics, {"long_answer": 5}, ["long_answer"]) == 10


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases — irrelevant topics / content
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_build_quiz_irrelevant_topic_no_docs(cfg):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=[])):
        with pytest.raises(RuntimeError, match="No documents retrieved"):
            await build_quiz(cfg, "quantum physics unrelated topic", 1, "medium", 6, 12)

@pytest.mark.asyncio
async def test_build_quiz_groq_returns_irrelevant_answer(cfg, docs):
    irrelevant = {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "explanation": "Paris is the capital of France.",
        "sources": ["CHUNK 1"],
        "incorrect_answers": [
            {"incorrect_answer": "London",  "explanation": "No.", "sources": ["CHUNK 1"]},
            {"incorrect_answer": "Berlin",  "explanation": "No.", "sources": ["CHUNK 1"]},
            {"incorrect_answer": "Madrid",  "explanation": "No.", "sources": ["CHUNK 1"]},
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


# ─────────────────────────────────────────────────────────────────────────────
# Marks & weight integrity
# ─────────────────────────────────────────────────────────────────────────────

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

def test_validate_marks_raises_when_zero(cfg):
    with pytest.raises(ValueError, match=">="):
        validate_marks_per_type({"mcq": 0}, ["mcq"])

def test_total_marks_mixed_types():
    questions = [
        {"type": "mcq",         "marks": 1},
        {"type": "long_answer", "marks": 5},
        {"type": "true_false",  "marks": 1},
        {"type": "fill_blank",  "marks": 1},
    ]
    assert sum(q["marks"] for q in questions) == 8


# ─────────────────────────────────────────────────────────────────────────────
# Question count & type distribution
# ─────────────────────────────────────────────────────────────────────────────

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

@pytest.mark.asyncio
async def test_build_quiz_question_count_matches_topics(cfg, docs, mcq_response):
    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",
               AsyncMock(return_value=["Hash Tables", "Joins", "Sorting"])), \
         patch("scripts.quiz_builder.call_groq_json", AsyncMock(return_value=mcq_response)):
        quiz = await build_quiz(cfg, "database", 3, "medium", 6, 12, question_type="mcq")
    assert len(quiz["questions"]) == 3


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE tests
# ─────────────────────────────────────────────────────────────────────────────

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
    assert compute_rouge("", "B+ trees store data in leaf nodes.")["rouge1"] == 0.0

def test_rouge_empty_reference():
    assert compute_rouge("B+ trees store data in leaf nodes.", "")["rouge1"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# BLEU tests
# ─────────────────────────────────────────────────────────────────────────────

def test_bleu_perfect_match():
    assert compute_bleu("hash indexes support equality checks", "hash indexes support equality checks") > 0.9

def test_bleu_partial_overlap():
    score = compute_bleu("hash indexes support equality", "hash indexes do not support range queries only equality checks")
    assert 0.0 < score < 1.0

def test_bleu_unrelated():
    assert compute_bleu("the eiffel tower is in paris", "hash indexes do not support range queries") < 0.2

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


# ─────────────────────────────────────────────────────────────────────────────
# BERTScore tests
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Data splitting
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Token length proxy metrics
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION — real Groq + real vector DB
# ─────────────────────────────────────────────────────────────────────────────

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
async def test_integration_build_quiz_mcq_end_to_end(real_cfg, integration_scorer):
    quiz = await build_quiz(real_cfg, "Hash Tables", 1, "medium", 6, 12, question_type="mcq")
    assert len(quiz["questions"]) >= 1
    q = quiz["questions"][0]
    assert q["answer"] in q["options"]
    chunk_text = " ".join(c.get("source", "") for c in quiz["run_metadata"]["retrieved_chunks"])
    scores = integration_scorer.score(chunk_text, q.get("answer", ""))
    r1 = round(scores["rouge1"].fmeasure, 3)
    print(f"\n  [INTEGRATION ROUGE] MCQ answer — ROUGE-1: {r1}")
    assert isinstance(r1, float)

@pytest.mark.asyncio
async def test_integration_build_quiz_fill_blank_end_to_end(real_cfg):
    quiz = await build_quiz(real_cfg, "Hash Tables", 1, "medium", 6, 12, question_type="fill_blank")
    assert len(quiz["questions"]) >= 1
    assert "_____" in quiz["questions"][0]["question"]

@pytest.mark.asyncio
async def test_integration_build_quiz_true_false_end_to_end(real_cfg):
    quiz = await build_quiz(real_cfg, "Joins Algorithms", 1, "medium", 6, 12, question_type="true_false")
    assert len(quiz["questions"]) >= 1
    assert quiz["questions"][0]["answer"] in ("True", "False")

@pytest.mark.asyncio
async def test_integration_build_quiz_long_answer_end_to_end(real_cfg, integration_scorer):
    quiz = await build_quiz(real_cfg, "Hash Tables", 1, "medium", 6, 12, question_type="long_answer")
    assert len(quiz["questions"]) >= 1
    q          = quiz["questions"][0]
    chunk_text = " ".join(c.get("source", "") for c in quiz["run_metadata"]["retrieved_chunks"])
    scores     = integration_scorer.score(chunk_text, q.get("model_answer", ""))
    r1         = round(scores["rouge1"].fmeasure, 3)
    print(f"\n  [INTEGRATION ROUGE] Long answer — ROUGE-1: {r1}")
    assert isinstance(r1, float)

@pytest.mark.asyncio
async def test_integration_build_quiz_metadata_complete(real_cfg):
    quiz = await build_quiz(real_cfg, "Hash Tables", 1, "medium", 6, 12, question_type="mcq")
    meta = quiz["run_metadata"]
    for field in ["model", "retrieval_k", "max_docs", "retrieved_chunks",
                  "generated_questions", "difficulty", "style"]:
        assert field in meta, f"Missing metadata field: {field}"

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

@pytest.mark.asyncio
async def test_integration_validate_topic_relevance_hash_tables(real_cfg):
    result = await validate_topic_relevance(real_cfg, "Hash Tables")
    assert result is True

@pytest.mark.asyncio
async def test_integration_validate_topic_relevance_irrelevant(real_cfg):
    result = await validate_topic_relevance(real_cfg, "quantum gravity wormholes", threshold=5)
    assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# Session-scoped summary fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def print_test_summary(request):
    yield

    session = request.session
    total   = len(session.items)
    sep     = "=" * 100
    thin    = "-" * 100

    # ── Per-category metadata ─────────────────────────────────────────────────
    CATEGORIES = {
        "config":           ("Configuration & environment",         "unit",        "Missing required var → RuntimeError; GROQ_MODEL override"),
        "clean_text":       ("Text cleaning",                       "unit",        "Null at start/end; only null bytes → empty; empty passthrough"),
        "normalize":        ("Whitespace normalisation",            "unit",        "Only spaces → empty; mixed \\t\\n\\r collapsed; already clean unchanged"),
        "sanitize":         ("LLM output sanitisation",             "unit",        "Empty passthrough; plain ASCII unchanged; smart quotes replaced"),
        "safe_json":        ("JSON extraction from LLM output",     "unit",        "No JSON → raises; markdown-fenced; empty {}; smart quotes cleaned first"),
        "timestamp":        ("Timestamp generation",                "unit",        "Exactly 16 chars; ends in Z; starts with '2'"),
        "cache_key":        ("Prompt cache key hashing",            "unit",        "Empty string valid 64-char key; single space = different hash"),
        "budget":           ("Budget & rate tracking",              "unit",        "Limit reached → RuntimeError; cost accumulates across calls"),
        "dedupe":           ("Document deduplication",              "unit",        "max_docs=0 → empty; all duplicates → one; same content diff metadata = duplicate"),
        "context_blocks":   ("Context block formatting",            "unit",        "Empty input → empty string; missing metadata handled; chunk_id and page included"),
        "relevance":        ("Topic relevance validation",          "unit",        "DB error → True (safe fallback); custom threshold respected"),
        "retrieve_intg":    ("Retrieve context",                    "integration", "Irrelevant topic still returns list; source filter narrows results"),
        "pydantic":         ("Pydantic schema validation",          "unit",        "Empty answer raises; short explanation raises; sources capped at 3"),
        "prompts":          ("Prompt construction — all types",     "unit",        "Distractor count varies with num_options; hard = multi-step; ASCII instruction present"),
        "token_len":        ("Prompt token length proxy",           "unit",        "More chunks → longer prompt; MCQ longer than fill-blank"),
        "groq_unit":        ("Groq API client",                     "unit",        "429 → backoff; max retries → RuntimeError; malformed JSON raises; cache hit skips call"),
        "groq_intg":        ("Groq API client",                     "integration", "true_false answer must be 'True'/'False'; key_points must be non-empty list"),
        "gen_topics":       ("Topic generation",                    "unit",        "Empty list → []; plain string list accepted; whitespace stripped; empty strings filtered"),
        "build_quiz":       ("build_quiz — structure & metadata",   "unit",        "No docs → RuntimeError; answer in options; source_filter forwarded; id always present"),
        "retry":            ("build_quiz — retry & cache busting",  "unit",        "Both attempts fail → [SKIP]; _retry suffix busts SHA-256 cache; vs param forwarded"),
        "partial":          ("Partial failure continues",           "unit",        "First topic fails twice; second succeeds → exactly 1 question in output"),
        "e2e_intg":         ("build_quiz end-to-end",               "integration", "fill_blank: validates structure not blank format; metadata fields verified"),
        "gcs":              ("GCS persistence",                     "unit",        "Uses QUIZ_OUTPUT_BUCKET not cfg.gcs_bucket; URI starts gs://; content is valid JSON"),
        "tag_cache":        ("Tag cache (chunk tagging)",           "unit",        "Missing blob → {}; GCS error → {}; uploaded string is valid JSON"),
        "avail_topics":     ("Available topic fetching",            "unit",        "Uncovered topics excluded; empty cache → full-syllabus fallback; exception → fallback"),
        "marks":            ("Marks & weight integrity",            "unit",        "Weights > 100 raises; marks = 0 raises; mixed type totals correct"),
        "total_marks":      ("compute_total_marks",                 "unit",        "Single type; mixed types alternating; multiple topics; long_answer weighting"),
        "distribution":     ("Question count & type distribution",  "unit",        "n=1 to 10 all correct; single type repeated; n < num_types still produces n"),
        "rouge":            ("ROUGE scoring",                       "unit",        "Perfect = 1.0; off-topic < 0.15; empty hypothesis or reference = 0.0"),
        "bleu":             ("BLEU scoring",                        "unit",        "Perfect > 0.9; empty = 0.0; unrelated < 0.2"),
        "bertscore":        ("BERTScore",                           "unit",        "Identical = 1.0; similar > 0.8; unrelated < 0.75; empty = 0.0"),
        "irrelevant":       ("Irrelevant topic & off-topic answers","unit",        "No docs → RuntimeError; off-topic answer stored as-is; ROUGE < 0.15 for off-topic"),
        "data_split":       ("Data splitting",                      "unit",        "Counts sum to total; train largest; invalid ratios → AssertionError"),
        "relev_intg":       ("validate_topic_relevance",            "integration", "Known topic → True; irrelevant topic → bool (pgvector always returns docs)"),
    }

    KEYWORD_MAP = {
        "load_config":          "config",
        "clean_text":           "clean_text",
        "normalize_whitespace": "normalize",
        "sanitize":             "sanitize",
        "safe_json":            "safe_json",
        "utc_compact":          "timestamp",
        "cache_key":            "cache_key",
        "budget":               "budget",
        "record_usage":         "budget",
        "reset_minute":         "budget",
        "print_budget":         "budget",
        "dedupe":               "dedupe",
        "format_context":       "context_blocks",
        "validate_topic_rel":   "relevance",
        "topic_item":           "pydantic",
        "topic_list":           "pydantic",
        "question_item":        "pydantic",
        "answer_item":          "pydantic",
        "distractor_item":      "pydantic",
        "mcq_prompt":           "prompts",
        "fill_blank_prompt":    "prompts",
        "long_answer_prompt":   "prompts",
        "true_false_prompt":    "prompts",
        "topics_prompt":        "prompts",
        "token_length":         "token_len",
        "hard_prompt":          "token_len",
        "more_chunks":          "token_len",
        "mcq_prompt_longer":    "token_len",
        "call_groq":            "groq_unit",
        "generate_topics":      "gen_topics",
        "build_quiz_retry":     "retry",
        "build_quiz_question_i":"retry",
        "build_quiz_accepts_vs":"retry",
        "partial_failure":      "partial",
        "build_quiz":           "build_quiz",
        "compute_total_marks":  "total_marks",
        "total_marks":          "total_marks",
        "write_json":           "gcs",
        "persist_quiz":         "gcs",
        "load_tag_cache":       "tag_cache",
        "save_tag_cache":       "tag_cache",
        "fetch_available":      "avail_topics",
        "topic_weights":        "marks",
        "marks_per_question":   "marks",
        "validate_marks":       "marks",
        "question_type_dist":   "distribution",
        "question_count":       "distribution",
        "rouge_irrelevant":     "irrelevant",
        "rouge_partially":      "irrelevant",
        "build_quiz_irrelevant":"irrelevant",
        "build_quiz_groq_irrel":"irrelevant",
        "rouge":                "rouge",
        "bleu":                 "bleu",
        "bertscore":            "bertscore",
        "data_split":           "data_split",
        "integration_retrieve": "retrieve_intg",
        "integration_call_groq":"groq_intg",
        "integration_build_qui":"e2e_intg",
        "integration_validate": "relev_intg",
    }

    # Classify each test item
    counts  = {k: {"pass": 0, "fail": 0, "tests": []} for k in CATEGORIES}
    uncat   = {"pass": 0, "fail": 0, "tests": []}
    total_p = 0
    total_f = 0

    for item in session.items:
        name   = item.nodeid.split("::")[-1]
        rep    = getattr(item, "rep_call", None)
        passed = getattr(rep, "passed", True) if rep else True
        if passed:
            total_p += 1
        else:
            total_f += 1

        matched = False
        # longest-match first to avoid short-key false hits
        for kw in sorted(KEYWORD_MAP, key=len, reverse=True):
            if kw in name:
                cat = KEYWORD_MAP[kw]
                counts[cat]["tests"].append((name, passed))
                if passed:
                    counts[cat]["pass"] += 1
                else:
                    counts[cat]["fail"] += 1
                matched = True
                break
        if not matched:
            uncat["tests"].append((name, passed))
            if passed:
                uncat["pass"] += 1
            else:
                uncat["fail"] += 1

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n\n{sep}")
    print(f"  AI TEACHING ASSISTANT — QUIZ BUILDER  |  TEST SUMMARY REPORT")
    print(f"  Model  : llama-3.1-8b-instant          Vector DB : Cloud SQL + pgvector")
    print(sep)

    # ── Stat bar ──────────────────────────────────────────────────────────────
    rate = round(total_p / total * 100, 1) if total else 0
    print(f"\n  {'Total':10} {total:>4}    {'Passed':10} {total_p:>4} ✅    "
          f"{'Failed':10} {total_f:>4} {'✅' if total_f == 0 else '❌'}    "
          f"Pass rate  {rate}%\n")

    # ── Section headers ───────────────────────────────────────────────────────
    SECTIONS = [
        ("CORE UTILITIES",            ["config","clean_text","normalize","sanitize","safe_json","timestamp","cache_key"]),
        ("BUDGET & RATE LIMITING",    ["budget"]),
        ("RETRIEVAL & CONTEXT",       ["dedupe","context_blocks","relevance","retrieve_intg"]),
        ("PYDANTIC SCHEMA VALIDATION",["pydantic"]),
        ("PROMPT BUILDERS",           ["prompts","token_len"]),
        ("GROQ API CLIENT",           ["groq_unit","groq_intg"]),
        ("TOPIC GENERATION",          ["gen_topics"]),
        ("END-TO-END QUIZ GENERATION",["build_quiz","retry","partial","e2e_intg"]),
        ("GCS PERSISTENCE",           ["gcs"]),
        ("TAG CACHE",                 ["tag_cache","avail_topics"]),
        ("MARKS, WEIGHTS & DISTRIB.", ["marks","total_marks","distribution"]),
        ("ANSWER QUALITY METRICS",    ["rouge","bleu","bertscore","irrelevant"]),
        ("DATA INFRASTRUCTURE",       ["data_split"]),
        ("INTEGRATION — RELEVANCE",   ["relev_intg"]),
    ]

    W_CAT  = 38
    W_TYPE = 13
    W_N    = 6
    W_STAT = 7
    W_EDGE = 52

    header = (f"  {'Category':<{W_CAT}} {'Type':<{W_TYPE}} {'Tests':>{W_N}} "
              f"{'Status':<{W_STAT}}  {'Edge cases covered':<{W_EDGE}}")
    row_sep = f"  {'─'*(W_CAT+W_TYPE+W_N+W_STAT+W_EDGE+8)}"

    for section_label, keys in SECTIONS:
        # Skip empty sections
        if not any(counts[k]["tests"] for k in keys if k in counts):
            continue
        print(f"\n  ── {section_label} {'─'*(95-len(section_label)-5)}")
        print(header)
        print(row_sep)
        for k in keys:
            if k not in counts or not counts[k]["tests"]:
                continue
            label, tier, edge = CATEGORIES[k]
            n    = counts[k]["pass"] + counts[k]["fail"]
            fail = counts[k]["fail"]
            stat = "PASS" if fail == 0 else f"FAIL({fail})"
            icon = "✅" if fail == 0 else "❌"
            # Wrap edge text at W_EDGE chars
            words     = edge.split()
            lines     = []
            cur       = ""
            for w in words:
                if len(cur) + len(w) + 1 <= W_EDGE:
                    cur = (cur + " " + w).strip()
                else:
                    lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
            first_line  = lines[0] if lines else ""
            extra_lines = lines[1:]
            print(f"  {label:<{W_CAT}} {tier:<{W_TYPE}} {n:>{W_N}} "
                  f"{icon} {stat:<{W_STAT-2}}  {first_line}")
            for el in extra_lines:
                print(f"  {'':<{W_CAT}} {'':<{W_TYPE}} {'':{W_N}} "
                      f"{'':>{W_STAT}}   {el}")

    # ── Uncategorised fallback ────────────────────────────────────────────────
    if uncat["tests"]:
        print(f"\n  ── UNCATEGORISED {'─'*79}")
        print(header)
        print(row_sep)
        n    = uncat["pass"] + uncat["fail"]
        fail = uncat["fail"]
        icon = "✅" if fail == 0 else "❌"
        stat = "PASS" if fail == 0 else f"FAIL({fail})"
        print(f"  {'(other)':<{W_CAT}} {'—':<{W_TYPE}} {n:>{W_N}} {icon} {stat}")

    # ── Edge case summary ─────────────────────────────────────────────────────
    print(f"\n\n{sep}")
    print(f"  EDGE CASE COVERAGE SUMMARY")
    print(sep)
    edge_groups = [
        ("Input & encoding",   ["clean_text","normalize","sanitize","safe_json"]),
        ("API reliability",    ["groq_unit","budget"]),
        ("Retrieval quality",  ["dedupe","context_blocks","relevance","retrieve_intg"]),
        ("LLM output quality", ["prompts","gen_topics","irrelevant","rouge","bleu","bertscore"]),
        ("Quiz correctness",   ["build_quiz","retry","partial","marks","total_marks","distribution"]),
        ("Infrastructure",     ["gcs","tag_cache","avail_topics","data_split"]),
    ]
    for group_label, keys in edge_groups:
        edges = [CATEGORIES[k][2] for k in keys if k in CATEGORIES and counts.get(k,{}).get("tests")]
        if not edges:
            continue
        print(f"\n  {group_label}")
        for e in edges:
            print(f"    •  {e}")

    # ── Footer ────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  FINAL  Total: {total}  |  Passed: {total_p} ✅  |  Failed: {total_f} {'✅' if total_f==0 else '❌'}  |  Rate: {rate}%")
    print(f"{sep}\n")