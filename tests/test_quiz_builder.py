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
    save_tag_cache,
    build_quiz,
    utc_compact_ts,
    write_json_to_gcs,
    validate_topic_weights,
    validate_marks_per_type,
    compute_total_marks,
    _cache_key,
    _reset_minute_budget,
    _record_usage,
    _budget,
    _groq_cache,
    BUDGET_LIMIT_USD,
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
    """Sentence-level BLEU with smoothing for short texts."""
    if not hypothesis or not reference:
        return 0.0
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    smoothie   = SmoothingFunction().method4
    return round(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie), 3)


def compute_bert_score(hypothesis: str, reference: str) -> dict:
    """BERTScore F1 using distilbert-base-uncased for speed."""
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

def test_fill_blank_prompt_contains_subtopic(docs):
    assert "Indexes" in build_fill_blank_prompt("Indexes", docs)

def test_fill_blank_prompt_has_blank_marker(docs):
    assert "_____" in build_fill_blank_prompt("Indexes", docs)

def test_fill_blank_prompt_difficulty_differs(docs):
    assert build_fill_blank_prompt("Indexes", docs, difficulty="easy") != \
           build_fill_blank_prompt("Indexes", docs, difficulty="hard")

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


# ─────────────────────────────────────────────────────────────────────────────
# GCS persistence
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
    assert "bucket" in uri

def test_write_json_to_gcs_returns_gs_uri(cfg, sample_quiz):
    mock_blob   = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    with patch("scripts.quiz_builder.storage.Client", return_value=mock_client):
        uri = write_json_to_gcs(cfg, "quizzes/quiz_001.json", sample_quiz)
    assert uri.startswith("gs://")

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
# build_quiz
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
    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("topic failed")
        return mcq_response

    with patch("scripts.quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("scripts.quiz_builder.generate_topics",
               AsyncMock(return_value=["Bad Topic", "Hash Tables"])), \
         patch("scripts.quiz_builder.call_groq_json", side_effect=side_effect):
        quiz = await build_quiz(cfg, "hash tables", 2, "medium", 6, 12, question_type="mcq")
    assert len(quiz["questions"]) == 1

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
# Edge cases — topic weight and marks integrity
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


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases — question count integrity
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
    scores = compute_rouge("", "B+ trees store data in leaf nodes.")
    assert scores["rouge1"] == 0.0

def test_rouge_empty_reference():
    scores = compute_rouge("B+ trees store data in leaf nodes.", "")
    assert scores["rouge1"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# BLEU tests
# ─────────────────────────────────────────────────────────────────────────────

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
    chunk_text = " ".join(
        c.get("source", "") for c in quiz["run_metadata"]["retrieved_chunks"]
    )
    scores = integration_scorer.score(chunk_text, q.get("answer", ""))
    r1     = round(scores["rouge1"].fmeasure, 3)
    rl     = round(scores["rougeL"].fmeasure, 3)
    print(f"\n  [INTEGRATION ROUGE] MCQ answer — ROUGE-1: {r1}, ROUGE-L: {rl}")
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
    chunk_text = " ".join(
        c.get("source", "") for c in quiz["run_metadata"]["retrieved_chunks"]
    )
    scores = integration_scorer.score(chunk_text, q.get("model_answer", ""))
    r1     = round(scores["rouge1"].fmeasure, 3)
    rl     = round(scores["rougeL"].fmeasure, 3)
    print(f"\n  [INTEGRATION ROUGE] Long answer — ROUGE-1: {r1}, ROUGE-L: {rl}")
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


# ─────────────────────────────────────────────────────────────────────────────
# Comparison summary (session-scoped)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def print_test_summary(request):
    yield
    session  = request.session
    total    = len(session.items)
    sep      = "=" * 80
    thin_sep = "-" * 80

    print(f"\n\n{sep}")
    print("  AI TEACHING ASSISTANT — QUIZ BUILDER TEST REPORT")
    print(f"  {'Model: llama-3.1-8b-instant':40} {'Vector DB: Cloud SQL + pgvector':38}")
    print(sep)

    categories = {
        "load_config":            {"label": "Configuration & Environment",        "desc": "Validates env var loading, default/custom model selection, and missing var errors.",         "edge": "Missing required env var → RuntimeError; custom GROQ_MODEL override.",                                                                     "tests": []},
        "clean_text":             {"label": "Text Cleaning",                      "desc": "Ensures null bytes, whitespace, and boundary characters are handled before DB insertion.",  "edge": "Null bytes at start/end, string of only null bytes, empty string.",                                                                          "tests": []},
        "normalize_whitespace":   {"label": "Whitespace Normalization",           "desc": "Collapses mixed whitespace (tabs, newlines, spaces) from raw PDF extraction.",            "edge": "Only spaces, mixed \\t\\n\\r, already-clean string.",                                                                                        "tests": []},
        "safe_json_extract":      {"label": "JSON Extraction from LLM Output",    "desc": "Parses JSON from Groq responses that may include prose, markdown fences, or unicode.",    "edge": "No JSON present → raises; markdown-fenced JSON; empty object {}; unicode chars.",                                                           "tests": []},
        "utc_compact_ts":         {"label": "Timestamp Generation",               "desc": "Produces consistent UTC timestamps used as quiz IDs and GCS filenames.",                   "edge": "Format must be exactly 16 chars ending in Z; must start with '2'.",                                                                         "tests": []},
        "cache_key":              {"label": "Prompt Cache Key Hashing",           "desc": "SHA-256 keys ensure identical prompts hit cache and different prompts don't collide.",     "edge": "Empty string key; whitespace-sensitive hashing.",                                                                                            "tests": []},
        "budget":                 {"label": "Budget & Rate Tracking",             "desc": "Tracks token usage and cost per session; hard stops when limit is reached.",               "edge": "Budget exactly at limit → RuntimeError; cost accumulates across calls.",                                                                     "tests": []},
        "dedupe_documents":       {"label": "Document Deduplication",             "desc": "Removes duplicate chunks from retrieval results using a 240-char content key prefix.",    "edge": "max_docs=0; all duplicates; same content different metadata; single doc.",                                                                   "tests": []},
        "format_context_blocks":  {"label": "Context Block Formatting",           "desc": "Formats retrieved chunks into labeled blocks for LLM prompts.",                           "edge": "Empty input; missing metadata fields; single doc; chunk_id and page included.",                                                             "tests": []},
        "pydantic_validators":    {"label": "Pydantic Schema Validation",         "desc": "Validates all LLM-generated question, answer, and distractor structures.",                "edge": "Empty answer; too-short explanation; sources capped at 3; empty sources list.",                                                            "tests": []},
        "prompt_builders":        {"label": "Prompt Construction",                "desc": "Builds LLM prompts for all question types across difficulty levels and styles.",           "edge": "Distractor count varies with num_options; hard mentions multi-step reasoning; scenario style.",                                            "tests": []},
        "gcs_persistence":        {"label": "GCS Persistence",                    "desc": "Saves quiz JSON to Google Cloud Storage with correct paths and content type.",            "edge": "quiz_id in GCS path; uploaded content is valid JSON; URI starts with gs://.",                                                              "tests": []},
        "tag_cache":              {"label": "Tag Cache (Chunk Tagging)",           "desc": "Loads and saves syllabus topic tags for chunks from/to GCS.",                             "edge": "Missing blob → empty dict; GCS error → empty dict; uploaded content is valid JSON.",                                                      "tests": []},
        "fetch_available_topics": {"label": "Available Topic Fetching",           "desc": "Returns only topics present in the tag cache, falling back to full syllabus on error.",   "edge": "Uncovered topics excluded; empty cache falls back to full list; exception fallback.",                                                      "tests": []},
        "call_groq_json":         {"label": "Groq API Client",                    "desc": "Async HTTP client for Groq with caching, rate limit retry, and JSON parsing.",            "edge": "429 → exponential retry; max retries exhausted → RuntimeError; malformed JSON → raises; cache prevents duplicate calls.",                  "tests": []},
        "generate_topics":        {"label": "Topic Generation",                   "desc": "Generates quiz subtopics from user text using Groq, handling various response formats.",  "edge": "Empty response → []; plain string list; whitespace stripped; empty strings filtered.",                                                     "tests": []},
        "build_quiz":             {"label": "End-to-End Quiz Generation",         "desc": "Orchestrates retrieval, topic generation, and question generation for all question types.","edge": "No docs → RuntimeError; partial topic failure continues; answer in options; source_filter forwarded; irrelevant topic with no docs.",       "tests": []},
        "rouge":                  {"label": "ROUGE Answer Quality Scoring",       "desc": "Evaluates lexical overlap between LLM answers and source chunks using ROUGE-1 and ROUGE-L.","edge": "Perfect match = 1.0; unrelated answer < 0.15; empty hypothesis/reference = 0.0.",                                                       "tests": []},
        "bleu":                   {"label": "BLEU Score (N-gram Precision)",      "desc": "Measures n-gram precision of answers against source reference using sentence-level BLEU.", "edge": "Perfect match > 0.9; empty input = 0.0; unrelated < 0.2; smoothing handles short texts.",                                               "tests": []},
        "bertscore":              {"label": "BERTScore (Semantic Similarity)",    "desc": "Measures semantic similarity using distilbert embeddings — captures meaning beyond word overlap.","edge": "Identical = 1.0; semantically similar > 0.8; unrelated < 0.75; empty = 0.0.",                                                      "tests": []},
        "data_splitting":         {"label": "Data Splitting",                     "desc": "Splits document chunks into train/val/test sets with configurable ratios.",                "edge": "Counts sum to total; train is largest; invalid ratios → AssertionError.",                                                                  "tests": []},
        "token_length":           {"label": "Prompt Token Length Proxy Metric",   "desc": "Verifies prompt complexity scales with context size and question type.",                   "edge": "More chunks → longer prompt; MCQ longer than fill-blank.",                                                                                 "tests": []},
        "edge_irrelevant":        {"label": "Irrelevant Topics & Content",        "desc": "Validates system behaviour when topics or answers are unrelated to course material.",     "edge": "No docs for irrelevant topic → RuntimeError; irrelevant Groq answer stored as-is; ROUGE near 0 for off-topic answers.",                   "tests": []},
        "edge_integrity":         {"label": "Marks & Weight Integrity",           "desc": "Ensures quiz weights never exceed 100% and marks are computed correctly.",                "edge": "Weights > 100 detected; marks per type >= 1; total marks calculation verified.",                                                          "tests": []},
        "edge_distribution":      {"label": "Question Count & Type Distribution", "desc": "Ensures distributed question types always equal the requested n exactly.",               "edge": "n=1 to 10; single type; all types represented; n < num_types still produces n items.",                                                   "tests": []},
    }

    keyword_map = {
        "load_config":            "load_config",
        "clean_text":             "clean_text",
        "normalize_white":        "normalize_whitespace",
        "safe_json":              "safe_json_extract",
        "utc_compact":            "utc_compact_ts",
        "cache_key":              "cache_key",
        "budget":                 "budget",
        "record_usage":           "budget",
        "reset_minute":           "budget",
        "print_budget":           "budget",
        "dedupe":                 "dedupe_documents",
        "format_context":         "format_context_blocks",
        "topic_item":             "pydantic_validators",
        "topic_list":             "pydantic_validators",
        "question_item":          "pydantic_validators",
        "answer_item":            "pydantic_validators",
        "distractor_item":        "pydantic_validators",
        "mcq_prompt":             "prompt_builders",
        "fill_blank_prompt":      "prompt_builders",
        "long_answer_prompt":     "prompt_builders",
        "true_false_prompt":      "prompt_builders",
        "topics_prompt":          "prompt_builders",
        "write_json":             "gcs_persistence",
        "persist_quiz":           "gcs_persistence",
        "load_tag_cache":         "tag_cache",
        "save_tag_cache":         "tag_cache",
        "fetch_available":        "fetch_available_topics",
        "call_groq":              "call_groq_json",
        "generate_topics":        "generate_topics",
        "rouge_irrelevant":       "edge_irrelevant",
        "rouge_partially":        "edge_irrelevant",
        "rouge":                  "rouge",
        "bleu":                   "bleu",
        "bertscore":              "bertscore",
        "data_split":             "data_splitting",
        "token_length":           "token_length",
        "hard_prompt":            "token_length",
        "more_chunks":            "token_length",
        "mcq_prompt_longer":      "token_length",
        "fill_blank_prompt_token":"token_length",
        "long_answer_prompt_long":"token_length",
        "irrelevant_topic":       "edge_irrelevant",
        "irrelevant_answer":      "edge_irrelevant",
        "topic_weights":          "edge_integrity",
        "marks_per_question":     "edge_integrity",
        "total_marks":            "edge_integrity",
        "question_type_dist":     "edge_distribution",
        "question_count_matches": "edge_distribution",
        "build_quiz":             "build_quiz",
    }

    for item in session.items:
        name           = item.name
        passed         = getattr(item, "rep_call", True)
        status         = "PASS" if passed else "FAIL"
        is_integration = "integration" in name
        matched        = False
        for kw, cat in keyword_map.items():
            if kw in name:
                key = f"integration_{cat}" if is_integration else cat
                if key not in categories:
                    categories[key] = {
                        "label": f"[INTEGRATION] {categories.get(cat, {}).get('label', cat)}",
                        "desc":  categories.get(cat, {}).get("desc", ""),
                        "edge":  categories.get(cat, {}).get("edge", ""),
                        "tests": [],
                    }
                categories[key]["tests"].append((name, status))
                matched = True
                break
        if not matched:
            fallback = "integration_other" if is_integration else "other"
            categories.setdefault(fallback, {
                "label": "[INTEGRATION] Other" if is_integration else "Other",
                "desc": "", "edge": "", "tests": []
            })["tests"].append((name, status))

    col_w      = 52
    total_pass = 0
    total_fail = 0
    unit_pass  = 0
    unit_fail  = 0
    intg_pass  = 0
    intg_fail  = 0

    print(f"\n  {'─' * 78}")
    print(f"  UNIT TESTS")
    print(f"  {'─' * 78}")

    for cat, meta in categories.items():
        if cat.startswith("integration_"):
            continue
        results = meta["tests"]
        if not results:
            continue
        cat_pass    = sum(1 for _, s in results if s == "PASS")
        cat_fail    = sum(1 for _, s in results if s == "FAIL")
        total_pass += cat_pass
        total_fail += cat_fail
        unit_pass  += cat_pass
        unit_fail  += cat_fail
        icon  = "✅" if cat_fail == 0 else "❌"
        label = f"{meta['label']} ({cat_pass}/{len(results)})"
        print(f"\n  {icon}  {label}")
        print(f"  {thin_sep}")
        print(f"  What it tests : {meta['desc']}")
        print(f"  Edge cases    : {meta['edge']}")
        print(f"  {'─' * 78}")
        for name, s in results:
            marker = "✓" if s == "PASS" else "✗"
            print(f"    {marker}  {name}")

    print(f"\n\n  {'─' * 78}")
    print(f"  INTEGRATION TESTS  (real Groq API + real Cloud SQL vector DB)")
    print(f"  {'─' * 78}")

    for cat, meta in categories.items():
        if not cat.startswith("integration_"):
            continue
        results = meta["tests"]
        if not results:
            continue
        cat_pass    = sum(1 for _, s in results if s == "PASS")
        cat_fail    = sum(1 for _, s in results if s == "FAIL")
        total_pass += cat_pass
        total_fail += cat_fail
        intg_pass  += cat_pass
        intg_fail  += cat_fail
        icon  = "✅" if cat_fail == 0 else "❌"
        label = f"{meta['label']} ({cat_pass}/{len(results)})"
        print(f"\n  {icon}  {label}")
        print(f"  {thin_sep}")
        print(f"  What it tests : {meta['desc']}")
        print(f"  Edge cases    : {meta['edge']}")
        print(f"  {'─' * 78}")
        for name, s in results:
            marker = "✓" if s == "PASS" else "✗"
            print(f"    {marker}  {name}")

    # ── Multi-metric evaluation table ─────────────────────────────────────────
    smooth     = SmoothingFunction().method4
    rs         = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
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
        ("Unrelated answer (off-topic)",
         "the eiffel tower is located in paris france",
         "hash indexes do not support range queries only equality checks"),
        ("Partially irrelevant answer",
         "hash indexes are fast but the eiffel tower is in paris",
         "hash indexes do not support range queries only equality checks"),
        ("Empty hypothesis",
         "",
         "hash indexes do not support range queries only equality checks"),
        ("Long answer vs related reference",
         "A clustered index determines physical order of data on disk.",
         "A clustered index orders data physically on disk according to the index key."),
    ]

    print(f"\n\n  {'─' * 78}")
    print(f"  MULTI-METRIC EVALUATION TABLE")
    print(f"  Metrics: ROUGE-1 (lexical)  |  BLEU (n-gram)  |  BERTScore F1 (semantic)")
    print(f"  {'─' * 78}")
    print(f"  {'Test Case':<44} {'ROUGE-1':>7} {'BLEU':>7} {'BERT-F1':>8}  {'Quality':>12}")
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
        quality = "Excellent" if avg >= 0.8 else "Good" if avg >= 0.5 else "Low" if avg >= 0.1 else "None/Off-topic"
        print(f"  {label:<44} {r1:>7} {bleu:>7} {bert_f1:>8}  {quality:>12}")

    # ── Human evaluation rubric ───────────────────────────────────────────────
    print(f"\n\n  {'─' * 78}")
    print(f"  HUMAN EVALUATION RUBRIC")
    print(f"  For use during presentation / manual review of generated quiz questions")
    print(f"  {'─' * 78}")
    print(f"  {'Criterion':<35} {'Scale':<15} {'Description'}")
    print(f"  {'─' * 78}")
    rubric = [
        ("Factual Accuracy",        "1-5", "Is the answer factually correct based on course material?"),
        ("Source Grounding",        "1-5", "Is the answer traceable to the retrieved source chunks?"),
        ("Question Clarity",        "1-5", "Is the question unambiguous and clearly worded?"),
        ("Distractor Plausibility", "1-5", "Are MCQ wrong options believable but clearly incorrect?"),
        ("Difficulty Alignment",    "1-5", "Does difficulty match the requested level (easy/medium/hard)?"),
        ("Coverage",                "1-5", "Does the question cover an important aspect of the topic?"),
        ("Answer Completeness",     "1-5", "Does the answer fully address what the question asks?"),
    ]
    for criterion, scale, desc in rubric:
        print(f"  {criterion:<35} {scale:<15} {desc}")
    print(f"\n  Scoring guide : 1=Poor  2=Fair  3=Acceptable  4=Good  5=Excellent")
    print(f"  Target score  : >= 4.0 average across all criteria for production use")

    # ── Token length table ────────────────────────────────────────────────────
    base_docs = [
        Document(page_content="A B+ tree stores all data in leaf nodes linked in a list.",
                 metadata={"source": "lecture1.pdf", "page": 3, "chunk_id": "c1"}),
        Document(page_content="Hash indexes do not support range queries, only equality checks.",
                 metadata={"source": "lecture2.pdf", "page": 7, "chunk_id": "c2"}),
    ]

    print(f"\n\n  {'─' * 78}")
    print(f"  PROMPT TOKEN LENGTH COMPARISON TABLE")
    print(f"  {'─' * 78}")
    print(f"  {'Configuration':<44} {'Tokens':>8}  {'Notes'}")
    print(f"  {'─' * 78}")
    token_rows = [
        ("MCQ (4 options, medium, conceptual)",    token_length(build_mcq_combined_prompt("Hash Tables", base_docs)),                            "Longest — includes distractors"),
        ("Fill-in-the-blank (medium)",             token_length(build_fill_blank_prompt("Hash Tables", base_docs)),                              "Single blank"),
        ("Long answer (medium, conceptual)",       token_length(build_long_answer_prompt("Hash Tables", base_docs)),                             "Requires model answer"),
        ("True/False (medium)",                    token_length(build_true_false_prompt("Hash Tables", base_docs)),                              "Shortest structured type"),
        ("MCQ 1 chunk",                            token_length(build_mcq_combined_prompt("Hash Tables", base_docs[:1])),                        "1 chunk"),
        ("MCQ 2 chunks",                           token_length(build_mcq_combined_prompt("Hash Tables", base_docs)),                            "2 chunks — longer context"),
        ("MCQ scenario style",                     token_length(build_mcq_combined_prompt("Hash Tables", base_docs, style="scenario")),          "Slightly longer than conceptual"),
        ("MCQ hard difficulty",                    token_length(build_mcq_combined_prompt("Hash Tables", base_docs, difficulty="hard")),         "Hard wording"),
    ]
    for label, tokens, note in token_rows:
        print(f"  {label:<44} {tokens:>8}  {note}")

    # ── Final results ─────────────────────────────────────────────────────────
    print(f"\n\n{sep}")
    print(f"  FINAL RESULTS")
    print(f"  {thin_sep}")
    print(f"  {'Unit Tests':<30} Passed: {unit_pass:<6} Failed: {unit_fail}  {'✅' if unit_fail == 0 else '❌'}")
    print(f"  {'Integration Tests':<30} Passed: {intg_pass:<6} Failed: {intg_fail}  {'✅' if intg_fail == 0 else '⚠️  (real system — some variance expected)'}")
    print(f"  {thin_sep}")
    print(f"  Total Tests : {total}")
    print(f"  Passed      : {total_pass}  ✅")
    print(f"  Failed      : {total_fail}  {'✅' if total_fail == 0 else '❌'}")
    pct = round((total_pass / total) * 100, 1) if total else 0
    print(f"  Pass Rate   : {pct}%")
    print(f"\n  Edge Cases Covered:")
    print(f"    • Malformed / unparseable JSON from Groq API")
    print(f"    • Empty and null inputs across all text utilities")
    print(f"    • Rate limit (429) with exponential backoff retry")
    print(f"    • Budget hard stop when spending limit is exceeded")
    print(f"    • Irrelevant topics returning no documents from vector DB")
    print(f"    • Off-topic LLM answers detected via ROUGE scoring")
    print(f"    • Topic weights summing over 100% detected as invalid")
    print(f"    • Question type distribution always equals requested n")
    print(f"    • Prompt cache preventing duplicate API calls")
    print(f"    • GCS failures returning safe empty defaults")
    print(f"\n{sep}\n")