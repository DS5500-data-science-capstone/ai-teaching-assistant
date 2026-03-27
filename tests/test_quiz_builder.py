from __future__ import annotations

import json
import math
import random
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
from pydantic import ValidationError
from langchain_core.documents import Document
from rouge_score import rouge_scorer

from quiz_builder import (
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
    safe_json_extract,
    save_tag_cache,
    build_quiz,
    utc_compact_ts,
    write_json_to_gcs,
    _cache_key,
    _reset_minute_budget,
    _record_usage,
    _budget,
    BUDGET_LIMIT_USD,
)


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


def token_length(text: str) -> int:
    return len(text.split())


# ── load_config ───────────────────────────────────────────────────────────────

# load_config: must raise RuntimeError when a required env var is missing
def test_load_config_missing_env_raises():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="Missing required env var"):
            load_config()

# load_config: must return AppConfig when all required env vars are present
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


# ── clean_text ────────────────────────────────────────────────────────────────

# clean_text: null bytes must be removed from text before embedding
def test_clean_text_removes_null_bytes():
    assert clean_text("hello\x00world") == "helloworld"

# clean_text: leading and trailing whitespace must be stripped
def test_clean_text_strips_whitespace():
    assert clean_text("  padded  ") == "padded"

# clean_text: empty string must return empty string without error
def test_clean_text_empty():
    assert clean_text("") == ""

# clean_text: text with no issues must pass through unchanged
def test_clean_text_no_change():
    assert clean_text("normal text") == "normal text"

# clean_text: multiple null bytes scattered in text must all be removed
def test_clean_text_multiple_null_bytes():
    assert clean_text("a\x00b\x00c") == "abc"


# ── normalize_whitespace ──────────────────────────────────────────────────────

# normalize_whitespace: collapses multiple spaces into one for consistent tokenization
def test_normalize_whitespace_collapses_spaces():
    assert normalize_whitespace("a   b   c") == "a b c"

# normalize_whitespace: handles mixed newline and tab characters from raw PDF extraction
def test_normalize_whitespace_newlines_and_tabs():
    assert normalize_whitespace("a\n\tb\r\nc") == "a b c"

# normalize_whitespace: already-clean text should pass through unchanged
def test_normalize_whitespace_already_clean():
    assert normalize_whitespace("already clean") == "already clean"


# ── safe_json_extract ─────────────────────────────────────────────────────────

# safe_json_extract: extracts a JSON object embedded in surrounding text
def test_safe_json_extract_object():
    assert safe_json_extract('text {"key": "val"} end') == {"key": "val"}

# safe_json_extract: when input is a JSON array, the regex matches the inner object first
def test_safe_json_extract_array():
    assert safe_json_extract('[{"topic": "Hash"}]') == {"topic": "Hash"}

# safe_json_extract: correctly parses nested JSON objects
def test_safe_json_extract_nested():
    assert safe_json_extract('{"a": {"b": 1}}') == {"a": {"b": 1}}

# safe_json_extract: raises when no valid JSON is present in the string
def test_safe_json_extract_no_json_raises():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        safe_json_extract("no json")


# ── utc_compact_ts ────────────────────────────────────────────────────────────

# utc_compact_ts: output must be exactly 16 chars and end with Z
def test_utc_compact_ts_format():
    ts = utc_compact_ts()
    assert len(ts) == 16
    assert ts.endswith("Z")


# ── _cache_key ────────────────────────────────────────────────────────────────

# _cache_key: same input must always produce the same hash
def test_cache_key_deterministic():
    assert _cache_key("hello") == _cache_key("hello")

# _cache_key: different inputs must produce different hashes
def test_cache_key_different_inputs():
    assert _cache_key("hello") != _cache_key("world")


# ── budget tracking ───────────────────────────────────────────────────────────

# _reset_minute_budget: must run without raising any errors
def test_reset_minute_budget_runs_without_error():
    _reset_minute_budget()

# _record_usage: token counts must increase by at least prompt + completion tokens
def test_record_usage_updates_totals():
    before = _budget["total_tokens"]
    _record_usage("llama-3.1-8b-instant", 100, 50)
    assert _budget["total_tokens"] >= before + 150

# _record_usage: must raise RuntimeError when total cost exceeds the budget limit
def test_record_usage_raises_on_budget_exceeded():
    with patch.dict("quiz_builder._budget", {"total_cost_usd": BUDGET_LIMIT_USD,
                                              "requests_this_minute": 0,
                                              "tokens_this_minute": 0,
                                              "minute_start": 0.0,
                                              "total_requests": 0,
                                              "total_tokens": 0}):
        with pytest.raises(RuntimeError, match="Hard stop"):
            _record_usage("llama-3.1-8b-instant", 1, 1)

# print_budget_summary: must print without raising any errors
def test_print_budget_summary_runs(capsys):
    print_budget_summary()
    captured = capsys.readouterr()
    assert "Session total" in captured.out


# ── dedupe_documents ──────────────────────────────────────────────────────────

# dedupe_documents: duplicate documents sharing the same content prefix must be removed
def test_dedupe_removes_duplicates(docs):
    result = dedupe_documents(docs + docs, max_docs=10)
    assert len(result) == 2

# dedupe_documents: max_docs cap must be respected even when more unique docs exist
def test_dedupe_respects_max(docs):
    assert len(dedupe_documents(docs, max_docs=1)) == 1

# dedupe_documents: empty input list must return an empty list without error
def test_dedupe_empty_input():
    assert dedupe_documents([], max_docs=5) == []

# dedupe_documents: original document order must be preserved after deduplication
def test_dedupe_preserves_order(docs):
    result = dedupe_documents(docs, max_docs=10)
    assert result[0].page_content == docs[0].page_content

# dedupe_documents: content key is truncated at 240 chars, so docs differing only after char 240 are treated as duplicates
def test_dedupe_truncates_key_at_240_chars():
    base    = "x" * 240
    doc_a   = Document(page_content=base + "AAA", metadata={})
    doc_b   = Document(page_content=base + "BBB", metadata={})
    result  = dedupe_documents([doc_a, doc_b], max_docs=10)
    assert len(result) == 1


# ── format_context_blocks ─────────────────────────────────────────────────────

# format_context_blocks: each document must be labeled with its chunk index
def test_format_context_blocks_chunk_labels(docs):
    out = format_context_blocks(docs)
    assert "[CHUNK 1]" in out and "[CHUNK 2]" in out

# format_context_blocks: source filename from metadata must appear in the output
def test_format_context_blocks_includes_source(docs):
    out = format_context_blocks(docs)
    assert "lecture1.pdf" in out

# format_context_blocks: actual page content must be included in the formatted block
def test_format_context_blocks_includes_content(docs):
    out = format_context_blocks(docs)
    assert "B+ tree" in out

# format_context_blocks: empty input must return an empty string
def test_format_context_blocks_empty():
    assert format_context_blocks([]) == ""

# format_context_blocks: missing metadata fields must not raise errors
def test_format_context_blocks_missing_metadata():
    doc = Document(page_content="Some content.", metadata={})
    out = format_context_blocks([doc])
    assert "CHUNK 1" in out
    assert "Some content." in out


# ── Pydantic validators ───────────────────────────────────────────────────────

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


# ── Prompt builders ───────────────────────────────────────────────────────────

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

def test_mcq_prompt_distractor_count(docs):
    assert "3 incorrect" in build_mcq_combined_prompt("Hash Tables", docs, num_options=4)
    assert "2 incorrect" in build_mcq_combined_prompt("Hash Tables", docs, num_options=3)

def test_mcq_prompt_scenario_style(docs):
    p = build_mcq_combined_prompt("Hash Tables", docs, style="scenario").lower()
    assert "scenario" in p or "real-world" in p

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

def test_true_false_prompt_contains_subtopic(docs):
    assert "Concurrency" in build_true_false_prompt("Concurrency", docs)

def test_true_false_prompt_difficulty_differs(docs):
    assert build_true_false_prompt("MVCC", docs, difficulty="easy") != \
           build_true_false_prompt("MVCC", docs, difficulty="hard")

def test_topics_prompt_contains_user_text(docs):
    assert "database indexing" in build_topics_prompt("database indexing", docs, n=3)

def test_topics_prompt_contains_n(docs):
    assert "5" in build_topics_prompt("indexing", docs, n=5)


# ── GCS / persistence ─────────────────────────────────────────────────────────

# write_json_to_gcs: must call GCS upload with JSON-serialized payload
def test_write_json_to_gcs(cfg, sample_quiz):
    mock_blob   = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    with patch("quiz_builder.storage.Client", return_value=mock_client):
        uri = write_json_to_gcs(cfg, "quizzes/quiz_001.json", sample_quiz)

    mock_blob.upload_from_string.assert_called_once()
    assert "bucket" in uri

# persist_quiz: must call write_json_to_gcs with correct path prefix
@pytest.mark.asyncio
async def test_persist_quiz(cfg, sample_quiz):
    with patch("quiz_builder.write_json_to_gcs", return_value="gs://bucket/quizzes/quiz.json") as mock_write:
        uri = await persist_quiz(cfg, sample_quiz)
    mock_write.assert_called_once()
    assert "gs://" in uri


# ── Tag cache ─────────────────────────────────────────────────────────────────

# load_tag_cache: must return empty dict when blob does not exist
def test_load_tag_cache_returns_empty_when_missing(cfg):
    mock_blob   = MagicMock()
    mock_blob.exists.return_value = False
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    with patch("quiz_builder.storage.Client", return_value=mock_client):
        result = load_tag_cache(cfg)
    assert result == {}

# load_tag_cache: must return parsed dict when blob exists
def test_load_tag_cache_returns_data_when_exists(cfg):
    mock_blob   = MagicMock()
    mock_blob.exists.return_value = True
    mock_blob.download_as_text.return_value = json.dumps({"key": {"topic": "Hash Tables"}})
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    with patch("quiz_builder.storage.Client", return_value=mock_client):
        result = load_tag_cache(cfg)
    assert result == {"key": {"topic": "Hash Tables"}}

# load_tag_cache: must return empty dict on any GCS exception
def test_load_tag_cache_returns_empty_on_error(cfg):
    with patch("quiz_builder.storage.Client", side_effect=Exception("GCS error")):
        result = load_tag_cache(cfg)
    assert result == {}

# save_tag_cache: must call GCS upload with JSON content
def test_save_tag_cache(cfg):
    mock_blob   = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    with patch("quiz_builder.storage.Client", return_value=mock_client):
        save_tag_cache(cfg, {"key": {"topic": "Hash Tables"}})

    mock_blob.upload_from_string.assert_called_once()


# ── fetch_available_topics ────────────────────────────────────────────────────

# fetch_available_topics: must return only topics present in the tag cache
@pytest.mark.asyncio
async def test_fetch_available_topics_filters_by_cache(cfg):
    cache = {
        "k1": {"topic": "Hash Tables"},
        "k2": {"topic": "Modern SQL"},
    }
    with patch("quiz_builder.load_tag_cache", return_value=cache):
        topics = await fetch_available_topics(cfg)
    assert "Hash Tables" in topics
    assert "Modern SQL" in topics

# fetch_available_topics: must return full syllabus list when cache load fails
@pytest.mark.asyncio
async def test_fetch_available_topics_fallback_on_error(cfg):
    with patch("quiz_builder.load_tag_cache", side_effect=Exception("fail")):
        topics = await fetch_available_topics(cfg)
    assert len(topics) > 0


# ── RUBRIC: ROUGE evaluation ──────────────────────────────────────────────────

# rouge: a model answer identical to the reference must score 1.0 on both metrics
def test_rouge_perfect_match():
    scores = compute_rouge("B+ trees store data in leaf nodes.", "B+ trees store data in leaf nodes.")
    assert scores["rouge1"] == 1.0
    assert scores["rougeL"] == 1.0

# rouge: a completely unrelated hypothesis must score below 0.3
def test_rouge_low_score_on_unrelated():
    scores = compute_rouge("The sky is blue and clouds are white.", "B+ trees store data in leaf nodes.")
    assert scores["rouge1"] < 0.3

# rouge: a partial overlap must score between 0 and 1 exclusively
def test_rouge_partial_overlap():
    scores = compute_rouge("B+ trees store data in nodes.", "B+ trees store all data in leaf nodes linked in a list.")
    assert 0.0 < scores["rouge1"] < 1.0

# rouge: long_answer model_answer must score above 0.1 against a related reference
def test_rouge_long_answer_quality(long_answer_response):
    reference = "A clustered index orders data physically on disk according to the index key."
    scores    = compute_rouge(long_answer_response["model_answer"], reference)
    assert scores["rouge1"] > 0.1


# ── RUBRIC: Data splitting ────────────────────────────────────────────────────

# data split: chunk counts across train/val/test must sum to total chunk count
def test_data_split_counts_sum_to_total(docs):
    chunks = docs * 10
    train, val, test = split_chunks(chunks)
    assert len(train) + len(val) + len(test) == len(chunks)

# data split: train set must be the largest partition
def test_data_split_train_is_largest(docs):
    chunks = docs * 10
    train, val, test = split_chunks(chunks)
    assert len(train) > len(val)
    assert len(train) > len(test)

# data split: val and test must each be non-empty for sufficiently large input
def test_data_split_val_and_test_nonempty(docs):
    chunks = docs * 10
    _, val, test = split_chunks(chunks)
    assert len(val) > 0
    assert len(test) > 0

# data split: invalid ratios that exceed 1.0 must raise AssertionError
def test_data_split_ratios_invalid_raises():
    with pytest.raises(AssertionError):
        split_chunks([], train=0.8, val=0.2, test=0.2)


# ── RUBRIC: Token-length proxy metric ────────────────────────────────────────

# token length: easy and hard prompts must differ in token count since difficulty wording differs
def test_hard_prompt_longer_than_easy(docs):
    easy_tokens = token_length(build_mcq_combined_prompt("Hash Tables", docs, difficulty="easy"))
    hard_tokens = token_length(build_mcq_combined_prompt("Hash Tables", docs, difficulty="hard"))
    assert easy_tokens != hard_tokens

# token length: fill_blank prompt token count must be positive
def test_fill_blank_prompt_token_length_positive(docs):
    assert token_length(build_fill_blank_prompt("Indexes", docs)) > 0

# token length: long_answer and true_false prompts must differ since they have different instructions
def test_long_answer_prompt_longer_than_true_false(docs):
    la = token_length(build_long_answer_prompt("Concurrency", docs))
    tf = token_length(build_true_false_prompt("Concurrency", docs))
    assert la != tf

# token length: context with more chunks must produce a longer prompt
def test_more_chunks_produce_longer_prompt(docs):
    single = token_length(build_mcq_combined_prompt("Hash Tables", docs[:1]))
    both   = token_length(build_mcq_combined_prompt("Hash Tables", docs))
    assert both > single


# ── RUBRIC: Numeric comparison MCQ vs fill_blank ──────────────────────────────

# comparison: MCQ prompt must be longer than fill_blank due to distractor instructions
def test_mcq_prompt_longer_than_fill_blank(docs):
    assert token_length(build_mcq_combined_prompt("Hash Tables", docs)) > \
           token_length(build_fill_blank_prompt("Hash Tables", docs))

# comparison: MCQ explanation must be longer than the answer itself
def test_mcq_explanation_longer_than_answer(mcq_response):
    assert len(mcq_response["explanation"]) > len(mcq_response["answer"])

# comparison: fill_blank answer must be no longer than MCQ answer
def test_fill_blank_answer_shorter_than_mcq(mcq_response, fill_blank_response):
    assert token_length(fill_blank_response["answer"]) <= token_length(mcq_response["answer"])


# ── RUBRIC: Numeric comparison easy vs hard ───────────────────────────────────

# comparison: hard MCQ prompt must differ in token count from easy
def test_hard_mcq_more_tokens_than_easy(docs):
    assert token_length(build_mcq_combined_prompt("Hash Tables", docs, difficulty="hard")) != \
           token_length(build_mcq_combined_prompt("Hash Tables", docs, difficulty="easy"))

# comparison: hard fill_blank prompt must contain at least as many tokens as easy
def test_hard_fill_blank_more_tokens_than_easy(docs):
    assert token_length(build_fill_blank_prompt("Indexes", docs, difficulty="hard")) >= \
           token_length(build_fill_blank_prompt("Indexes", docs, difficulty="easy"))

# comparison: hard long_answer prompt must contain at least as many tokens as easy
def test_hard_long_answer_more_tokens_than_easy(docs):
    assert token_length(build_long_answer_prompt("Joins", docs, difficulty="hard")) >= \
           token_length(build_long_answer_prompt("Joins", docs, difficulty="easy"))


# ── Async pipeline ────────────────────────────────────────────────────────────

# call_groq_json: mocked API response must be parsed and returned as a dict
@pytest.mark.asyncio
async def test_call_groq_json_returns_parsed(cfg):
    fake = {
        "choices": [{"message": {"content": '{"question": "What is a B+ tree?"}'}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=fake)
    mock_resp.raise_for_status = MagicMock()

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_post.return_value.__aexit__  = AsyncMock(return_value=False)
        result = await call_groq_json(cfg, "unique_prompt_parsed_001")

    assert result["question"] == "What is a B+ tree?"

# call_groq_json: identical prompt must hit cache and not make a second API call
@pytest.mark.asyncio
async def test_call_groq_json_cache_hit(cfg):
    fake = {
        "choices": [{"message": {"content": '{"cached": true}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10},
    }
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=fake)
    mock_resp.raise_for_status = MagicMock()

    prompt = "unique_cache_test_prompt_abc123"
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_post.return_value.__aexit__  = AsyncMock(return_value=False)
        await call_groq_json(cfg, prompt)
        await call_groq_json(cfg, prompt)
        assert mock_post.call_count == 1

# call_groq_json: 429 rate limit response must trigger a retry
@pytest.mark.asyncio
async def test_call_groq_json_retries_on_429(cfg):
    rate_limit_resp = AsyncMock()
    rate_limit_resp.status = 429
    rate_limit_resp.raise_for_status = MagicMock()

    success_resp = AsyncMock()
    success_resp.status = 200
    success_resp.json = AsyncMock(return_value={
        "choices": [{"message": {"content": '{"retried": true}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10},
    })
    success_resp.raise_for_status = MagicMock()

    with patch("aiohttp.ClientSession.post") as mock_post, \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_post.return_value.__aenter__ = AsyncMock(
            side_effect=[rate_limit_resp, success_resp]
        )
        mock_post.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await call_groq_json(cfg, "unique_retry_prompt_xyz789")

    assert result["retried"] is True

# generate_topics: must return a list of plain strings from the mocked Groq response
@pytest.mark.asyncio
async def test_generate_topics_returns_strings(cfg, docs, topics_response):
    with patch("quiz_builder.call_groq_json", AsyncMock(return_value=topics_response)):
        topics = await generate_topics(cfg, "hash tables", docs, n=2)
    assert all(isinstance(t, str) for t in topics)

# generate_topics: returned list must not exceed the requested n
@pytest.mark.asyncio
async def test_generate_topics_respects_n(cfg, docs, topics_response):
    with patch("quiz_builder.call_groq_json", AsyncMock(return_value=topics_response)):
        topics = await generate_topics(cfg, "hash tables", docs, n=1)
    assert len(topics) <= 1

# build_quiz (mcq): output must contain required MCQ fields
@pytest.mark.asyncio
async def test_build_quiz_mcq_structure(cfg, docs, mcq_response):
    with patch("quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("quiz_builder.call_groq_json",   AsyncMock(return_value=mcq_response)):
        quiz = await build_quiz(cfg, "hash tables", 1, "medium", 6, 12, question_type="mcq")

    q = quiz["questions"][0]
    assert q["type"] == "mcq"
    assert "question" in q and "answer" in q and "options" in q and "explanation" in q

# build_quiz (fill_blank): question must contain the blank marker and correct type
@pytest.mark.asyncio
async def test_build_quiz_fill_blank_structure(cfg, docs, fill_blank_response):
    with patch("quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("quiz_builder.generate_topics",  AsyncMock(return_value=["Indexes"])), \
         patch("quiz_builder.call_groq_json",   AsyncMock(return_value=fill_blank_response)):
        quiz = await build_quiz(cfg, "indexes", 1, "medium", 6, 12, question_type="fill_blank")

    q = quiz["questions"][0]
    assert q["type"] == "fill_blank"
    assert "_____" in q["question"]

# build_quiz (long_answer): output must contain model_answer and key_points fields
@pytest.mark.asyncio
async def test_build_quiz_long_answer_structure(cfg, docs, long_answer_response):
    with patch("quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("quiz_builder.generate_topics",  AsyncMock(return_value=["Indexes"])), \
         patch("quiz_builder.call_groq_json",   AsyncMock(return_value=long_answer_response)):
        quiz = await build_quiz(cfg, "indexes", 1, "medium", 6, 12, question_type="long_answer")

    q = quiz["questions"][0]
    assert q["type"] == "long_answer"
    assert "model_answer" in q and "key_points" in q

# build_quiz (true_false): answer must be exactly "True" or "False"
@pytest.mark.asyncio
async def test_build_quiz_true_false_structure(cfg, docs, true_false_response):
    with patch("quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("quiz_builder.generate_topics",  AsyncMock(return_value=["Concurrency"])), \
         patch("quiz_builder.call_groq_json",   AsyncMock(return_value=true_false_response)):
        quiz = await build_quiz(cfg, "concurrency", 1, "medium", 6, 12, question_type="true_false")

    q = quiz["questions"][0]
    assert q["type"] == "true_false"
    assert q["answer"] in ("True", "False")

# build_quiz: run_metadata must record the model name used for generation
@pytest.mark.asyncio
async def test_build_quiz_metadata_model_name(cfg, docs, mcq_response):
    with patch("quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("quiz_builder.call_groq_json",   AsyncMock(return_value=mcq_response)):
        quiz = await build_quiz(cfg, "hash tables", 1, "medium", 6, 12)

    assert quiz["run_metadata"]["model"] == "llama-3.1-8b-instant"

# build_quiz: must raise RuntimeError when no documents are retrieved
@pytest.mark.asyncio
async def test_build_quiz_no_docs_raises(cfg):
    with patch("quiz_builder.retrieve_context", AsyncMock(return_value=[])):
        with pytest.raises(RuntimeError, match="No documents retrieved"):
            await build_quiz(cfg, "hash tables", 1, "medium", 6, 12)

# build_quiz: difficulty value must be stored correctly in the quiz output
@pytest.mark.asyncio
async def test_build_quiz_difficulty_stored(cfg, docs, mcq_response):
    with patch("quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("quiz_builder.call_groq_json",   AsyncMock(return_value=mcq_response)):
        easy = await build_quiz(cfg, "hash tables", 1, "easy",  6, 12)
        hard = await build_quiz(cfg, "hash tables", 1, "hard",  6, 12)

    assert easy["difficulty"] == "easy"
    assert hard["difficulty"] == "hard"

# build_quiz: all topics returning bad responses must raise RuntimeError
@pytest.mark.asyncio
async def test_build_quiz_skips_invalid_groq_response(cfg, docs):
    bad_response = {"question": "x", "answer": "", "explanation": "", "sources": [], "incorrect_answers": []}
    with patch("quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("quiz_builder.call_groq_json",   AsyncMock(side_effect=Exception("bad response"))):
        with pytest.raises(RuntimeError, match="Quiz generation failed"):
            await build_quiz(cfg, "hash tables", 1, "medium", 6, 12, question_type="mcq")

# ── Comparison summary printed at end of session ─────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def print_comparison_summary():
    yield
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    base_docs = [
        Document(page_content="A B+ tree stores all data in leaf nodes linked in a list.",
                 metadata={"source": "lecture1.pdf", "page": 3, "chunk_id": "c1"}),
        Document(page_content="Hash indexes do not support range queries, only equality checks.",
                 metadata={"source": "lecture2.pdf", "page": 7, "chunk_id": "c2"}),
    ]

    comparisons = {
        "Question Type (prompt token length)": {
            "MCQ":         token_length(build_mcq_combined_prompt("Hash Tables", base_docs)),
            "Fill-blank":  token_length(build_fill_blank_prompt("Hash Tables", base_docs)),
            "Long answer":  token_length(build_long_answer_prompt("Hash Tables", base_docs)),
            "True/False":  token_length(build_true_false_prompt("Hash Tables", base_docs)),
        },
        "Difficulty / MCQ (prompt token length)": {
            "Easy":   token_length(build_mcq_combined_prompt("Hash Tables", base_docs, difficulty="easy")),
            "Medium": token_length(build_mcq_combined_prompt("Hash Tables", base_docs, difficulty="medium")),
            "Hard":   token_length(build_mcq_combined_prompt("Hash Tables", base_docs, difficulty="hard")),
        },
        "Difficulty / Fill-blank (prompt token length)": {
            "Easy":   token_length(build_fill_blank_prompt("Hash Tables", base_docs, difficulty="easy")),
            "Medium": token_length(build_fill_blank_prompt("Hash Tables", base_docs, difficulty="medium")),
            "Hard":   token_length(build_fill_blank_prompt("Hash Tables", base_docs, difficulty="hard")),
        },
        "Difficulty / Long Answer (prompt token length)": {
            "Easy":   token_length(build_long_answer_prompt("Hash Tables", base_docs, difficulty="easy")),
            "Medium": token_length(build_long_answer_prompt("Hash Tables", base_docs, difficulty="medium")),
            "Hard":   token_length(build_long_answer_prompt("Hash Tables", base_docs, difficulty="hard")),
        },
        "Difficulty / True-False (prompt token length)": {
            "Easy":   token_length(build_true_false_prompt("Hash Tables", base_docs, difficulty="easy")),
            "Medium": token_length(build_true_false_prompt("Hash Tables", base_docs, difficulty="medium")),
            "Hard":   token_length(build_true_false_prompt("Hash Tables", base_docs, difficulty="hard")),
        },
        "Prompt Style / MCQ (prompt token length)": {
            "Conceptual": token_length(build_mcq_combined_prompt("Hash Tables", base_docs, style="conceptual")),
            "Scenario":   token_length(build_mcq_combined_prompt("Hash Tables", base_docs, style="scenario")),
            "Definition": token_length(build_mcq_combined_prompt("Hash Tables", base_docs, style="definition")),
        },
        "MCQ Options / distractor count (prompt token length)": {
            "3 options (2 distractors)": token_length(build_mcq_combined_prompt("Hash Tables", base_docs, num_options=3)),
            "4 options (3 distractors)": token_length(build_mcq_combined_prompt("Hash Tables", base_docs, num_options=4)),
            "5 options (4 distractors)": token_length(build_mcq_combined_prompt("Hash Tables", base_docs, num_options=5)),
        },
        "Retrieval context size (prompt token length)": {
            "1 chunk":  token_length(build_mcq_combined_prompt("Hash Tables", base_docs[:1])),
            "2 chunks": token_length(build_mcq_combined_prompt("Hash Tables", base_docs)),
        },
        "ROUGE scores (model answer vs reference)": {
            "ROUGE-1 (perfect match)":   round(scorer.score("B+ trees store data in leaf nodes.", "B+ trees store data in leaf nodes.")["rouge1"].fmeasure, 3),
            "ROUGE-L (perfect match)":   round(scorer.score("B+ trees store data in leaf nodes.", "B+ trees store data in leaf nodes.")["rougeL"].fmeasure, 3),
            "ROUGE-1 (partial overlap)": round(scorer.score("B+ trees store data in nodes.", "B+ trees store all data in leaf nodes linked in a list.")["rouge1"].fmeasure, 3),
            "ROUGE-L (partial overlap)": round(scorer.score("B+ trees store data in nodes.", "B+ trees store all data in leaf nodes linked in a list.")["rougeL"].fmeasure, 3),
            "ROUGE-1 (unrelated)":       round(scorer.score("The sky is blue.", "B+ trees store data in leaf nodes.")["rouge1"].fmeasure, 3),
            "ROUGE-L (unrelated)":       round(scorer.score("The sky is blue.", "B+ trees store data in leaf nodes.")["rougeL"].fmeasure, 3),
        },
    }

    col_w = 36
    val_w = 10
    sep   = "=" * 80

    print(f"\n\n{sep}")
    print("  QUIZ BUILDER — CUSTOMIZATION COMPARISON SUMMARY")
    print(sep)

    for section, values in comparisons.items():
        print(f"\n  {section}")
        print("  " + "-" * (col_w + val_w + 4))
        for label, value in values.items():
            print(f"  {label:<{col_w}} {str(value):>{val_w}}")

    print(f"\n{sep}\n")


# build_quiz: partial topic failure must not prevent other topics from generating
@pytest.mark.asyncio
async def test_build_quiz_partial_failure_continues(cfg, docs, mcq_response):
    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("topic failed")
        return mcq_response

    with patch("quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("quiz_builder.generate_topics",  AsyncMock(return_value=["Bad Topic", "Hash Tables"])), \
         patch("quiz_builder.call_groq_json",   side_effect=side_effect):
        quiz = await build_quiz(cfg, "hash tables", 2, "medium", 6, 12, question_type="mcq")

    assert len(quiz["questions"]) == 1