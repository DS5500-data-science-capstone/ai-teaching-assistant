from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import ValidationError
from langchain_core.documents import Document

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
    format_context_blocks,
    generate_topics,
    normalize_whitespace,
    safe_json_extract,
    build_quiz,
    utc_compact_ts,
    _cache_key,
    _reset_minute_budget,
    _record_usage,
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


# clean_text: removes null bytes that can corrupt downstream text processing
def test_clean_text_removes_null_bytes():
    assert clean_text("hello\x00world") == "helloworld"

# clean_text: strips leading/trailing whitespace before storing or embedding text
def test_clean_text_strips_whitespace():
    assert clean_text("  padded  ") == "padded"

# clean_text: empty string input should return empty string without error
def test_clean_text_empty():
    assert clean_text("") == ""

# normalize_whitespace: collapses multiple spaces into one for consistent tokenization
def test_normalize_whitespace_collapses_spaces():
    assert normalize_whitespace("a   b   c") == "a b c"

# normalize_whitespace: handles mixed newline and tab characters from raw PDF extraction
def test_normalize_whitespace_newlines_and_tabs():
    assert normalize_whitespace("a\n\tb\r\nc") == "a b c"

# normalize_whitespace: already-clean text should pass through unchanged
def test_normalize_whitespace_already_clean():
    assert normalize_whitespace("already clean") == "already clean"

# safe_json_extract: extracts a JSON object embedded in surrounding text (typical LLM output)
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

# utc_compact_ts: output must be exactly 16 chars (YYYYMMDDTHHMMSSz) and end with Z
def test_utc_compact_ts_format():
    ts = utc_compact_ts()
    assert len(ts) == 16
    assert ts.endswith("Z")

# _cache_key: same input must always produce the same hash (deterministic)
def test_cache_key_deterministic():
    assert _cache_key("hello") == _cache_key("hello")

# _cache_key: different inputs must produce different hashes to avoid cache collisions
def test_cache_key_different_inputs():
    assert _cache_key("hello") != _cache_key("world")

# _reset_minute_budget: must run without raising any errors
def test_reset_minute_budget_runs_without_error():
    _reset_minute_budget()

# _record_usage: token counts must increase by at least prompt + completion tokens
def test_record_usage_updates_totals():
    from quiz_builder import _budget
    before = _budget["total_tokens"]
    _record_usage("llama-3.1-8b-instant", 100, 50)
    assert _budget["total_tokens"] >= before + 150

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

# TopicItem: valid topic string must be stored as-is
def test_topic_item_valid():
    assert TopicItem(topic="Hash Tables").topic == "Hash Tables"

# TopicItem: leading/trailing whitespace must be stripped from the topic field
def test_topic_item_stripped():
    assert TopicItem(topic="  Hash Tables  ").topic == "Hash Tables"

# TopicItem: topics shorter than 3 characters must fail validation
def test_topic_item_too_short_raises():
    with pytest.raises(ValidationError):
        TopicItem(topic="DB")

# TopicList: an empty topics list must fail validation
def test_topic_list_empty_raises():
    with pytest.raises(ValidationError):
        TopicList(topics=[])

# QuestionItem: valid question and source must be stored correctly
def test_question_item_valid():
    q = QuestionItem(question="What is a B+ tree index used for?", sources=["CHUNK 1"])
    assert "B+ tree" in q.question

# QuestionItem: whitespace around the question string must be stripped
def test_question_item_stripped():
    assert QuestionItem(question="  What is a B+ tree?  ", sources=["CHUNK 1"]).question == "What is a B+ tree?"

# QuestionItem: questions under 10 characters must fail validation
def test_question_item_too_short_raises():
    with pytest.raises(ValidationError):
        QuestionItem(question="Short?", sources=["CHUNK 1"])

# QuestionItem: questions over 500 characters must fail validation
def test_question_item_too_long_raises():
    with pytest.raises(ValidationError):
        QuestionItem(question="x" * 501, sources=["CHUNK 1"])

# QuestionItem: at least one source must be provided
def test_question_item_empty_sources_raises():
    with pytest.raises(ValidationError):
        QuestionItem(question="What is a B+ tree index?", sources=[])

# QuestionItem: sources list must be capped at 3 entries by the validator
def test_question_item_sources_capped_at_three():
    q = QuestionItem(question="What is a B+ tree index?", sources=["C1", "C2", "C3", "C4"])
    assert len(q.sources) == 3

# AnswerItem: valid answer must be stored correctly
def test_answer_item_valid():
    assert AnswerItem(answer="Dense", explanation="One entry per key.", sources=["CHUNK 1"]).answer == "Dense"

# AnswerItem: whitespace around answer must be stripped by the validator
def test_answer_item_stripped():
    assert AnswerItem(answer="  Dense  ", explanation="One entry per key.", sources=["CHUNK 1"]).answer == "Dense"

# AnswerItem: empty answer string must fail validation
def test_answer_item_empty_raises():
    with pytest.raises(ValidationError):
        AnswerItem(answer="", explanation="Explanation.", sources=["CHUNK 1"])

# AnswerItem: explanation under 5 characters must fail validation
def test_answer_item_short_explanation_raises():
    with pytest.raises(ValidationError):
        AnswerItem(answer="Dense", explanation="OK", sources=["CHUNK 1"])

# DistractorItem: valid distractor must be stored correctly
def test_distractor_item_valid():
    d = DistractorItem(incorrect_answer="Sparse", explanation="Skips some keys.", sources=["CHUNK 1"])
    assert d.incorrect_answer == "Sparse"

# DistractorItem: incorrect answers under 2 characters must fail validation
def test_distractor_item_too_short_raises():
    with pytest.raises(ValidationError):
        DistractorItem(incorrect_answer="X", explanation="Too short.", sources=["CHUNK 1"])

# DistractorItem: at least one source must be provided for the distractor
def test_distractor_item_empty_sources_raises():
    with pytest.raises(ValidationError):
        DistractorItem(incorrect_answer="Sparse", explanation="Skips keys.", sources=[])

# build_mcq_combined_prompt: subtopic must appear in the generated prompt
def test_mcq_prompt_contains_subtopic(docs):
    assert "Hash Tables" in build_mcq_combined_prompt("Hash Tables", docs)

# build_mcq_combined_prompt: easy difficulty must include "straightforward" wording
def test_mcq_prompt_easy_wording(docs):
    assert "straightforward" in build_mcq_combined_prompt("Hash Tables", docs, difficulty="easy").lower()

# build_mcq_combined_prompt: hard difficulty must include multi-step or deep reasoning wording
def test_mcq_prompt_hard_wording(docs):
    p = build_mcq_combined_prompt("Hash Tables", docs, difficulty="hard").lower()
    assert "multi-step" in p or "deep" in p

# build_mcq_combined_prompt: easy and hard prompts must differ from each other
def test_mcq_prompt_difficulty_differs(docs):
    assert build_mcq_combined_prompt("Hash Tables", docs, difficulty="easy") != \
           build_mcq_combined_prompt("Hash Tables", docs, difficulty="hard")

# build_mcq_combined_prompt: num_options controls how many distractors are requested
def test_mcq_prompt_distractor_count(docs):
    assert "3 incorrect" in build_mcq_combined_prompt("Hash Tables", docs, num_options=4)
    assert "2 incorrect" in build_mcq_combined_prompt("Hash Tables", docs, num_options=3)

# build_mcq_combined_prompt: scenario style must include scenario or real-world wording
def test_mcq_prompt_scenario_style(docs):
    p = build_mcq_combined_prompt("Hash Tables", docs, style="scenario").lower()
    assert "scenario" in p or "real-world" in p

# build_fill_blank_prompt: subtopic must appear in the generated prompt
def test_fill_blank_prompt_contains_subtopic(docs):
    assert "Indexes" in build_fill_blank_prompt("Indexes", docs)

# build_fill_blank_prompt: blank placeholder must be present for the student to fill
def test_fill_blank_prompt_has_blank_marker(docs):
    assert "_____" in build_fill_blank_prompt("Indexes", docs)

# build_fill_blank_prompt: easy and hard difficulty prompts must differ
def test_fill_blank_prompt_difficulty_differs(docs):
    assert build_fill_blank_prompt("Indexes", docs, difficulty="easy") != \
           build_fill_blank_prompt("Indexes", docs, difficulty="hard")

# build_long_answer_prompt: subtopic must appear in the generated prompt
def test_long_answer_prompt_contains_subtopic(docs):
    assert "Query Optimization" in build_long_answer_prompt("Query Optimization", docs)

# build_long_answer_prompt: easy and hard difficulty prompts must differ
def test_long_answer_prompt_difficulty_differs(docs):
    assert build_long_answer_prompt("Joins", docs, difficulty="easy") != \
           build_long_answer_prompt("Joins", docs, difficulty="hard")

# build_true_false_prompt: subtopic must appear in the generated prompt
def test_true_false_prompt_contains_subtopic(docs):
    assert "Concurrency" in build_true_false_prompt("Concurrency", docs)

# build_true_false_prompt: easy and hard difficulty prompts must differ
def test_true_false_prompt_difficulty_differs(docs):
    assert build_true_false_prompt("MVCC", docs, difficulty="easy") != \
           build_true_false_prompt("MVCC", docs, difficulty="hard")

# build_topics_prompt: user-provided text must be included in the topics prompt
def test_topics_prompt_contains_user_text(docs):
    assert "database indexing" in build_topics_prompt("database indexing", docs, n=3)

# build_topics_prompt: requested number of topics must appear in the prompt
def test_topics_prompt_contains_n(docs):
    assert "5" in build_topics_prompt("indexing", docs, n=5)

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

# build_quiz (mcq): output must contain required MCQ fields in the question entry
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

# build_quiz: must raise RuntimeError when no documents are retrieved from the vector store
@pytest.mark.asyncio
async def test_build_quiz_no_docs_raises(cfg):
    with patch("quiz_builder.retrieve_context", AsyncMock(return_value=[])):
        with pytest.raises(RuntimeError, match="No documents retrieved"):
            await build_quiz(cfg, "hash tables", 1, "medium", 6, 12)

# build_quiz: difficulty value must be stored correctly in the quiz output for both levels
@pytest.mark.asyncio
async def test_build_quiz_difficulty_stored(cfg, docs, mcq_response):
    with patch("quiz_builder.retrieve_context", AsyncMock(return_value=docs)), \
         patch("quiz_builder.generate_topics",  AsyncMock(return_value=["Hash Tables"])), \
         patch("quiz_builder.call_groq_json",   AsyncMock(return_value=mcq_response)):
        easy = await build_quiz(cfg, "hash tables", 1, "easy", 6, 12)
        hard = await build_quiz(cfg, "hash tables", 1, "hard", 6, 12)

    assert easy["difficulty"] == "easy"
    assert hard["difficulty"] == "hard"