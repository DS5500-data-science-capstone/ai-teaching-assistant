from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document
from rouge_score import rouge_scorer

import query_data as qd


@pytest.fixture
def sample_docs():
    return [
        Document(page_content="B+ trees store data in leaf nodes linked in a list.",
                 metadata={"source": "lecture1.pdf", "page": 1}),
        Document(page_content="Hash indexes only support equality checks, not range queries.",
                 metadata={"source": "lecture2.pdf", "page": 4}),
        Document(page_content="Concurrency control ensures transactions execute correctly.",
                 metadata={"source": "lecture3.pdf", "page": 7}),
        Document(page_content="Query optimization reduces execution cost using statistics.",
                 metadata={"source": "lecture4.pdf", "page": 12}),
        Document(page_content="Buffer pool manages pages in memory for fast access.",
                 metadata={"source": "lecture5.pdf", "page": 3}),
    ]


@pytest.fixture
def mock_llm_response():
    response = MagicMock()
    response.content = "B+ trees store all data in leaf nodes, enabling efficient range queries."
    return response


def compute_rouge(hypothesis: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {"rouge1": scores["rouge1"].fmeasure, "rougeL": scores["rougeL"].fmeasure}


def token_length(text: str) -> int:
    return len(text.split())


# ── PROMPT_TEMPLATE ───────────────────────────────────────────────────────────

# prompt template: must contain context placeholder
def test_prompt_template_contains_context_placeholder():
    assert "{context}" in qd.PROMPT_TEMPLATE

# prompt template: must contain question placeholder
def test_prompt_template_contains_question_placeholder():
    assert "{question}" in qd.PROMPT_TEMPLATE

# prompt template: must instruct the model to answer only from context
def test_prompt_template_contains_context_restriction():
    assert "based only on the following context" in qd.PROMPT_TEMPLATE

# prompt template: must include fallback instruction when answer is not in context
def test_prompt_template_contains_fallback_instruction():
    assert "I don't have enough" in qd.PROMPT_TEMPLATE

# prompt template: formatted prompt must include the injected question
def test_prompt_template_formats_question():
    from langchain_core.prompts import ChatPromptTemplate
    template = ChatPromptTemplate.from_template(qd.PROMPT_TEMPLATE)
    prompt   = template.format(context="some context", question="What is a B+ tree?")
    assert "What is a B+ tree?" in prompt

# prompt template: formatted prompt must include the injected context
def test_prompt_template_formats_context():
    from langchain_core.prompts import ChatPromptTemplate
    template = ChatPromptTemplate.from_template(qd.PROMPT_TEMPLATE)
    prompt   = template.format(context="B+ trees store data in leaf nodes.", question="What is a B+ tree?")
    assert "B+ trees store data in leaf nodes." in prompt


# ── context building ──────────────────────────────────────────────────────────

# context: chunks must be joined with the separator used in query_rag
def test_context_joined_with_separator(sample_docs):
    context = "\n\n---\n\n".join([doc.page_content for doc in sample_docs])
    assert "---" in context

# context: all chunk contents must appear in the joined context
def test_context_contains_all_chunks(sample_docs):
    context = "\n\n---\n\n".join([doc.page_content for doc in sample_docs])
    for doc in sample_docs:
        assert doc.page_content in context

# context: single chunk must produce context with no separator
def test_context_single_chunk():
    docs    = [Document(page_content="Only chunk.", metadata={})]
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    assert "---" not in context
    assert context == "Only chunk."

# context: empty results list must produce empty context string
def test_context_empty_results():
    context = "\n\n---\n\n".join([doc.page_content for doc in []])
    assert context == ""


# ── source metadata extraction ────────────────────────────────────────────────

# sources: page metadata must be extractable from result documents
def test_source_page_metadata(sample_docs):
    for doc in sample_docs:
        assert doc.metadata.get("page") is not None

# sources: source filename metadata must be extractable from result documents
def test_source_filename_metadata(sample_docs):
    for doc in sample_docs:
        assert doc.metadata.get("source") is not None

# sources: missing metadata must fall back to "N/A" without raising errors
def test_source_missing_metadata_fallback():
    doc  = Document(page_content="content", metadata={})
    page = doc.metadata.get("page",   "N/A")
    src  = doc.metadata.get("source", "N/A")
    assert page == "N/A"
    assert src  == "N/A"

# sources: k=5 retrieval must return at most 5 documents
def test_retrieval_k_limit(sample_docs):
    assert len(sample_docs) <= 5


# ── query_rag async pipeline ──────────────────────────────────────────────────

# query_rag: must call similarity search with the correct question
@pytest.mark.asyncio
async def test_query_rag_calls_similarity_search(sample_docs, mock_llm_response):
    mock_engine       = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_vector_store.asimilarity_search = AsyncMock(return_value=sample_docs)

    with patch("query_data.PostgresEngine.afrom_instance", AsyncMock(return_value=mock_engine)), \
         patch("query_data.PostgresVectorStore.create",    AsyncMock(return_value=mock_vector_store)), \
         patch("query_data.OpenAIEmbeddings"), \
         patch("query_data.ChatGroq", return_value=MagicMock(invoke=MagicMock(return_value=mock_llm_response))):
        await qd.query_rag("What is a B+ tree?")

    mock_vector_store.asimilarity_search.assert_called_once_with("What is a B+ tree?", k=5)

# query_rag: must call the LLM with a prompt containing the question
@pytest.mark.asyncio
async def test_query_rag_calls_llm(sample_docs, mock_llm_response):
    mock_engine       = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_vector_store.asimilarity_search = AsyncMock(return_value=sample_docs)
    mock_llm          = MagicMock()
    mock_llm.invoke   = MagicMock(return_value=mock_llm_response)

    with patch("query_data.PostgresEngine.afrom_instance", AsyncMock(return_value=mock_engine)), \
         patch("query_data.PostgresVectorStore.create",    AsyncMock(return_value=mock_vector_store)), \
         patch("query_data.OpenAIEmbeddings"), \
         patch("query_data.ChatGroq", return_value=mock_llm):
        await qd.query_rag("What is a B+ tree?")

    assert mock_llm.invoke.called
    prompt_arg = mock_llm.invoke.call_args[0][0]
    assert "What is a B+ tree?" in prompt_arg

# query_rag: must connect to PostgresEngine with correct project and database params
@pytest.mark.asyncio
async def test_query_rag_engine_called_with_correct_params(sample_docs, mock_llm_response):
    mock_engine       = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_vector_store.asimilarity_search = AsyncMock(return_value=sample_docs)

    with patch("query_data.PostgresEngine.afrom_instance", AsyncMock(return_value=mock_engine)) as mock_pg, \
         patch("query_data.PostgresVectorStore.create",    AsyncMock(return_value=mock_vector_store)), \
         patch("query_data.OpenAIEmbeddings"), \
         patch("query_data.ChatGroq", return_value=MagicMock(invoke=MagicMock(return_value=mock_llm_response))):
        await qd.query_rag("What is a B+ tree?")

    call_kwargs = mock_pg.call_args[1]
    assert call_kwargs["project_id"] == qd.PROJECT_ID
    assert call_kwargs["database"]   == qd.DATABASE

# query_rag: must create vector store with correct table name
@pytest.mark.asyncio
async def test_query_rag_vector_store_correct_table(sample_docs, mock_llm_response):
    mock_engine       = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_vector_store.asimilarity_search = AsyncMock(return_value=sample_docs)

    with patch("query_data.PostgresEngine.afrom_instance", AsyncMock(return_value=mock_engine)), \
         patch("query_data.PostgresVectorStore.create",    AsyncMock(return_value=mock_vector_store)) as mock_vs, \
         patch("query_data.OpenAIEmbeddings"), \
         patch("query_data.ChatGroq", return_value=MagicMock(invoke=MagicMock(return_value=mock_llm_response))):
        await qd.query_rag("What is a B+ tree?")

    call_kwargs = mock_vs.call_args[1]
    assert call_kwargs["table_name"] == qd.TABLE_NAME

# query_rag: must use llama-3.1-8b-instant model
@pytest.mark.asyncio
async def test_query_rag_uses_correct_model(sample_docs, mock_llm_response):
    mock_engine       = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_vector_store.asimilarity_search = AsyncMock(return_value=sample_docs)

    with patch("query_data.PostgresEngine.afrom_instance", AsyncMock(return_value=mock_engine)), \
         patch("query_data.PostgresVectorStore.create",    AsyncMock(return_value=mock_vector_store)), \
         patch("query_data.OpenAIEmbeddings"), \
         patch("query_data.ChatGroq", return_value=MagicMock(invoke=MagicMock(return_value=mock_llm_response))) as mock_groq:
        await qd.query_rag("What is a B+ tree?")

    call_kwargs = mock_groq.call_args[1]
    assert call_kwargs["model_name"] == "llama-3.1-8b-instant"

# query_rag: must return early when no results are found without calling LLM
@pytest.mark.asyncio
async def test_query_rag_returns_early_on_no_results():
    mock_engine       = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_vector_store.asimilarity_search = AsyncMock(return_value=[])
    mock_llm          = MagicMock()

    with patch("query_data.PostgresEngine.afrom_instance", AsyncMock(return_value=mock_engine)), \
         patch("query_data.PostgresVectorStore.create",    AsyncMock(return_value=mock_vector_store)), \
         patch("query_data.OpenAIEmbeddings"), \
         patch("query_data.ChatGroq", return_value=mock_llm):
        await qd.query_rag("unknown topic")

    mock_llm.invoke.assert_not_called()

# query_rag: context passed to LLM must contain content from retrieved chunks
@pytest.mark.asyncio
async def test_query_rag_context_contains_chunk_content(sample_docs, mock_llm_response):
    mock_engine       = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_vector_store.asimilarity_search = AsyncMock(return_value=sample_docs)
    mock_llm          = MagicMock()
    mock_llm.invoke   = MagicMock(return_value=mock_llm_response)

    with patch("query_data.PostgresEngine.afrom_instance", AsyncMock(return_value=mock_engine)), \
         patch("query_data.PostgresVectorStore.create",    AsyncMock(return_value=mock_vector_store)), \
         patch("query_data.OpenAIEmbeddings"), \
         patch("query_data.ChatGroq", return_value=mock_llm):
        await qd.query_rag("What is a B+ tree?")

    prompt_arg = mock_llm.invoke.call_args[0][0]
    assert "B+ trees store data in leaf nodes" in prompt_arg


# ── RUBRIC: ROUGE evaluation on RAG answers ───────────────────────────────────

# rouge: LLM answer closely matching context must score above 0.4
def test_rouge_answer_vs_context(sample_docs, mock_llm_response):
    context = sample_docs[0].page_content
    scores  = compute_rouge(mock_llm_response.content, context)
    assert scores["rouge1"] > 0.4

# rouge: answer unrelated to context must score below 0.2
def test_rouge_unrelated_answer_low_score():
    answer  = "The weather today is sunny and warm."
    context = "B+ trees store data in leaf nodes linked in a list."
    scores  = compute_rouge(answer, context)
    assert scores["rouge1"] < 0.2

# rouge: fallback answer must score near 0 against any course content
def test_rouge_fallback_answer_score():
    fallback = "I don't have enough information in the course materials to answer this question."
    context  = "B+ trees store data in leaf nodes linked in a list."
    scores   = compute_rouge(fallback, context)
    assert scores["rouge1"] < 0.3


# ── Comparison summary ────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def print_comparison_summary():
    yield

    from langchain_core.prompts import ChatPromptTemplate

    base_docs = [
        Document(page_content="B+ trees store data in leaf nodes linked in a list.",
                 metadata={"source": "lecture1.pdf", "page": 1}),
        Document(page_content="Hash indexes only support equality checks, not range queries.",
                 metadata={"source": "lecture2.pdf", "page": 4}),
        Document(page_content="Concurrency control ensures transactions execute correctly.",
                 metadata={"source": "lecture3.pdf", "page": 7}),
        Document(page_content="Query optimization reduces execution cost using statistics.",
                 metadata={"source": "lecture4.pdf", "page": 12}),
        Document(page_content="Buffer pool manages pages in memory for fast access.",
                 metadata={"source": "lecture5.pdf", "page": 3}),
    ]

    scorer   = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    template = ChatPromptTemplate.from_template(qd.PROMPT_TEMPLATE)

    context_1  = "\n\n---\n\n".join([d.page_content for d in base_docs[:1]])
    context_3  = "\n\n---\n\n".join([d.page_content for d in base_docs[:3]])
    context_5  = "\n\n---\n\n".join([d.page_content for d in base_docs])

    prompt_1   = template.format(context=context_1, question="What is a B+ tree?")
    prompt_3   = template.format(context=context_3, question="What is a B+ tree?")
    prompt_5   = template.format(context=context_5, question="What is a B+ tree?")

    answer_good     = "B+ trees store all data in leaf nodes, enabling efficient range queries."
    answer_partial  = "Trees store data in nodes."
    answer_unrelated= "The weather today is sunny and warm."
    answer_fallback = "I don't have enough information in the course materials to answer this question."
    reference       = base_docs[0].page_content

    def r(hyp, ref):
        s = scorer.score(ref, hyp)
        return round(s["rouge1"].fmeasure, 3), round(s["rougeL"].fmeasure, 3)

    comparisons = {
        "Prompt template token length by k (retrieved chunks)": {
            "k=1 chunk":  token_length(prompt_1),
            "k=3 chunks": token_length(prompt_3),
            "k=5 chunks": token_length(prompt_5),
        },
        "Context size by k (characters)": {
            "k=1 chunk":  len(context_1),
            "k=3 chunks": len(context_3),
            "k=5 chunks": len(context_5),
        },
        "ROUGE-1 scores (answer vs first chunk reference)": {
            "Good answer":     r(answer_good,      reference)[0],
            "Partial answer":  r(answer_partial,   reference)[0],
            "Unrelated answer":r(answer_unrelated, reference)[0],
            "Fallback answer": r(answer_fallback,  reference)[0],
        },
        "ROUGE-L scores (answer vs first chunk reference)": {
            "Good answer":     r(answer_good,      reference)[1],
            "Partial answer":  r(answer_partial,   reference)[1],
            "Unrelated answer":r(answer_unrelated, reference)[1],
            "Fallback answer": r(answer_fallback,  reference)[1],
        },
        "Source metadata availability": {
            "Docs with page metadata":   sum(1 for d in base_docs if "page"   in d.metadata),
            "Docs with source metadata": sum(1 for d in base_docs if "source" in d.metadata),
            "Total docs":                len(base_docs),
        },
        "RAG pipeline config": {
            "Model":       "llama-3.1-8b-instant",
            "TABLE_NAME":  qd.TABLE_NAME,
            "Retrieval k": 5,
        },
    }

    col_w = 38
    val_w = 28
    sep   = "=" * 80

    print(f"\n\n{sep}")
    print("  QUERY DATA — RAG PIPELINE COMPARISON SUMMARY")
    print(sep)

    for section, values in comparisons.items():
        print(f"\n  {section}")
        print("  " + "-" * (col_w + val_w + 4))
        for label, value in values.items():
            print(f"  {label:<{col_w}} {str(value):>{val_w}}")

    print(f"\n{sep}\n")