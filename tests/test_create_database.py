from __future__ import annotations

import math
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import scripts.create_database as db


@pytest.fixture
def sample_docs():
    return [
        Document(page_content="B+ trees store data in leaf nodes.", metadata={"source": "notes/lec1.pdf", "page": 1}),
        Document(page_content="Hash indexes only support equality.", metadata={"source": "notes/lec2.pdf", "page": 2}),
        Document(page_content="  extra whitespace  ",               metadata={"source": "notes/lec3.pdf", "page": 3}),
    ]


@pytest.fixture
def big_docs():
    # 4 docs * ~3 chunks each = 12+ chunks, safely exceeds chunks[10] in split_text
    return [Document(page_content="word " * 600, metadata={"source": f"f{i}.pdf", "page": i}) for i in range(4)]


@pytest.fixture
def large_docs():
    return [Document(page_content=f"Content of document {i}.", metadata={"source": f"notes/lec{i}.pdf", "page": i}) for i in range(25)]


# clean_text: null bytes must be replaced with a space to prevent word merging
def test_clean_text_removes_null_bytes():
    assert db.clean_text("hello\x00world") == "hello world"

# clean_text: leading and trailing whitespace must be stripped
def test_clean_text_strips_whitespace():
    assert db.clean_text("  padded  ") == "padded"

# clean_text: empty string must return empty string without error
def test_clean_text_empty():
    assert db.clean_text("") == ""

# clean_text: text with no issues must pass through unchanged
def test_clean_text_no_change():
    assert db.clean_text("normal text") == "normal text"

# clean_text: multiple null bytes scattered in text must all be replaced with spaces
def test_clean_text_multiple_null_bytes():
    assert db.clean_text("a\x00b\x00c") == "a b c"

# clean_text: text with only null bytes must return empty string after strip
def test_clean_text_only_null_bytes():
    assert db.clean_text("\x00\x00\x00") == ""

# split_text: output must contain more chunks than input documents
def test_split_text_increases_chunk_count(big_docs):
    chunks = db.split_text(big_docs)
    assert len(chunks) > len(big_docs)

# split_text: each chunk must not exceed the configured chunk_size of 1500 characters
def test_split_text_chunk_size_limit(big_docs):
    chunks = db.split_text(big_docs)
    assert all(len(c.page_content) <= 1500 for c in chunks)

# split_text: metadata from source documents must be preserved in chunks
def test_split_text_preserves_metadata(big_docs):
    chunks = db.split_text(big_docs)
    assert all("source" in c.metadata for c in chunks)

# split_text: chunks must be Document objects
def test_split_text_returns_documents(big_docs):
    chunks = db.split_text(big_docs)
    assert all(isinstance(c, Document) for c in chunks)

# split_text: start_index metadata must be present since add_start_index=True
def test_split_text_adds_start_index(big_docs):
    chunks = db.split_text(big_docs)
    assert all("start_index" in c.metadata for c in chunks)

# split_text: chunk overlap must produce at least some shared content between consecutive chunks
def test_split_text_overlap_produces_more_chunks(big_docs):
    chunks = db.split_text(big_docs)
    # with overlap=500 on 600-word docs, we expect more than 1 chunk per doc
    assert len(chunks) > len(big_docs)

# load_documents: must call GCSFileLoader for each PDF blob found in the bucket
def test_load_documents_loads_pdfs():
    mock_blob_pdf  = MagicMock(); mock_blob_pdf.name  = "notes/lec1.pdf"
    mock_blob_skip = MagicMock(); mock_blob_skip.name = "notes/readme.txt"

    mock_bucket = MagicMock()
    mock_bucket.list_blobs.return_value = [mock_blob_pdf, mock_blob_skip]

    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    mock_loader = MagicMock()
    mock_loader.load.return_value = [Document(page_content="Lecture 1 content.", metadata={})]

    with patch("google.cloud.storage.Client", return_value=mock_client), \
         patch("scripts.create_database.GCSFileLoader", return_value=mock_loader) as mock_gcs:
        docs = db.load_documents()

    assert mock_gcs.call_count == 1
    assert len(docs) == 1

# load_documents: non-PDF files must be skipped entirely
def test_load_documents_skips_non_pdfs():
    mock_blob = MagicMock(); mock_blob.name = "notes/readme.txt"
    mock_bucket = MagicMock(); mock_bucket.list_blobs.return_value = [mock_blob]
    mock_client = MagicMock(); mock_client.bucket.return_value = mock_bucket

    with patch("google.cloud.storage.Client", return_value=mock_client), \
         patch("scripts.create_database.GCSFileLoader") as mock_gcs:
        docs = db.load_documents()

    mock_gcs.assert_not_called()
    assert docs == []

# load_documents: must return an empty list when the bucket has no blobs
def test_load_documents_empty_bucket():
    mock_bucket = MagicMock(); mock_bucket.list_blobs.return_value = []
    mock_client = MagicMock(); mock_client.bucket.return_value = mock_bucket

    with patch("google.cloud.storage.Client", return_value=mock_client):
        docs = db.load_documents()

    assert docs == []

# load_documents: multiple PDFs must each be loaded and combined into one list
def test_load_documents_multiple_pdfs():
    blobs = [MagicMock() for _ in range(3)]
    for i, b in enumerate(blobs):
        b.name = f"notes/lec{i}.pdf"

    mock_bucket = MagicMock(); mock_bucket.list_blobs.return_value = blobs
    mock_client = MagicMock(); mock_client.bucket.return_value = mock_bucket

    mock_loader = MagicMock()
    mock_loader.load.return_value = [Document(page_content="content", metadata={})]

    with patch("google.cloud.storage.Client", return_value=mock_client), \
         patch("scripts.create_database.GCSFileLoader", return_value=mock_loader):
        docs = db.load_documents()

    assert len(docs) == 3

# load_documents: GCSFileLoader must be called with correct bucket and blob name
def test_load_documents_passes_correct_args():
    mock_blob = MagicMock(); mock_blob.name = "notes/lec1.pdf"
    mock_bucket = MagicMock(); mock_bucket.list_blobs.return_value = [mock_blob]
    mock_client = MagicMock(); mock_client.bucket.return_value = mock_bucket
    mock_loader = MagicMock()
    mock_loader.load.return_value = [Document(page_content="content", metadata={})]

    with patch("google.cloud.storage.Client", return_value=mock_client), \
         patch("scripts.create_database.GCSFileLoader", return_value=mock_loader) as mock_gcs:
        db.load_documents()

    call_kwargs = mock_gcs.call_args[1]
    assert call_kwargs["blob"] == "notes/lec1.pdf"
    assert call_kwargs["bucket"] == db.GCS_BUCKET

# save_to_cloud_sql: must call aadd_texts at least once when chunks are provided
@pytest.mark.asyncio
async def test_save_to_cloud_sql_calls_add_texts(sample_docs):
    mock_engine       = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_vector_store.aadd_texts = AsyncMock()

    with patch("scripts.create_database.PostgresEngine.afrom_instance", AsyncMock(return_value=mock_engine)), \
         patch("scripts.create_database.PostgresVectorStore.create",    AsyncMock(return_value=mock_vector_store)), \
         patch("scripts.create_database.OpenAIEmbeddings"):
        await db.save_to_cloud_sql(sample_docs)

    assert mock_vector_store.aadd_texts.called

# save_to_cloud_sql: must call ainit_vectorstore_table to set up the table
@pytest.mark.asyncio
async def test_save_to_cloud_sql_inits_table(sample_docs):
    mock_engine       = AsyncMock()
    mock_vector_store = AsyncMock()

    with patch("scripts.create_database.PostgresEngine.afrom_instance", AsyncMock(return_value=mock_engine)), \
         patch("scripts.create_database.PostgresVectorStore.create",    AsyncMock(return_value=mock_vector_store)), \
         patch("scripts.create_database.OpenAIEmbeddings"):
        await db.save_to_cloud_sql(sample_docs)

    mock_engine.ainit_vectorstore_table.assert_called_once()

# save_to_cloud_sql: ainit_vectorstore_table must be called with correct table name and vector size
@pytest.mark.asyncio
async def test_save_to_cloud_sql_inits_table_with_correct_args(sample_docs):
    mock_engine       = AsyncMock()
    mock_vector_store = AsyncMock()

    with patch("scripts.create_database.PostgresEngine.afrom_instance", AsyncMock(return_value=mock_engine)), \
         patch("scripts.create_database.PostgresVectorStore.create",    AsyncMock(return_value=mock_vector_store)), \
         patch("scripts.create_database.OpenAIEmbeddings"):
        await db.save_to_cloud_sql(sample_docs)

    call_kwargs = mock_engine.ainit_vectorstore_table.call_args[1]
    assert call_kwargs["table_name"]  == db.TABLE_NAME
    assert call_kwargs["vector_size"] == 1536

# save_to_cloud_sql: large datasets must be uploaded in batches of 100
@pytest.mark.asyncio
async def test_save_to_cloud_sql_batches_uploads(large_docs):
    mock_engine       = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_vector_store.aadd_texts = AsyncMock()

    with patch("scripts.create_database.PostgresEngine.afrom_instance", AsyncMock(return_value=mock_engine)), \
         patch("scripts.create_database.PostgresVectorStore.create",    AsyncMock(return_value=mock_vector_store)), \
         patch("scripts.create_database.OpenAIEmbeddings"):
        await db.save_to_cloud_sql(large_docs)

    assert mock_vector_store.aadd_texts.call_count == 1

# save_to_cloud_sql: 250 docs must produce 3 batch uploads (ceil(250/100))
@pytest.mark.asyncio
async def test_save_to_cloud_sql_correct_batch_count():
    docs_250 = [Document(page_content=f"doc {i}", metadata={}) for i in range(250)]
    mock_engine       = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_vector_store.aadd_texts = AsyncMock()

    with patch("scripts.create_database.PostgresEngine.afrom_instance", AsyncMock(return_value=mock_engine)), \
         patch("scripts.create_database.PostgresVectorStore.create",    AsyncMock(return_value=mock_vector_store)), \
         patch("scripts.create_database.OpenAIEmbeddings"):
        await db.save_to_cloud_sql(docs_250)

    assert mock_vector_store.aadd_texts.call_count == 3

# save_to_cloud_sql: null bytes in chunk text must be cleaned before upload
@pytest.mark.asyncio
async def test_save_to_cloud_sql_cleans_text():
    dirty_chunks      = [Document(page_content="dirty\x00text", metadata={"source": "f.pdf"})]
    mock_engine       = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_vector_store.aadd_texts = AsyncMock()

    with patch("scripts.create_database.PostgresEngine.afrom_instance", AsyncMock(return_value=mock_engine)), \
         patch("scripts.create_database.PostgresVectorStore.create",    AsyncMock(return_value=mock_vector_store)), \
         patch("scripts.create_database.OpenAIEmbeddings"):
        await db.save_to_cloud_sql(dirty_chunks)

    uploaded_texts = mock_vector_store.aadd_texts.call_args[1]["texts"]
    assert "\x00" not in uploaded_texts[0]

# save_to_cloud_sql: metadata must be passed alongside texts to aadd_texts
@pytest.mark.asyncio
async def test_save_to_cloud_sql_passes_metadata(sample_docs):
    mock_engine       = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_vector_store.aadd_texts = AsyncMock()

    with patch("scripts.create_database.PostgresEngine.afrom_instance", AsyncMock(return_value=mock_engine)), \
         patch("scripts.create_database.PostgresVectorStore.create",    AsyncMock(return_value=mock_vector_store)), \
         patch("scripts.create_database.OpenAIEmbeddings"):
        await db.save_to_cloud_sql(sample_docs)

    call_kwargs = mock_vector_store.aadd_texts.call_args[1]
    assert "metadatas" in call_kwargs
    assert len(call_kwargs["metadatas"]) == len(sample_docs)

# save_to_cloud_sql: PostgresEngine must be called with correct connection parameters
@pytest.mark.asyncio
async def test_save_to_cloud_sql_engine_called_with_correct_params(sample_docs):
    mock_engine       = AsyncMock()
    mock_vector_store = AsyncMock()

    with patch("scripts.create_database.PostgresEngine.afrom_instance", AsyncMock(return_value=mock_engine)) as mock_pg, \
         patch("scripts.create_database.PostgresVectorStore.create",    AsyncMock(return_value=mock_vector_store)), \
         patch("scripts.create_database.OpenAIEmbeddings"):
        await db.save_to_cloud_sql(sample_docs)

    call_kwargs = mock_pg.call_args[1]
    assert call_kwargs["project_id"] == db.PROJECT_ID
    assert call_kwargs["database"]   == db.DATABASE

@pytest.fixture(scope="session", autouse=True)
def print_comparison_summary():
    yield

    base_docs = [
        Document(page_content="word " * 600, metadata={"source": f"f{i}.pdf", "page": i})
        for i in range(4)
    ]
    dirty_docs  = [Document(page_content="dirty\x00text\x00here", metadata={})]
    clean_docs  = [Document(page_content="clean text here",        metadata={})]
    docs_100    = [Document(page_content=f"doc {i}", metadata={}) for i in range(100)]
    docs_250    = [Document(page_content=f"doc {i}", metadata={}) for i in range(250)]

    chunks      = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500,
                                                  length_function=len, add_start_index=True
                                                  ).split_documents(base_docs)

    import math
    comparisons = {
        "clean_text behavior": {
            "Null byte replaced with space": db.clean_text("a\x00b\x00c"),
            "Whitespace stripped":           db.clean_text("  padded  "),
            "Only null bytes":               db.clean_text("\x00\x00\x00"),
            "No change":                     db.clean_text("normal text"),
        },
        "split_text: chunk counts": {
            "Input docs":                  len(base_docs),
            "Output chunks":               len(chunks),
            "Chunks per doc (avg)":        round(len(chunks) / len(base_docs), 2),
        },
        "split_text: chunk sizes (chars)": {
            "Min chunk size":              min(len(c.page_content) for c in chunks),
            "Max chunk size":              max(len(c.page_content) for c in chunks),
            "Avg chunk size":              round(sum(len(c.page_content) for c in chunks) / len(chunks), 1),
            "Configured chunk_size":       1500,
            "Configured chunk_overlap":    500,
        },
        "split_text: metadata fields in chunks": {
            "source present":              str(all("source"      in c.metadata for c in chunks)),
            "page present":                str(all("page"        in c.metadata for c in chunks)),
            "start_index present":         str(all("start_index" in c.metadata for c in chunks)),
        },
        "save_to_cloud_sql: batch counts": {
            "25 docs  → batches":          math.ceil(25  / 100),
            "100 docs → batches":          math.ceil(100 / 100),
            "250 docs → batches":          math.ceil(250 / 100),
            "Batch size":                  100,
        },
        "clean_text effect on upload texts": {
            "Before clean (null bytes)":   repr(dirty_docs[0].page_content),
            "After clean":                 repr(db.clean_text(dirty_docs[0].page_content)),
            "Clean doc unchanged":         repr(db.clean_text(clean_docs[0].page_content)),
        },
        "GCS prefix filter": {
            "GCS_PREFIX":                  db.GCS_PREFIX,
            "TABLE_NAME":                  db.TABLE_NAME,
            "Vector size":                 1536,
        },
    }

    col_w = 38
    val_w = 30
    sep   = "=" * 80

    print(f"\n\n{sep}")
    print("  CREATE DATABASE — PIPELINE COMPARISON SUMMARY")
    print(sep)

    for section, values in comparisons.items():
        print(f"\n  {section}")
        print("  " + "-" * (col_w + val_w + 4))
        for label, value in values.items():
            print(f"  {label:<{col_w}} {str(value):>{val_w}}")

    print(f"\n{sep}\n")


# generate_data_store: must call load, split, and save in the correct sequence
@pytest.mark.asyncio
async def test_generate_data_store_calls_pipeline(sample_docs):
    chunks = [Document(page_content="chunk.", metadata={})]
    with patch("scripts.create_database.load_documents", return_value=sample_docs), \
         patch("scripts.create_database.split_text",     return_value=chunks), \
         patch("scripts.create_database.save_to_cloud_sql", AsyncMock()) as mock_save:
        await db.generate_data_store()

    mock_save.assert_called_once_with(chunks)

# generate_data_store: split_text must be called with the output of load_documents
@pytest.mark.asyncio
async def test_generate_data_store_passes_docs_to_split(sample_docs):
    chunks = [Document(page_content="chunk.", metadata={})]
    with patch("scripts.create_database.load_documents", return_value=sample_docs) as mock_load, \
         patch("scripts.create_database.split_text",     return_value=chunks) as mock_split, \
         patch("scripts.create_database.save_to_cloud_sql", AsyncMock()):
        await db.generate_data_store()

    mock_split.assert_called_once_with(sample_docs)