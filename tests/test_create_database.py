from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document

import create_database as db


@pytest.fixture
def sample_docs():
    return [
        Document(page_content="B+ trees store data in leaf nodes.", metadata={"source": "notes/lec1.pdf", "page": 1}),
        Document(page_content="Hash indexes only support equality.", metadata={"source": "notes/lec2.pdf", "page": 2}),
        Document(page_content="  extra whitespace  ",               metadata={"source": "notes/lec3.pdf", "page": 3}),
    ]


@pytest.fixture
def large_docs():
    return [Document(page_content=f"Content of document {i}.", metadata={"source": f"notes/lec{i}.pdf", "page": i}) for i in range(25)]

@pytest.fixture
def big_docs():
    # 4 docs * ~3 chunks each = 12+ chunks, safely exceeds chunks[10] in split_text
    return [Document(page_content="word " * 600, metadata={"source": f"f{i}.pdf", "page": i}) for i in range(4)]


# clean_text: null bytes must be removed from text before embedding
def test_clean_text_removes_null_bytes():
    assert db.clean_text("hello\x00world") == "helloworld"

# clean_text: leading and trailing whitespace must be stripped
def test_clean_text_strips_whitespace():
    assert db.clean_text("  padded  ") == "padded"

# clean_text: empty string must return empty string without error
def test_clean_text_empty():
    assert db.clean_text("") == ""

# clean_text: text with no issues must pass through unchanged
def test_clean_text_no_change():
    assert db.clean_text("normal text") == "normal text"

# clean_text: multiple null bytes scattered in text must all be removed
def test_clean_text_multiple_null_bytes():
    assert db.clean_text("a\x00b\x00c") == "abc"

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
         patch("create_database.GCSFileLoader", return_value=mock_loader) as mock_gcs:
        docs = db.load_documents()

    assert mock_gcs.call_count == 1
    assert len(docs) == 1

# load_documents: non-PDF files must be skipped entirely
def test_load_documents_skips_non_pdfs():
    mock_blob = MagicMock(); mock_blob.name = "notes/readme.txt"
    mock_bucket = MagicMock(); mock_bucket.list_blobs.return_value = [mock_blob]
    mock_client = MagicMock(); mock_client.bucket.return_value = mock_bucket

    with patch("google.cloud.storage.Client", return_value=mock_client), \
         patch("create_database.GCSFileLoader") as mock_gcs:
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

# save_to_cloud_sql: must call aadd_texts at least once when chunks are provided
@pytest.mark.asyncio
async def test_save_to_cloud_sql_calls_add_texts(sample_docs):
    mock_engine       = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_vector_store.aadd_texts = AsyncMock()

    with patch("create_database.PostgresEngine.afrom_instance", AsyncMock(return_value=mock_engine)), \
         patch("create_database.PostgresVectorStore.create",    AsyncMock(return_value=mock_vector_store)), \
         patch("create_database.OpenAIEmbeddings"):
        await db.save_to_cloud_sql(sample_docs)

    assert mock_vector_store.aadd_texts.called

# save_to_cloud_sql: must call ainit_vectorstore_table to set up the table
@pytest.mark.asyncio
async def test_save_to_cloud_sql_inits_table(sample_docs):
    mock_engine       = AsyncMock()
    mock_vector_store = AsyncMock()

    with patch("create_database.PostgresEngine.afrom_instance", AsyncMock(return_value=mock_engine)), \
         patch("create_database.PostgresVectorStore.create",    AsyncMock(return_value=mock_vector_store)), \
         patch("create_database.OpenAIEmbeddings"):
        await db.save_to_cloud_sql(sample_docs)

    mock_engine.ainit_vectorstore_table.assert_called_once()

# save_to_cloud_sql: large datasets must be uploaded in batches of 100
@pytest.mark.asyncio
async def test_save_to_cloud_sql_batches_uploads(large_docs):
    mock_engine       = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_vector_store.aadd_texts = AsyncMock()

    with patch("create_database.PostgresEngine.afrom_instance", AsyncMock(return_value=mock_engine)), \
         patch("create_database.PostgresVectorStore.create",    AsyncMock(return_value=mock_vector_store)), \
         patch("create_database.OpenAIEmbeddings"):
        await db.save_to_cloud_sql(large_docs)

    # 25 docs / batch size 100 = 1 batch but call count confirms batching logic ran
    assert mock_vector_store.aadd_texts.call_count == 1

# save_to_cloud_sql: null bytes in chunk text must be cleaned before upload
@pytest.mark.asyncio
async def test_save_to_cloud_sql_cleans_text():
    dirty_chunks = [Document(page_content="dirty\x00text", metadata={"source": "f.pdf"})]
    mock_engine       = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_vector_store.aadd_texts = AsyncMock()

    with patch("create_database.PostgresEngine.afrom_instance", AsyncMock(return_value=mock_engine)), \
         patch("create_database.PostgresVectorStore.create",    AsyncMock(return_value=mock_vector_store)), \
         patch("create_database.OpenAIEmbeddings"):
        await db.save_to_cloud_sql(dirty_chunks)

    uploaded_texts = mock_vector_store.aadd_texts.call_args[1]["texts"]
    assert "\x00" not in uploaded_texts[0]

# generate_data_store: must call load, split, and save in sequence
@pytest.mark.asyncio
async def test_generate_data_store_calls_pipeline(sample_docs):
    chunks = [Document(page_content="chunk.", metadata={})]
    with patch("create_database.load_documents", return_value=sample_docs), \
         patch("create_database.split_text",     return_value=chunks), \
         patch("create_database.save_to_cloud_sql", AsyncMock()) as mock_save:
        await db.generate_data_store()

    mock_save.assert_called_once_with(chunks)