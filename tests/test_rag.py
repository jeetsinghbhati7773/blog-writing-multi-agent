from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
import pytest

from src.rag.loader import load_document_bytes
from src.rag.splitter import split_documents
from src.rag.vectorstore import ChromaVectorStore
from src.rag.retriever import DocumentRetriever, format_source_label


@pytest.fixture
def temp_db_dir():
    temp_dir = tempfile.mkdtemp(prefix="test_chroma_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_load_txt():
    txt_content = b"This is a technical documentation file about LangGraph and LangChain."
    docs = load_document_bytes(txt_content, "sample.txt")
    assert len(docs) == 1
    assert "LangGraph" in docs[0]["page_content"]
    assert docs[0]["metadata"]["filename"] == "sample.txt"
    assert docs[0]["metadata"]["file_type"] == "txt"
    assert docs[0]["metadata"]["page"] == 1


def test_load_md():
    md_content = b"# Architecture Overview\n\n- Component 1: RAG\n- Component 2: VectorDB"
    docs = load_document_bytes(md_content, "arch.md")
    assert len(docs) == 1
    assert docs[0]["metadata"]["file_type"] == "md"
    assert "Component 1" in docs[0]["page_content"]


def test_load_unsupported_file():
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_document_bytes(b"some bytes", "file.xyz")


def test_load_empty_file():
    with pytest.raises(ValueError, match="empty"):
        load_document_bytes(b"", "empty.txt")


def test_split_documents():
    long_text = "Word " * 500  # long text
    docs = [{
        "page_content": long_text,
        "metadata": {"filename": "long.txt", "file_type": "txt", "page": 1}
    }]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=30)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["page_content"]) <= 300
        assert chunk["metadata"]["filename"] == "long.txt"


def test_chroma_vectorstore(temp_db_dir):
    vs = ChromaVectorStore(db_path=temp_db_dir, collection_name="test_docs")
    assert vs.count_chunks() == 0

    chunks = [
        {
            "page_content": "Transformers use self-attention mechanism to process input sequences in parallel.",
            "metadata": {"filename": "transformers.pdf", "file_type": "pdf", "page": 12, "chunk_index": 0}
        },
        {
            "page_content": "Recurrent neural networks process sequential data timestep by timestep.",
            "metadata": {"filename": "rnn_overview.docx", "file_type": "docx", "page": 1, "chunk_index": 0}
        }
    ]

    added = vs.add_documents(chunks)
    assert added == 2
    assert vs.count_chunks() == 2

    # Query
    results = vs.query("What mechanism do Transformers use?", n_results=1)
    assert len(results) == 1
    assert "self-attention" in results[0]["content"]
    assert results[0]["metadata"]["filename"] == "transformers.pdf"

    # List documents
    doc_list = vs.list_documents()
    assert len(doc_list) == 2
    filenames = [d["filename"] for d in doc_list]
    assert "transformers.pdf" in filenames
    assert "rnn_overview.docx" in filenames

    # Delete single document
    deleted = vs.delete_document("rnn_overview.docx")
    assert deleted is True
    assert vs.count_chunks() == 1

    # Clear documents
    vs.clear_documents()
    assert vs.count_chunks() == 0


def test_retriever_source_formatting():
    meta_pdf = {"filename": "paper.pdf", "file_type": "pdf", "page": 5}
    meta_txt = {"filename": "notes.txt", "file_type": "txt", "page": 1}

    assert format_source_label(meta_pdf) == "📄 paper.pdf — Page 5"
    assert format_source_label(meta_txt) == "📄 notes.txt"


def test_retriever_empty_answer(temp_db_dir):
    vs = ChromaVectorStore(db_path=temp_db_dir, collection_name="test_empty")
    retriever = DocumentRetriever(vectorstore=vs)

    answer, sources, raw_chunks = retriever.answer_question("How does quantum computing work?", mode="strict")
    assert "not contain enough information" in answer
    assert sources == []
    assert raw_chunks == []


def test_retriever_build_search_query(temp_db_dir):
    vs = ChromaVectorStore(db_path=temp_db_dir, collection_name="test_query_builder")
    retriever = DocumentRetriever(vectorstore=vs)

    # Empty history
    query1 = retriever.build_search_query("Explain self-attention")
    assert query1 == "Explain self-attention"

    # Multi-turn history with short query
    history = [
        {"role": "user", "content": "Tell me about Transformer models in deep learning."},
        {"role": "assistant", "content": "Transformers use self-attention mechanism."},
    ]
    query2 = retriever.build_search_query("What is it?", chat_history=history)
    assert "Transformer models" in query2


def test_retriever_build_prompt_messages(temp_db_dir):
    vs = ChromaVectorStore(db_path=temp_db_dir, collection_name="test_prompt_builder")
    retriever = DocumentRetriever(vectorstore=vs)

    chunks = [{
        "content": "Self-attention computes dynamic weights across input tokens.",
        "metadata": {"filename": "paper.pdf", "file_type": "pdf", "page": 3}
    }]
    history = [
        {"role": "user", "content": "What are neural networks?"},
        {"role": "assistant", "content": "Neural networks are computational models."},
    ]

    messages_strict = retriever.build_prompt_messages("Explain self-attention", history, chunks, mode="strict")
    assert len(messages_strict) >= 3
    assert "DOCUMENT CONTEXT" in messages_strict[0].content
    assert "paper.pdf — Page 3" in messages_strict[0].content

    messages_general = retriever.build_prompt_messages("Hello!", history, [], mode="general")
    assert "helpful, expert technical AI assistant" in messages_general[0].content


def test_embedding_determinism():
    from src.rag.embeddings import LightweightEmbeddingFunction
    embedder = LightweightEmbeddingFunction(dim=384)
    vec1 = embedder._embed_text("Deep Learning and Neural Networks")
    vec2 = embedder._embed_text("Deep Learning and Neural Networks")

    assert len(vec1) == 384
    assert vec1 == vec2  # 100% deterministic across process runs


def test_cosine_similarity_precision():
    from src.rag.vectorstore import _cosine_similarity
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [1.0, 0.0, 0.0]
    vec_c = [0.0, 1.0, 0.0]

    assert abs(_cosine_similarity(vec_a, vec_b) - 1.0) < 1e-6
    assert abs(_cosine_similarity(vec_a, vec_c) - 0.0) < 1e-6
    assert _cosine_similarity([], []) == 0.0


