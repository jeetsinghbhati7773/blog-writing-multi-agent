from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from src.agent.nodes import get_llm
from src.rag.vectorstore import ChromaVectorStore


def format_source_label(meta: Dict[str, Any]) -> str:
    """
    Formats metadata into clean human-readable source citations.
    Example: 📄 transformers.pdf — Page 12
    """
    filename = meta.get("filename", "Document")
    page = meta.get("page")
    if page and str(page) not in ("None", "0", "") and meta.get("file_type") == "pdf":
        return f"📄 {filename} — Page {page}"
    return f"📄 {filename}"


class DocumentRetriever:
    """
    Retriever module for performing similarity queries and generating grounded answers using ChromaDB context.
    """

    def __init__(self, vectorstore: Optional[ChromaVectorStore] = None):
        self.vectorstore = vectorstore or ChromaVectorStore()

    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top k relevant document chunks from ChromaDB for the given query.
        """
        return self.vectorstore.query(query, n_results=k)

    def format_sources(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Extracts deduplicated source citations from retrieved chunks.
        """
        sources = []
        seen = set()
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            label = format_source_label(meta)
            if label not in seen:
                seen.add(label)
                sources.append(label)
        return sources

    def answer_question(self, query: str, k: int = 5) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """
        Answers a user question based strictly on retrieved document chunks.
        Returns (answer_text, formatted_sources_list, raw_retrieved_chunks).
        """
        chunks = self.retrieve_relevant_chunks(query, k=k)
        sources = self.format_sources(chunks)

        if not chunks:
            return (
                "The uploaded documents do not contain enough information to answer this question. (No documents uploaded or no relevant matches found in memory).",
                [],
                [],
            )

        context_blocks = []
        for i, c in enumerate(chunks):
            meta = c.get("metadata", {})
            src_label = format_source_label(meta)
            context_blocks.append(f"[{src_label}]\n{c['content']}")

        context_str = "\n\n---\n\n".join(context_blocks)

        system_prompt = """You are a strict, helpful AI assistant answering questions about user-uploaded documents.

CRITICAL INSTRUCTIONS:
- Base your answer STRICTLY on the provided Document Context below.
- Do NOT introduce outside facts or invent document details not found in the context.
- If the provided context does not contain enough information to answer the question, clearly state:
  "The uploaded documents do not contain enough information to answer this question."
- Be concise, clear, accurate, and directly address the user's prompt.
"""

        user_prompt = f"Document Context:\n{context_str}\n\nUser Question: {query}"

        try:
            llm = get_llm()
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            answer = response.content.strip()
        except Exception as e:
            answer = f"Error generating answer from LLM: {e}"

        return answer, sources, chunks
