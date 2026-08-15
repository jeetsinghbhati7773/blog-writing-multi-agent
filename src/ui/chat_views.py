from __future__ import annotations

import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.agent.nodes import get_llm
from src.rag.loader import load_document_bytes
from src.rag.splitter import split_documents
from src.rag.vectorstore import ChromaVectorStore
from src.rag.retriever import DocumentRetriever


def render_ai_chat_view():
    """
    Renders normal multi-turn AI Chat mode using Streamlit session state.
    """
    st.subheader("💬 AI Chat")
    st.caption("Multi-turn technical conversation powered by LLM.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Chat controls
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Clear", help="Clear Chat History", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

    # Display past messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything... (e.g. Explain LangGraph vs LangChain)"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state["messages"].append({"role": "user", "content": prompt})

        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Build message history for LLM
            history_messages = [
                SystemMessage(content="You are a helpful, expert technical AI assistant. Provide clear, accurate, and structured answers.")
            ]
            for m in st.session_state["messages"]:
                if m["role"] == "user":
                    history_messages.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    history_messages.append(AIMessage(content=m["content"]))

            try:
                llm = get_llm()
                # Stream response
                for chunk in llm.stream(history_messages):
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"Error generating response: {e}"
                message_placeholder.error(full_response)

        st.session_state["messages"].append({"role": "assistant", "content": full_response})


def render_document_chat_view():
    """
    Renders Document Chat (RAG) mode using persistent ChromaDB storage.
    """
    st.subheader("📚 Document Chat (ChromaDB RAG)")
    st.caption("Upload PDF, TXT, MD, or DOCX documents and ask questions grounded strictly in your files.")

    vectorstore = ChromaVectorStore()
    retriever = DocumentRetriever(vectorstore=vectorstore)

    # -----------------------------
    # Document Upload & Management Section
    # -----------------------------
    with st.expander("📂 **Document Management & Uploader**", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload Document(s)",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, Markdown (.md), DOCX",
        )

        if uploaded_files:
            if st.button("📥 Process & Index Uploaded Files", type="primary"):
                total_chunks = 0
                processed_files = 0
                for f in uploaded_files:
                    try:
                        file_bytes = f.read()
                        filename = f.name
                        docs = load_document_bytes(file_bytes, filename)
                        chunks = split_documents(docs)
                        added = vectorstore.add_documents(chunks)
                        total_chunks += added
                        processed_files += 1
                        st.toast(f"✅ Indexed {filename} ({added} chunks)", icon="📄")
                    except Exception as e:
                        st.error(f"Failed to process '{f.name}': {e}")

                if processed_files > 0:
                    st.success(f"Successfully processed {processed_files} document(s) into ChromaDB ({total_chunks} total chunks).")
                    st.rerun()

        # Display uploaded document memory summary
        doc_list = vectorstore.list_documents()
        if doc_list:
            st.markdown("#### **Uploaded Documents in Memory**")
            col_docs, col_clear = st.columns([5, 2])
            with col_docs:
                import pandas as pd
                df = pd.DataFrame(doc_list)
                st.dataframe(df, use_container_width=True, hide_index=True)
            with col_clear:
                st.write("")
                st.write("")
                if st.button("🗑️ Clear Document Memory", type="secondary", use_container_width=True):
                    vectorstore.clear_documents()
                    st.toast("Cleared ChromaDB document memory!", icon="🧹")
                    st.rerun()
        else:
            st.info("No documents currently stored in ChromaDB memory. Upload documents above to start.")

    st.divider()

    # -----------------------------
    # Document Chat Session State & History
    # -----------------------------
    if "doc_messages" not in st.session_state:
        st.session_state["doc_messages"] = []

    # Chat controls header
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Clear Chat", help="Clear Document Chat History", use_container_width=True):
            st.session_state["doc_messages"] = []
            st.rerun()

    # Display past document chat messages
    for msg in st.session_state["doc_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.markdown("**Sources:**")
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

    # Document Chat Input
    if prompt := st.chat_input("Ask a question about your uploaded documents..."):
        # Render user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state["doc_messages"].append({"role": "user", "content": prompt, "sources": []})

        # Answer question using RAG retriever
        with st.chat_message("assistant"):
            with st.spinner("Searching document memory & synthesizing answer..."):
                answer, sources, raw_chunks = retriever.answer_question(prompt, k=5)
                st.markdown(answer)

                if sources:
                    st.markdown("**Sources:**")
                    for s in sources:
                        st.markdown(f"- {s}")

                with st.expander("🔍 View Retrieved Document Chunks"):
                    for c in raw_chunks:
                        st.caption(f"**From:** {c.get('metadata',{}).get('filename')} (Page {c.get('metadata',{}).get('page', 1)})")
                        st.text(c.get("content", ""))
                        st.divider()

        st.session_state["doc_messages"].append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
