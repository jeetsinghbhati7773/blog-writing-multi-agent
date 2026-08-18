from __future__ import annotations

import pandas as pd
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.agent.nodes import get_llm
from src.rag.loader import load_document_bytes
from src.rag.splitter import split_documents
from src.rag.vectorstore import ChromaVectorStore
from src.rag.retriever import DocumentRetriever


def render_unified_chat_view():
    """
    Renders a unified multi-turn AI Chat & PDF/Document RAG Assistant.
    Combines multi-turn conversation memory with grounded document context, source citations, and streaming responses.
    """
    st.subheader("💬 DOCUMENT INTELLIGENCE")
    st.caption("Ask questions about your uploaded knowledge base or general technical topics.")

    vectorstore = ChromaVectorStore()
    retriever = DocumentRetriever(vectorstore=vectorstore)

    # -----------------------------
    # 1. Document Management & Uploader Drawer
    # -----------------------------
    doc_count = vectorstore.count_chunks()
    doc_list = vectorstore.list_documents()

    expander_title = f"📂 **Document Memory ({len(doc_list)} files, {doc_count} chunks)**" if doc_list else "📂 **Upload PDFs & Documents**"
    with st.expander(expander_title, expanded=(doc_count == 0)):
        uploaded_files = st.file_uploader(
            "Upload Document(s)",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, Markdown (.md), DOCX",
            key="unified_file_uploader",
        )

        if uploaded_files:
            if st.button("📥 Process & Index Uploaded Files", type="primary", key="unified_process_btn"):
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
        if doc_list:
            st.markdown("#### **Indexed Documents in ChromaDB**")
            col_docs, col_clear = st.columns([5, 2])
            with col_docs:
                df = pd.DataFrame(doc_list)
                st.dataframe(df, use_container_width=True, hide_index=True)
            with col_clear:
                st.write("")
                st.write("")
                if st.button("🗑️ Clear Document Memory", type="secondary", use_container_width=True, key="unified_clear_docs_btn"):
                    vectorstore.clear_documents()
                    st.toast("Cleared ChromaDB document memory!", icon="🧹")
                    st.rerun()
        else:
            st.info("No documents currently indexed. Upload PDF, TXT, MD, or DOCX files above to enable grounded RAG Q&A.")

    # -----------------------------
    # 2. Chat Controls & Search Mode Selector
    # -----------------------------
    col_mode, col_clear_chat = st.columns([4, 1.5])
    with col_mode:
        rag_mode = st.radio(
            "Retriever Mode:",
            ["🔀 Hybrid (Docs + AI)", "📚 Strict Document RAG", "🤖 General AI Chat"],
            index=0 if doc_count > 0 else 2,
            horizontal=True,
            help="Hybrid: prioritizes uploaded documents + AI. Strict: answers strictly from PDF context. General: AI chat with connected MCP tools & capabilities.",
            key="unified_rag_mode_radio",
        )
    with col_clear_chat:
        st.write("")
        if st.button("🗑️ Clear Chat History", help="Clear Chat History", use_container_width=True, key="unified_clear_chat_btn"):
            st.session_state["unified_messages"] = []
            st.rerun()

    st.divider()

    # Map mode string to code key
    mode_key = "hybrid"
    if "Strict" in rag_mode:
        mode_key = "strict"
    elif "General" in rag_mode:
        mode_key = "general"

    # Session State Initialization
    if "unified_messages" not in st.session_state:
        st.session_state["unified_messages"] = []

    # Auto-load MCP tools for General AI & Hybrid modes
    mcp_tools_list = []
    if mode_key in ("general", "hybrid"):
        try:
            from src.mcp_client import get_all_tools, run_async
            if "mcp_tools_cache" not in st.session_state or not st.session_state["mcp_tools_cache"]:
                st.session_state["mcp_tools_cache"] = get_all_tools()
            mcp_tools_list = st.session_state["mcp_tools_cache"]
            for t in mcp_tools_list:
                if hasattr(t, "coroutine") and t.coroutine and not getattr(t, "func", None):
                    coro = t.coroutine
                    t.func = lambda *args, _c=coro, **kwargs: run_async(_c(*args, **kwargs))
        except Exception:
            mcp_tools_list = []

    # -----------------------------
    # 3. Display Past Messages & Citations
    # -----------------------------
    for msg in st.session_state["unified_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption("📌 **Cited Sources:** " + ", ".join([f"`{s}`" for s in msg["sources"]]))
            if msg.get("chunks"):
                with st.expander("🔍 View Retrieved Document Chunks"):
                    for c in msg["chunks"]:
                        meta = c.get("metadata", {})
                        st.caption(f"**From:** `{meta.get('filename')}` (Page {meta.get('page', 1)}) — Relevance Distance: `{c.get('distance', 0.0):.4f}`")
                        st.text(c.get("content", ""))
                        st.divider()

    # -----------------------------
    # 4. Chat Input & Streaming Assistant Execution
    # -----------------------------
    placeholder_text = "Ask a question about your PDFs or request actions (e.g., 'Add expense $45 for lunch')..." if doc_count > 0 else "Ask anything or use tools (e.g. 'Add $3000 medical expense')..."
    if prompt := st.chat_input(placeholder_text):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant chunks if mode is not pure general chat
        raw_chunks = []
        sources = []
        if mode_key != "general" and doc_count > 0:
            raw_chunks = retriever.retrieve_relevant_chunks(
                prompt, k=5, chat_history=st.session_state["unified_messages"]
            )
            sources = retriever.format_sources(raw_chunks)

        # Check strict RAG empty condition
        if mode_key == "strict" and not raw_chunks:
            answer = "The uploaded documents do not contain enough information to answer this question. (No documents uploaded or no relevant matches found in memory)."
            with st.chat_message("assistant"):
                st.warning(answer)
            st.session_state["unified_messages"].append({"role": "user", "content": prompt})
            st.session_state["unified_messages"].append({"role": "assistant", "content": answer, "sources": [], "chunks": []})
            return

        # Prepare messages
        if mode_key == "general" and mcp_tools_list:
            tool_names_str = ", ".join([f"`{t.name}`" for t in mcp_tools_list])
            mcp_system = (
                f"You are an active AI Assistant equipped with direct executable tools: {tool_names_str}.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. If the user asks to add, record, calculate, query, web-search, or list expenses, stocks, web scores, or data, YOU MUST CALL THE APPROPRIATE TOOL FROM THE LIST ABOVE IMMEDIATELY.\n"
                "2. ONLY call tools that are strictly present in the list above.\n"
                "3. NEVER output spreadsheet instructions, Mint/YNAB/PocketGuard guides, SQL queries, Python scripts, Markdown tables, or DIY guides.\n"
                "4. Keep all text responses extremely brief, direct, and concise (1-2 sentences maximum)."
            )
            messages = [SystemMessage(content=mcp_system)]
            for msg in st.session_state["unified_messages"][-10:]:
                r_role = msg.get("role")
                r_content = msg.get("content", "")
                if r_role == "user":
                    messages.append(HumanMessage(content=r_content))
                elif r_role == "assistant":
                    messages.append(AIMessage(content=r_content))
            if not st.session_state["unified_messages"] or st.session_state["unified_messages"][-1].get("content") != prompt:
                messages.append(HumanMessage(content=prompt))
        else:
            messages = retriever.build_prompt_messages(
                query=prompt,
                chat_history=st.session_state["unified_messages"],
                chunks=raw_chunks,
                mode=mode_key,
            )

        st.session_state["unified_messages"].append({"role": "user", "content": prompt})

        # Generate streaming assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                llm = get_llm()
                if mcp_tools_list and mode_key in ("general", "hybrid"):
                    llm_with_tools = llm.bind_tools(mcp_tools_list)
                    response_msg = llm_with_tools.invoke(messages)

                    # Handle Tool Calls if any
                    if hasattr(response_msg, "tool_calls") and response_msg.tool_calls:
                        tool_map = {t.name: t for t in mcp_tools_list}
                        tool_outputs = []
                        for tc in response_msg.tool_calls:
                            t_name = tc.get("name")
                            t_args = tc.get("args", {})
                            with st.status(f"🔧 Executing Tool `{t_name}`...", expanded=True) as status_box:
                                st.write(f"**Arguments:** `{t_args}`")
                                if t_name in tool_map:
                                    from src.mcp_client import execute_tool
                                    t_res = execute_tool(tool_map[t_name], t_args)
                                    st.write(f"**Result:** `{t_res}`")
                                    tool_outputs.append(f"Tool `{t_name}` output: {t_res}")
                                    status_box.update(label=f"✅ Tool `{t_name}` Completed", state="complete", expanded=False)
                                else:
                                    status_box.update(label=f"❌ Tool `{t_name}` Not Found", state="error")

                        # Synthesize concise 1-sentence response
                        synthesis_prompt = (
                            f"Tool Execution Output:\n" + "\n".join(tool_outputs) + "\n\n"
                            f"System Instruction: Output ONLY a single, short 1-sentence confirmation of the result for user request: '{prompt}'. "
                            f"STRICTLY FORBIDDEN: Do NOT output markdown tables, spreadsheets, Mint/YNAB guides, python scripts, step-by-step DIY guides, or unprompted follow-up options."
                        )
                        full_response = llm.invoke([SystemMessage(content=synthesis_prompt)]).content.strip()
                        message_placeholder.markdown(full_response)
                    else:
                        full_response = response_msg.content.strip() if hasattr(response_msg, "content") else str(response_msg)
                        message_placeholder.markdown(full_response)
                else:
                    for chunk in llm.stream(messages):
                        content = chunk.content if hasattr(chunk, "content") else str(chunk)
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"Error generating response: {e}"
                message_placeholder.error(full_response)

            if sources:
                st.caption("📌 **Cited Sources:** " + ", ".join([f"`{s}`" for s in sources]))
            if raw_chunks:
                with st.expander("🔍 View Retrieved Document Chunks"):
                    for c in raw_chunks:
                        meta = c.get("metadata", {})
                        st.caption(f"**From:** `{meta.get('filename')}` (Page {meta.get('page', 1)}) — Relevance Distance: `{c.get('distance', 0.0):.4f}`")
                        st.text(c.get("content", ""))
                        st.divider()

        st.session_state["unified_messages"].append({
            "role": "assistant",
            "content": full_response,
            "sources": sources,
            "chunks": raw_chunks,
        })


def render_ai_chat_view():
    """
    Backwards compatible alias for AI Chat mode.
    """
    render_unified_chat_view()


def render_document_chat_view():
    """
    Backwards compatible alias for Document Chat mode.
    """
    render_unified_chat_view()

