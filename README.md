# AI Blog Writing Agent — Multi-Agent Content Generation with LangGraph & HITL

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3-green.svg)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)
[![Pytest Passed](https://img.shields.io/badge/Tests-28%2F28%20Passed-brightgreen.svg)](tests/test_agent.py)

An autonomous, multi-agent AI system designed to research, filter, plan, validate, draft, fact-check, edit, optimize SEO, and generate visual assets for comprehensive, production-grade technical blog articles. Features **Dual Human-in-the-Loop (HITL)** checkpoints for plan approval and final content review. Built with **LangGraph**, **LangChain**, **Groq LLM**, and **Tavily Web Search**.

---

## 💡 Problem Statement

Writing deep technical blog posts requires significant manual effort: conducting web research, filtering source authority, resolving conflicting claims, organizing structured outlines, drafting individual sections, ensuring factual accuracy, avoiding repetition, optimizing SEO, and generating architecture diagrams. Single-prompt LLM generation often leads to hallucinations, repetitive prose, shallow coverage, and token context overflow on long articles.

---

## 🚀 Solution & Architecture

**AI Blog Writing Agent** solves this by orchestrating a specialized graph of multi-agent state nodes with human checkpoints:

```mermaid
flowchart TD
    Start([User Input Brief & Style Preset]) --> Router[Router Agent]
    
    Router -->|Needs Web Research| Research[Research Agent - Tavily]
    Router -->|Evergreen / Closed-Book| Orchestrator[Orchestrator / Planner]
    
    Research --> SourceFilter[Source Quality Filter]
    SourceFilter --> Contradiction[Contradiction Detector]
    Contradiction --> StructuredResearch[Structured Research Synthesizer]
    StructuredResearch --> Orchestrator
    
    Orchestrator --> OutlineValidator[Outline Quality Validator]
    OutlineValidator --> HITL1{🧑 HITL #1: Plan Approval}
    
    HITL1 -->|Approve| WorkerFanout[Parallel Worker Agents]
    HITL1 -->|Modify / Regenerate| Orchestrator
    
    WorkerFanout --> MergeContent[Merge & Reducer Node]
    MergeContent --> FactChecker[Fact & Citation Checker]
    FactChecker --> Critic[🧐 Critic / Quality Agent]
    
    Critic -->|Quality Pass| SEOAgent[SEO Optimization Agent]
    Critic -->|Needs Revision| Revision[Revision Agent] --> Critic
    
    SEOAgent --> HITL2{🧑 HITL #2: Final Content Review}
    
    HITL2 -->|Approve| Visual[Visual & Diagram Agent]
    HITL2 -->|Request Revision| Revision
    
    Visual --> Final([Published Article & Assets])
```

---

## 🌟 Key Features

1. **🧑 Dual Human-in-the-Loop (HITL) Checkpoints**:
   - **HITL Checkpoint #1 (Plan Review)**: Pauses execution after the Orchestrator & Outline Validator so you can review, modify, or regenerate the outline before section generation.
   - **HITL Checkpoint #2 (Final Content Review)**: Pauses after section workers, Critic, Fact Checker, and SEO Agent complete so you can inspect the full draft, SEO metadata, and fact check scores before publication.

2. **🧐 Autonomous Critic & Quality Audit Loop**:
   - Evaluates drafts across 7 quality dimensions (*factual consistency, relevance, completeness, readability, repetition, structure, grounding*).
   - Automatically triggers a `revision_node` loop if quality score is below threshold (< 7.5 / 10).

3. **🛡️ Fact & Citation Grounding Checker**:
   - Cross-references draft claims against retrieved research evidence sources, calculating a 0–10 factual grounding score and claim verification table.

4. **🔎 Source Quality Filter & Contradiction Detector**:
   - `source_filter_node`: Scores search results by authority, relevance, freshness, and deduplication before planning.
   - `contradiction_detector_node`: Identifies conflicting claims across web sources and provides resolution guidance to the Orchestrator.

5. **🎨 6 Article Style Profiles**:
   - Selectable writing presets (`Technical Tutorial`, `Beginner Friendly`, `Research Style`, `Developer Blog`, `LinkedIn Post`, `SEO Blog`) driving Planner & Worker prompts.

6. **🎯 SEO Optimization Agent**:
   - Generates meta titles, meta descriptions, primary/secondary keywords, clean URL slugs, readability levels, and FAQ sections.

7. **⚡ Smart Parallel Worker Fanout**:
   - Concurrently generates article sections using LangGraph's dynamic `Send` pattern, with shared rate limiting (`groq_rate_limiter`) to avoid API rate spikes.

8. **📊 Offline Evaluation Benchmark Suite**:
   - Standalone evaluation framework (`tests/eval_benchmark.py`) assessing Relevance, Completeness, Citation Coverage, Word Target Adherence, and Execution Latency across test datasets.

---

## 📂 Project Structure

```
ai-blog-writing-agent/
├── app.py                     # Main Streamlit application entry point
├── requirements.txt           # Python dependencies
├── pytest.ini                 # Pytest configuration
├── .env.example               # Environment variable placeholders
├── README.md                  # Project documentation
│
├── src/
│   ├── agent/
│   │   ├── graph.py           # LangGraph StateGraph assembly & compilation
│   │   ├── nodes.py           # Router, Research, Filter, Contradiction, Orchestrator, HITL, Worker, Reducer, FactChecker, Critic, SEO, Visual nodes
│   │   ├── rate_limiter.py    # Request rate limiter for LLM calls
│   │   ├── schemas.py         # Pydantic models & LangGraph TypedDict State
│   │   └── tools.py           # Tavily web search & image generation tools
│   │
│   ├── paths.py               # Output & image filesystem path helpers
│   │
│   └── ui/
│       ├── components.py      # Streamlit header, content brief, progress bar, dual HITL review cards
│       ├── helpers.py         # Streaming utilities & past blog loaders
│       ├── renderer.py        # Markdown & image rendering engine
│       ├── theme.py           # Custom CSS styling tokens
│       └── views.py           # Article workspace, SEO tab, FactCheck tab, Critic tab, and archive views
│
└── tests/
    ├── test_agent.py          # Complete unit test suite (28 test cases)
    └── eval_benchmark.py      # Offline evaluation benchmark suite
```

---

## 🛠 Technology Stack

- **Orchestration**: LangGraph (`StateGraph`, `MemorySaver`, `interrupt`, `Command`)
- **LLM Core**: Groq API (`llama-3.3-70b-versatile` / `qwen-2.5-32b`) / ChatOpenAI
- **Web Research**: Tavily Search API
- **Data Schemas**: Pydantic v2, TypedDict
- **User Interface**: Streamlit
- **Diagrams & Visuals**: Mermaid.js, Gemini API / Pollinations AI
- **Observability**: LangChain Tracing V2 (LangSmith)

---

## ⚡ Quickstart Guide

### 1. Prerequisites
- Python 3.10 or higher installed
- Free API keys from [Groq Console](https://console.groq.com/keys) and [Tavily](https://tavily.com)

### 2. Installation

```bash
# Clone repository
git clone https://github.com/jeetsinghbhati7773/blog-writing-multi-agent.git
cd blog-writing-multi-agent

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Setup

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

`.env` configuration:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
TAVILY_API_KEY=your_tavily_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_optional
```

### 4. Running the Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🧪 Testing & Evaluation

### 1. Automated Unit Test Suite

Run the unit test suite covering router, research filter, contradiction detector, outline validator, critic loop, fact checker, SEO agent, and HITL interrupts:

```bash
pytest tests/test_agent.py -v
```

### 2. Offline Evaluation Benchmark Suite

Run the evaluation benchmark script:

```bash
python tests/eval_benchmark.py
```

Outputs composite quality, relevance, completeness, citation coverage, and latency metrics across test prompts.
