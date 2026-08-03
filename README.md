# ✍️ Blog Writing Agent

An autonomous **multi-agent AI system** that generates high-quality technical blogs using **LangGraph**, **LangChain**, **Tavily Search**, **Google Gemini Imagen**, and **Streamlit**. The application researches technical topics, plans structured outlines, writes blog sections in parallel, generates AI-powered diagrams, and produces publication-ready Markdown articles.

---

# 🚀 Features

* 🤖 **Multi-Agent Workflow** powered by LangGraph
* 🧭 **Intelligent Research Routing**

  * Closed Book
  * Hybrid
  * Open Book
* 🔎 **Live Web Research** using Tavily Search API
* 📝 **Automated Blog Planning**

  * Title generation
  * Audience selection
  * Writing tone
  * Section planning
  * Word count allocation
* ⚡ **Parallel Content Generation** for faster blog creation
* 🖼️ **AI Diagram & Image Generation** using Google Gemini Imagen
* 📄 **Markdown Export** with embedded images
* 📊 **Interactive Streamlit Dashboard**
* 📦 ZIP download containing blog and generated assets
* 📚 Blog history and previous output loading

---

# 🏗️ Architecture

```text
                 User Topic
                     │
                     ▼
             ┌────────────────┐
             │  Router Agent  │
             └────────────────┘
               │         │
      No Research    Research Required
               │         │
               ▼         ▼
          Orchestrator  Research Agent
               │         │
               └────┬────┘
                    ▼
          Blog Outline Planning
                    │
                    ▼
        Parallel Worker Agents
     (One Worker per Blog Section)
                    │
                    ▼
             Content Reducer
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
  Image Decision          Merge Sections
        │                        │
        ▼                        ▼
 Gemini Imagen            Markdown Blog
        │                        │
        └───────────┬────────────┘
                    ▼
            Streamlit Dashboard
```

---

# 📂 Project Structure

```text
blog-writing-agent/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
│
├── src/
│   ├── agent/
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   ├── schemas.py
│   │   └── tools.py
│   │
│   └── ui/
│       ├── helpers.py
│       ├── renderer.py
│       └── views.py
│
├── outputs/
├── images/
└── notebooks/
```

---

# ⚙️ Tech Stack

| Category         | Technology           |
| ---------------- | -------------------- |
| Agent Framework  | LangGraph            |
| LLM Framework    | LangChain            |
| LLM              | OpenAI GPT           |
| Web Research     | Tavily Search API    |
| Image Generation | Google Gemini Imagen |
| Frontend         | Streamlit            |
| Validation       | Pydantic             |
| Language         | Python               |

---

# 🔄 Workflow

## 1. Router Agent

Analyzes the user topic and determines whether live web research is required.

Research Modes:

* Closed Book
* Hybrid
* Open Book

---

## 2. Research Agent

When research is needed, the agent:

* Generates search queries
* Retrieves recent web evidence
* Collects citations and snippets
* Passes evidence to downstream agents

---

## 3. Orchestrator Agent

Creates the complete writing plan, including:

* Blog title
* Target audience
* Writing tone
* Section outline
* Word count goals
* Visual requirements

---

## 4. Parallel Worker Agents

Each section is generated independently using parallel LangGraph workers, improving efficiency while maintaining consistent structure.

---

## 5. Reducer

The reducer:

* Merges all generated sections
* Produces the final Markdown document
* Identifies concepts that require visual aids

---

## 6. Image Generation

Google Gemini Imagen generates:

* Technical diagrams
* Architecture illustrations
* Workflow graphics

Generated images are saved locally and automatically embedded into the Markdown output.

---

# 💻 Installation

## Clone Repository

```bash
git clone <repository-url>

cd blog-writing-agent
```

## Create Virtual Environment

### Windows

```powershell
python -m venv venv
```

### Linux/macOS

```bash
python3 -m venv venv
```

---

## Activate Environment

### Windows

```powershell
.\venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key

TAVILY_API_KEY=your_tavily_api_key

GOOGLE_API_KEY=your_google_api_key
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

Open your browser:

```
http://localhost:8501
```

---

# 📊 Streamlit Dashboard

The application provides multiple tabs:

* 🧩 **Plan** – Review blog outline and metadata
* 🔎 **Research Evidence** – Inspect collected web sources
* 📝 **Markdown Preview** – View and download the generated blog
* 🖼️ **Images & Diagrams** – Preview AI-generated visuals
* 🧾 **Logs** – Monitor LangGraph execution and workflow state

---

# 📦 Output

Generated content is organized as:

```
outputs/
    blog-name.md

images/
    architecture.png
    workflow.png
    diagram.png
```

---

# 🔑 Environment Variables

| Variable       | Required | Purpose             |
| -------------- | -------- | ------------------- |
| OPENAI_API_KEY | ✅        | LLM generation      |
| TAVILY_API_KEY | Optional | Live web research   |
| GOOGLE_API_KEY | Optional | AI image generation |

---

# 🌟 Highlights

* Modular multi-agent architecture
* Intelligent routing for research
* Parallel content generation
* Automated technical diagrams
* Research-backed Markdown blogs
* Interactive Streamlit interface
* Extensible LangGraph workflow
* Production-ready project structure

