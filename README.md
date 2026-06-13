# RAG Knowledge Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A Retrieval-Augmented Generation (RAG) + multi-tool AI agent that answers questions from your documents **and** calls external APIs (web search, weather, calculator) — all through a clean chat interface.

---

## Demo

| Feature | Screenshot |
|---------|-----------|
| 📄 Upload & chat with documents | |
| 🌐 Web search, weather, math | |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI (app.py)              │
│   ┌──────────────┐          ┌────────────────────┐  │
│   │  Sidebar      │          │  Chat Panel        │  │
│   │  - Upload doc │          │  - Message history │  │
│   │  - List docs  │◄────────►│  - Agent responses │  │
│   │  - Clear KB   │          │  - Tool traces     │  │
│   └──────┬───────┘          └─────────┬──────────┘  │
│          │                            │              │
└──────────┼────────────────────────────┼──────────────┘
           │                            │
           ▼                            ▼
   ┌───────────────┐          ┌─────────────────┐
   │  ingest_doc() │          │  AgentExecutor   │
   │  (rag.py)     │          │  (agent_core.py) │
   └───────┬───────┘          └────────┬────────┘
           │                           │
           ▼                           ▼
   ┌───────────────┐          ┌─────────────────┐
   │  ChromaDB     │          │  Tools           │
   │  (vector      │◄────────►│  - knowledge_base│
   │   store)      │          │  - web_search    │
   │               │          │  - weather       │
   │  OpenAI       │          │  - calculator    │
   │  Embeddings   │          │                  │
   └───────────────┘          └─────────────────┘
                                        │
                                        ▼
                              ┌─────────────────┐
                              │  External APIs   │
                              │  - DuckDuckGo    │
                              │  - Open-Meteo    │
                              │  - OpenAI        │
                              └─────────────────┘
```

### Component Overview

| Module | File | Responsibility |
|--------|------|----------------|
| **RAG Engine** | `agent/rag.py` | Document parsing (PDF/DOCX/TXT/MD), chunking, embedding with OpenAI, ChromaDB storage/retrieval |
| **Agent Core** | `agent/agent_core.py` | LangChain AgentExecutor with OpenAI function-calling LLM, tool routing, conversation management |
| **Tools** | `agent/tools.py` | Tool definitions: knowledge base search, web search (DuckDuckGo), weather (Open-Meteo), calculator |
| **Frontend** | `app.py` | Streamlit UI: document upload sidebar, chat interface |
| **Seed Data** | `scripts/seed_data.py` | Optional script to populate the KB with sample documents |

---

## Design Decisions

### 1. Why LangChain for the Agent Framework?

**Decision**: LangChain over raw OpenAI function calling or custom ReAct loop.

- LangChain's `AgentExecutor` handles tool routing, retries, and parse-error recovery out of the box.
- `create_openai_functions_agent` gives us native OpenAI tool-calling with structured args (Pydantic schemas).
- The `MessagesPlaceholder` pattern cleanly separates chat history from the current turn.

### 2. Why ChromaDB as the Vector Store?

**Decision**: ChromaDB over Pinecone / Weaviate / Qdrant.

- **Zero-infra**: Chroma runs as an embedded library — no separate server required.
- **Persistence**: Saves to disk at `vector_store/chroma_db/`; survives container restarts.
- **LangChain-native**: First-class integration via `langchain_chroma`.

Trade-off: Chroma doesn't scale to millions of vectors as well as Pinecone. For a knowledge base of hundreds of documents it is ideal. Scale-up path: swap to Pinecone/Qdrant with a one-line config change.

### 3. Why OpenAI `text-embedding-3-small`?

**Decision**: `text-embedding-3-small` (1536-dim) over `text-embedding-3-large` (3072-dim).

- Cheaper ($0.02/1M tokens vs $0.13/1M).
- Sufficient quality for semantic search on document chunks.
- LangChain supports it natively.

### 4. Why DuckDuckGo (no API key) for Web Search?

**Decision**: DuckDuckGo's free search as default; optional Tavily for production.

- **Zero onboarding**: Users can run the app immediately without registering for a search API.
- DuckDuckGo may rate-limit in heavy use. **Production recommendation**: Switch to Tavily (`TAVILY_API_KEY`) — it is designed for LLM agents and costs ~$1 per 1000 queries.

### 5. Why Open-Meteo for Weather?

**Decision**: Open-Meteo over WeatherAPI / OpenWeatherMap.

- Open-Meteo is completely **free and open source** with no API key required.
- Uses official weather data from national meteorological services.
- Supports geocoding and 7-day forecasts.

### 6. Why Streamlit over Gradio?

**Decision**: Streamlit for the frontend.

- Streamlit's `chat_message` component gives a natural chat UI with minimal code.
- Better file upload handling with progress indicators.
- Simpler state management via `st.session_state`.
- More mature ecosystem for data/document apps.

### 7. Chunking Strategy

```
Chunk size: 1000 tokens | Overlap: 200 tokens | Separators: \n\n → \n → . → ...
```

- 1000 tokens gives enough context for most answers without exceeding LLM context windows.
- 200-token overlap preserves cross-chunk context (paragraphs that span chunk boundaries).
- RecursiveCharacterTextSplitter respects natural boundaries first.

---

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/rag-knowledge-agent.git
cd rag-knowledge-agent

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY

# 5. Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Seed Sample Data (Optional)

```bash
python scripts/seed_data.py
```

This ingests sample company policy and product docs so you can start querying immediately.

---

## Deployment

### Option 1: Streamlit Community Cloud (Free, recommended)

1. Push the repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **"Deploy an app"**.
3. Select the repo, branch, and set `app.py` as the entry point.
4. Add `OPENAI_API_KEY` as a **Secret** in Streamlit Cloud settings.

### Option 2: Docker (Any cloud — AWS, GCP, Railway, Render)

```bash
docker build -t rag-knowledge-agent .
docker run -p 8501:8501 --env-file .env rag-knowledge-agent
```

Or with docker-compose:

```bash
docker-compose up --build
```

### Option 3: Hugging Face Spaces (Free GPU optional)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces).
2. Choose **Streamlit** as the SDK.
3. Push the repo.
4. Set `OPENAI_API_KEY` as a Space Secret.

### Option 4: Railway / Render

1. Connect your GitHub repo to Railway or Render.
2. Set the build command: `pip install -r requirements.txt`
3. Set the start command: `streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`
4. Add `OPENAI_API_KEY` as an environment variable.

---

## Usage

### Upload a Document

1. Click **"Choose a file"** in the sidebar.
2. Supported formats: `PDF`, `DOCX`, `TXT`, `MD`.
3. The system automatically parses, chunks, embeds, and indexes the content.

### Ask Questions

Try these queries after uploading:

| Category | Example |
|----------|---------|
| 📄 RAG | *"What is the vacation policy?"* |
| ℹ️ Mixed | *"How does our pricing compare to competitors?"* |
| 🌐 Web | *"What are the latest AI breakthroughs in 2025?"* |
| 🌤 Weather | *"What is the weather in Tokyo?"* |
| 🔢 Calculator | *"What is 15% of 3400?"* |

The agent decides which tool(s) to use. It will search the knowledge base first, then fall back to web search or other tools as needed.

### Manage the Knowledge Base

- **View**: The sidebar lists all indexed documents.
- **Clear**: Click **"Clear All Documents"** to delete the vector index and start fresh.

---

## Project Structure

```
rag-knowledge-agent/
├── app.py                  # Streamlit frontend
├── agent/
│   ├── __init__.py
│   ├── agent_core.py       # AgentExecutor orchestration
│   ├── rag.py              # RAG engine (load, chunk, embed, search)
│   └── tools.py            # Tool definitions
├── scripts/
│   └── seed_data.py        # Sample document ingestion
├── docs/
│   └── uploads/            # Uploaded document storage
├── vector_store/
│   └── chroma_db/          # Persistent vector index (gitignored)
├── .streamlit/
│   └── config.toml         # Streamlit theme & server settings
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **LLM** | OpenAI GPT-4o-mini / GPT-4o | Function-calling support, cheap, fast |
| **Embeddings** | OpenAI text-embedding-3-small | 1536-dim, $0.02/1M tokens |
| **Vector Store** | ChromaDB | Embedded, persistent, no infra |
| **Agent Framework** | LangChain | Mature tool-calling, conversation memory |
| **Document Parsing** | PyMuPDF, python-docx | Reliable, permissive licenses |
| **Web Search** | DuckDuckGo (default) / Tavily (optional) | Free tier available |
| **Weather** | Open-Meteo | Free, no API key, accurate |
| **Frontend** | Streamlit | Fast iteration, chat components |
| **Deployment** | Docker + Streamlit Cloud / Railway / HF Spaces | Multi-platform |

---

## Roadmap

- [ ] Multi-user sessions with per-user vector stores
- [ ] Support for images in documents (multi-modal RAG)
- [ ] Streaming responses (Stream lit markdown streaming)
- [ ] Conversation memory (summarize old turns)
- [ ] Tool-usage visualisation (which tool was called and when)
- [ ] Batch document upload (zip files)

---

## License

MIT
=======
# -RAG-ReAct-Agent-
基于 RAG（检索增强生成） + ReAct Agent 的知识库问答系统
>>>>>>> c87f959e7148259c9131658b0a4ab192400476a7
