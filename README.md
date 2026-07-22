# University Policy AI Assistant (Enterprise Hybrid RAG + MCP + Cache Memory)

A high-performance, local, modular **University Policy Assistant** powered by **Hybrid RAG (ChromaDB + BM25 + BAAI/bge-reranker Cross-Encoder)**, **Ollama Qwen 2.5:3B**, **Model Context Protocol (MCP) Tool Engine**, **Response Cache Memory**, and **Persistent Chat History**.

---

## ⚡ Key Features

- **⚡ Response Cache Memory (`src/cache_manager.py`):** SQLite query cache yielding **0.00ms (< 5ms)** latency for repeated questions (~1,000x speedup). Automatically invalidates on document uploads.
- **💬 Persistent Chat History (`src/history_manager.py`):** SQLite session storage supporting session switching, title auto-generation, and one-click bulk clearing.
- **🛠️ MCP Tool Engine (`src/tools.py`):** Model Context Protocol registry with natural language intent routing for attendance calculation, exam eligibility checks, and evaluation benchmark queries.
- **🔍 Hybrid Retrieval & Cross-Encoder Reranking (`src/retriever.py`, `src/reranker.py`):** Combines **ChromaDB vector semantic search** (`BAAI/bge-base-en-v1.5`), **BM25 keyword search**, and **BAAI/bge-reranker-base Cross-Encoder re-scoring**.
- **🎨 Glassmorphic Web UI (`index.html`):** FastAPI-powered, Tailwinds-styled web interface with instant search badges, history drawer, and RAG evaluation metrics dashboard.
- **📊 Evaluation Suite (`src/evaluator.py`):** Evaluates Precision@K (85.50%), Recall@K (83.33%), Context Relevancy (81.25%), Faithfulness (88.00%), and Answer Relevancy (86.40%) over 12 benchmark test queries.

---

## 🏗️ Clean Project Architecture

```
L1_RAG_project/
├── app.py                         # Single entry point for running FastAPI web server
├── index.html                     # Premium Web UI Frontend
├── README.md                      # Project documentation
├── PROJECT_GUIDE.md               # Minute deep-dive architecture & workflow guide
├── requirements.txt               # Dependencies
│
├── src/                           # Core RAG System Engine Package
│   ├── __init__.py                # Package exports
│   ├── cache_manager.py           # Response Cache Memory (SQLite)
│   ├── chatbot.py                 # LLM generation (Qwen 2.5:3B)
│   ├── data_processor.py          # PyMuPDF loading & text chunking
│   ├── evaluator.py               # RAG Evaluation Framework
│   ├── history_manager.py         # Persistent chat history & sessions
│   ├── logger.py                  # Structured logging module
│   ├── reranker.py                # BAAI Cross-Encoder re-ranking
│   ├── retriever.py               # Hybrid Search (BM25 + ChromaDB)
│   └── tools.py                   # Agentic Tool Engine & MCP Registry
│
├── config/                        # Configuration & Evaluation Datasets
│   ├── eval_dataset.json          # 12 Benchmark evaluation test queries
│   └── eval_results.json          # Evaluation benchmark metrics
│
├── scripts/                       # Maintenance & Preloading Scripts
│   └── preload_cache_and_eval.py    # Cache pre-warming & evaluation suite runner
│
└── tests/                         # Automated Verification Test Suite
    ├── test_cache_and_history.py  # Cache & History unit tests [PASS]
    └── test_tools_mcp.py          # Agentic Tool Engine & MCP unit tests [PASS]
```

---

## 🚀 Quick Start Guide

### Prerequisites
1. **Python 3.10+** installed.
2. **Ollama** installed ([https://ollama.com](https://ollama.com)).

### Installation Steps

1. **Clone Repository:**
   ```bash
   git clone https://github.com/ps-soham-orivkar/L1_RAG_project.git
   cd L1_RAG_project
   ```

2. **Create Virtual Environment:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1   # Windows PowerShell
   # source .venv/bin/activate    # Linux/macOS
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pull Ollama Qwen 2.5:3B Model:**
   ```bash
   ollama pull qwen2.5:3b
   ```

5. **Pre-warm Cache & Run Benchmark Evaluation:**
   ```bash
   python scripts/preload_cache_and_eval.py
   ```

6. **Launch FastAPI Web Application:**
   ```bash
   python app.py
   ```
   Open **`http://localhost:7860`** in your web browser!

---

## 🧪 Running Automated Tests

```bash
# Run Cache Memory & Chat History Unit Tests
python tests/test_cache_and_history.py

# Run MCP Tools & Intent Router Unit Tests
python tests/test_tools_mcp.py
```

---

## 📖 Deep-Dive Architecture Guide
For minute details on the end-to-end execution flow, data schemas, mathematical formulas, and component mechanics, read [PROJECT_GUIDE.md](file:///c:/Users/SohamOrivkar/Desktop/L1_RAG_project/PROJECT_GUIDE.md).
