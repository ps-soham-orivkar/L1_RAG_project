# University Policy Assistant using Hybrid RAG with Citations

## Overview
A local, modular University Policy Assistant that uses a Hybrid Retrieval-Augmented Generation (RAG) architecture. It combines semantic search (ChromaDB) and keyword search (BM25) to provide accurate answers to university policy-related queries. It uses the local Qwen2.5 (3B) model via Ollama for privacy and speed, and includes an MCP tool layer for basic tasks like attendance calculations. It also features a citation system to trace answers back to original documents.

## Features
* **PDF Document Processing:** Automatically loads PDFs from a preloaded directory and allows users to upload their own via the UI.
* **Text Chunking & Metadata:** Splits documents into 500-800 token chunks while preserving source and page metadata.
* **Hybrid RAG Retrieval:** Uses both vector similarity (ChromaDB + MiniLM) and keyword search (BM25) to find the most relevant context.
* **LLM Integration:** Connects to a local Ollama instance running Qwen2.5 (3B) to generate grounded responses.
* **Citation System:** Outputs the source document name and page number alongside generated answers.
* **MCP Tool Layer:** Includes basic python-based tools to "calculate attendance" and "check eligibility".
* **Gradio UI:** Clean, interactive chat interface.

## Tech Stack
* **UI:** Gradio
* **Backend:** Python
* **Framework:** LangChain
* **Embeddings:** MiniLM (sentence-transformers/all-MiniLM-L6-v2)
* **Vector DB:** ChromaDB (Local)
* **Keyword Search:** BM25 (rank_bm25)
* **LLM:** Ollama (qwen2.5:3b)
* **Data Processing:** PyMuPDFLoader (pymupdf)
* **MCP:** Python logic (custom)

## 🚀 Quick Start Guide (For Humans & AI Agents)

Follow these steps to set up and start the application on a new system.

### Prerequisites
1. **Python 3.10 or higher** installed.
2. **Ollama** installed ([https://ollama.com/](https://ollama.com/)).

---

### Step-by-Step Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ps-soham-orivkar/L1_RAG_project.git
   cd L1_RAG_project
   ```

2. **Set Up Virtual Environment (Recommended):**
   ```bash
   python -m venv .venv
   # On Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   > 💡 **Important Note for Windows Users & AI Agents:**
   > On Windows systems, to prevent C-extension DLL symbol errors with ChromaDB and `onnxruntime`, ensure you use a compatible NumPy version (< 2.0):
   > ```bash
   > pip install "numpy<2" onnxruntime==1.16.3
   > ```

4. **Pull & Start Ollama LLM Model:**
   Make sure the Ollama service is running, then pull the Qwen2.5 model:
   ```bash
   ollama pull qwen2.5:3b
   ```

5. **Run the Application:**
   ```bash
   python app.py
   ```
   Once launched, open your browser and navigate to **`http://localhost:7860`** to interact with the assistant UI!

---

## Usage

1. **Preload Documents (Optional):**
   Place your university policy PDF files inside the `data/` directory before launching `python app.py`. They will be automatically parsed, chunked, and indexed into the vector database on startup.
2. **Interactive UI:**
   - **Chat:** Ask policy-related questions directly in the Gradio chat interface.
   - **Document Upload:** Upload additional PDF documents at runtime using the sidebar file picker.
   - **Tools:** Ask natural queries like `calculate attendance 40 30` or `check eligibility 80` to trigger the Python tool layer.

## Project Structure
```text
L1RAG/
├── app.py                 # Gradio UI and main entry point
├── chatbot.py             # LLM prompt generation and citation extraction
├── data_processor.py      # PDF loading, chunking, and metadata handling
├── retriever.py           # Embeddings, ChromaDB, and Hybrid BM25 Search
├── tools.py               # MCP tool layer (attendance, eligibility)
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── data/                  # Directory for preloaded PDF policies
```

## Explanation of Hybrid RAG

**Retrieval-Augmented Generation (RAG)** is a technique where an AI model looks up specific documents before answering a question. 
In this project, we use **Hybrid RAG**, which combines two lookup methods to get the best of both worlds:
1. **Semantic Search (Vector Embeddings):** The system understands the "meaning" of your question and matches it to the underlying meaning of text chunks using mathematical vectors.
2. **Keyword Search (BM25):** The system looks for exact word matches (like specific acronyms or policy codes) in the text.

By combining these two methods, the assistant can find the right policy section even if you use different wording, while still accurately catching precise policy names or keywords. The retrieved sections are then passed to the local LLM (Qwen2.5) to formulate a natural, easy-to-read answer complete with citations!
