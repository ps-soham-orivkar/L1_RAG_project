# University Policy Assistant using Hybrid RAG with Citations

## Overview
A local, modular University Policy Assistant that uses a Hybrid Retrieval-Augmented Generation (RAG) architecture. It combines semantic search (ChromaDB) and keyword search (BM25) to provide accurate answers to university policy-related queries. It uses the local Mistral model via Ollama for privacy and speed, and includes an MCP tool layer for basic tasks like attendance calculations. It also features a citation system to trace answers back to original documents.

## Features
* **PDF Document Processing:** Automatically loads PDFs from a preloaded directory and allows users to upload their own via the UI.
* **Text Chunking & Metadata:** Splits documents into 500-800 token chunks while preserving source and page metadata.
* **Hybrid RAG Retrieval:** Uses both vector similarity (ChromaDB + MiniLM) and keyword search (BM25) to find the most relevant context.
* **LLM Integration:** Connects to a local Ollama instance running Mistral to generate grounded responses.
* **Citation System:** Outputs the source document name and page number alongside generated answers.
* **MCP Tool Layer:** Includes basic python-based tools to "calculate attendance" and "check eligibility".
* **Streamlit UI:** Clean, interactive chat interface.

## Tech Stack
* **UI:** Streamlit
* **Backend:** Python
* **Framework:** LangChain
* **Embeddings:** MiniLM (sentence-transformers/all-MiniLM-L6-v2)
* **Vector DB:** ChromaDB (Local)
* **Keyword Search:** BM25 (rank_bm25)
* **LLM:** Ollama (Mistral)
* **Data Processing:** PyPDFLoader
* **MCP:** Python logic (custom)

## Installation

1. **Install Python Dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
2. **Install Ollama:**
   Download and install [Ollama](https://ollama.com/) for your operating system.
3. **Pull Mistral Model:**
   ```bash
   ollama pull mistral
   ```

## Usage

1. **Preload Documents (Optional):**
   Place your university policy PDF files in the `data/` folder before starting the application.
2. **Run the App:**
   ```bash
   streamlit run app.py
   ```
3. **Interact:**
   * Ask questions in the chat interface.
   * Upload additional PDF documents using the sidebar.
   * Use tools by typing commands like `calculate attendance 40 30` or `check eligibility 80`.

## Project Structure
```text
L1RAG/
├── app.py                 # Streamlit UI and main entry point
├── chatbot.py             # LLM prompt generation and citation extraction
├── data_processor.py      # PDF loading, chunking, and metadata handling
├── retriever.py           # Embeddings, ChromaDB, and Hybrid BM25 Search
├── tools.py               # MCP tool layer (attendance, eligibility)
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── data/                  # Directory for preloaded PDF policies
    └── .gitkeep
```

## Explanation of Hybrid RAG

**Retrieval-Augmented Generation (RAG)** is a technique where an AI model looks up specific documents before answering a question. 
In this project, we use **Hybrid RAG**, which combines two lookup methods to get the best of both worlds:
1. **Semantic Search (Vector Embeddings):** The system understands the "meaning" of your question and matches it to the underlying meaning of text chunks using mathematical vectors.
2. **Keyword Search (BM25):** The system looks for exact word matches (like specific acronyms or policy codes) in the text.

By combining these two methods, the assistant can find the right policy section even if you use different wording, while still accurately catching precise policy names or keywords. The retrieved sections are then passed to the local LLM (Mistral) to formulate a natural, easy-to-read answer complete with citations!
