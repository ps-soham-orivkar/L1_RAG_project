# app.py
# Production FastAPI Web Application serving the Policy AI Premium Web Interface.
# Integrated with Cache Memory and Persistent Chat History.

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import time
import json
import shutil
import types
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from src.chatbot import generate_rag_response
from src.tools import route_query_to_tool, mcp_registry
from src.data_processor import load_documents_from_directory, chunk_documents
from src.retriever import HybridRetriever
from src.evaluator import run_full_evaluation
from src.logger import get_logger
from src.cache_manager import ResponseCache
from src.history_manager import ChatHistoryManager
from langchain_community.document_loaders import PyMuPDFLoader

logger = get_logger("App")

CHROMA_DIR = "./chroma_db"

# Initialize Global Components
retriever = HybridRetriever(persist_directory=CHROMA_DIR, embedding_model="BAAI/bge-base-en-v1.5")
cache_manager = ResponseCache()
history_manager = ChatHistoryManager()

def init_knowledge_base():
    """Load preloaded documents from the data directory on startup ONLY if database is empty."""
    if retriever.vector_store is not None and len(retriever.documents) > 0:
        logger.info(f"Loaded {len(retriever.documents)} chunks from persistent vector database. Skipping re-chunking and re-embedding.")
        return

    docs = load_documents_from_directory("data")
    if docs:
        chunks = chunk_documents(docs)
        retriever.ingest_documents(chunks)
        logger.info(f"Ingested {len(chunks)} chunks into new vector database on startup.")

init_knowledge_base()

# Create FastAPI app
app = FastAPI(title="Policy AI Premium Assistant")

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []
    session_id: Optional[str] = None

class ToolExecuteRequest(BaseModel):
    name: str
    arguments: Optional[Dict[str, Any]] = {}

@app.get("/api/tools")
def api_list_tools():
    """Returns list of registered MCP tools and JSON schemas."""
    return mcp_registry.list_tools()

@app.post("/api/tools/execute")
def api_execute_tool(req: ToolExecuteRequest):
    """Executes an MCP tool by name with arguments."""
    result = mcp_registry.execute_tool(req.name, req.arguments or {})
    return {"tool": req.name, "result": result}

@app.get("/", response_class=HTMLResponse)
def get_index():
    """Serves the premium UI index.html file."""
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    raise HTTPException(status_code=404, detail="index.html not found")

@app.post("/api/chat/stream")
def api_chat_stream(req: ChatRequest):
    """
    RAG Streaming Chat API endpoint (SSE format).
    Checks Cache Memory first for 0ms instant response.
    Streams LLM output token-by-token on cache miss.
    Saves conversation history to persistent storage.
    """
    session_id = req.session_id or "default_session"
    message = req.message.strip()
    history = req.history or []

    # 1. Save user query to history
    history_manager.add_message(session_id, "user", message)

    def event_generator():
        # 2. Check Intent Router / Direct Tools FIRST (0s latency)
        tool_response = route_query_to_tool(message, retriever)
        if tool_response:
            full_tool_ans = ""
            if isinstance(tool_response, types.GeneratorType):
                for chunk in tool_response:
                    full_tool_ans += chunk
                    yield f"data: {json.dumps({'type': 'content', 'text': chunk})}\n\n"
            else:
                full_tool_ans = str(tool_response)
                yield f"data: {json.dumps({'type': 'content', 'text': full_tool_ans})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            history_manager.add_message(session_id, "assistant", full_tool_ans, sources=[])
            return

        # 3. Check Cache Memory for RAG Queries
        cached_result = cache_manager.get(message)
        if cached_result:
            sources = cached_result["sources"]
            cached_text = cached_result["response"]
            
            # Send sources with cached flag
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'cached': True, 'lookup_time_ms': cached_result['lookup_time_ms']})}\n\n"
            yield f"data: {json.dumps({'type': 'content', 'text': cached_text})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            # Save assistant response to session history
            history_manager.add_message(session_id, "assistant", cached_text, sources=sources, cached=True)
            return

        try:
            # 4. Hybrid Retrieval + Reranking
            retrieved_chunks, rerank_scores = retriever.retrieve(
                query=message,
                top_k_candidates=10,
                final_top_n=4,
                use_reranker=True
            )

            sources = []
            for c in retrieved_chunks:
                src = c.metadata.get('source_name', 'Policy Doc')
                pg = c.metadata.get('page', 0)
                pg_str = str(pg + 1) if isinstance(pg, int) else str(pg)
                sources.append(f"{src} (Page {pg_str})")
            sources = list(dict.fromkeys(sources))

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'cached': False})}\n\n"

            # 5. Stream response live token-by-token
            full_response = ""
            generator = generate_rag_response(message, history, retrieved_chunks, rerank_scores)
            for partial in generator:
                full_response = partial
                yield f"data: {json.dumps({'type': 'content', 'text': partial})}\n\n"

            # If response is a refusal, guardrail, or error message, clear sources
            if "I am a specialized University Policy Assistant" in full_response or "specify" in full_response.lower():
                sources = []

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'cached': False})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            # 6. Save answer into Cache Memory & Persistent History
            if full_response and not full_response.startswith("Encountered an error") and not "I am a specialized University Policy Assistant" in full_response:
                cache_manager.set(message, full_response, sources)
            history_manager.add_message(session_id, "assistant", full_response, sources=sources, cached=False)

        except Exception as e:
            logger.error(f"Streaming Error: {e}")
            err_msg = f"Error: {str(e)}"
            yield f"data: {json.dumps({'type': 'content', 'text': err_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/chat")
def api_chat(req: ChatRequest):
    """
    Standard HTTP endpoint for non-streaming chat requests.
    """
    session_id = req.session_id or "default_session"
    message = req.message.strip()
    history = req.history or []

    history_manager.add_message(session_id, "user", message)

    # Check cache
    tool_response = route_query_to_tool(message, retriever)
    if tool_response:
        full_ans = ""
        if isinstance(tool_response, types.GeneratorType):
            for chunk in tool_response:
                full_ans = chunk
        else:
            full_ans = str(tool_response)

        history_manager.add_message(session_id, "assistant", full_ans, sources=[])
        return {"response": full_ans, "sources": [], "cached": False}

    cached_result = cache_manager.get(message)
    if cached_result:
        history_manager.add_message(session_id, "assistant", cached_result["response"], sources=cached_result["sources"], cached=True)
        return {
            "response": cached_result["response"],
            "sources": cached_result["sources"],
            "cached": True,
            "lookup_time_ms": cached_result["lookup_time_ms"]
        }

    try:
        retrieved_chunks, rerank_scores = retriever.retrieve(
            query=message,
            top_k_candidates=10,
            final_top_n=4,
            use_reranker=True
        )

        sources = []
        for c in retrieved_chunks:
            src = c.metadata.get('source_name', 'Policy Doc')
            pg = c.metadata.get('page', 0)
            pg_str = str(pg + 1) if isinstance(pg, int) else str(pg)
            sources.append(f"{src} (Page {pg_str})")
        sources = list(dict.fromkeys(sources))

        full_response = ""
        generator = generate_rag_response(message, history, retrieved_chunks, rerank_scores)
        for partial in generator:
            full_response = partial

        if "I am a specialized University Policy Assistant" in full_response or "specify" in full_response.lower():
            sources = []

        if full_response and not full_response.startswith("Encountered an error") and not "I am a specialized University Policy Assistant" in full_response:
            cache_manager.set(message, full_response, sources)
        history_manager.add_message(session_id, "assistant", full_response, sources=sources, cached=False)

        return {"response": full_response, "sources": sources, "cached": False}

    except Exception as e:
        logger.error(f"API Chat Error: {e}")
        return {"response": f"Encountered error in RAG backend: {str(e)}", "sources": [], "cached": False}

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """Handles PDF file upload, ingests documents, and invalidates cache."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_path = os.path.join(data_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source_name'] = file.filename
            
        chunks = chunk_documents(docs)
        retriever.ingest_documents(chunks)
        
        # Clear cache when documents change
        cleared_count = cache_manager.clear()
        
        logger.info(f"Uploaded & ingested '{file.filename}' ({len(chunks)} chunks). Invalidated {cleared_count} cache entries.")
        return {"message": f"Successfully ingested '{file.filename}' ({len(chunks)} chunks). Cache refreshed ({cleared_count} entries invalidated)."}
    except Exception as e:
        logger.error(f"Failed to ingest uploaded file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refresh-kb")
def api_refresh_kb():
    """Clears persistent vector DB, removes PDF policy documents, and invalidates cache."""
    try:
        retriever.reset_index()
        cleared_count = cache_manager.clear()
        
        data_dir = "data"
        removed_count = 0
        if os.path.exists(data_dir):
            for fname in os.listdir(data_dir):
                if fname.endswith(".pdf"):
                    fpath = os.path.join(data_dir, fname)
                    try:
                        os.remove(fpath)
                        removed_count += 1
                    except Exception as e:
                        logger.warning(f"Could not remove {fpath}: {e}")

        logger.info(f"Refreshed knowledge base: cleared vector DB, removed {removed_count} documents, cleared {cleared_count} cached entries.")
        return {"message": f"Knowledge Base refreshed! Vector DB cleared, {removed_count} document(s) removed, and cache invalidated."}
    except Exception as e:
        logger.error(f"Failed to refresh knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# History & Cache API Endpoints

@app.get("/api/history/sessions")
def api_get_sessions():
    """Returns list of recent chat sessions."""
    return history_manager.list_recent_sessions(limit=20)

@app.delete("/api/history/sessions")
def api_clear_all_sessions():
    """Clears all persistent chat sessions and messages."""
    count = history_manager.clear_all_sessions()
    return {"message": f"Cleared all chat history ({count} sessions removed)."}

@app.get("/api/history/session/{session_id}")
def api_get_session_messages(session_id: str):
    """Returns stored messages for a chat session."""
    return history_manager.get_session_messages(session_id)

@app.delete("/api/history/session/{session_id}")
def api_delete_session(session_id: str):
    """Deletes a chat session."""
    success = history_manager.delete_session(session_id)
    return {"success": success, "session_id": session_id}

@app.get("/api/cache/stats")
def api_cache_stats():
    """Returns cache statistics (total entries, hit rate, total hits/misses)."""
    return cache_manager.get_stats()

@app.get("/api/docs")
def api_docs():
    """Returns list of active PDF policy documents."""
    data_dir = "data"
    docs = []
    if os.path.exists(data_dir):
        for fname in os.listdir(data_dir):
            if fname.endswith(".pdf"):
                fpath = os.path.join(data_dir, fname)
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                docs.append({
                    "name": fname,
                    "pages": len(PyMuPDFLoader(fpath).load()) if os.path.exists(fpath) else 1,
                    "size": f"{size_mb:.1f} MB"
                })
    return docs

@app.get("/api/metrics")
def api_metrics():
    """Loads eval_results.json from config directory and attaches live cache statistics."""
    res = {
        "timestamp": "Not run yet",
        "retrieval_metrics": {"precision_at_k": "N/A", "recall_at_k": "N/A", "context_relevancy": "N/A"},
        "generation_metrics": {"faithfulness": "N/A", "answer_relevancy": "N/A"}
    }
    paths_to_check = [os.path.join("config", "eval_results.json"), "eval_results.json"]
    for path in paths_to_check:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    res = json.load(f)
                    break
            except Exception as e:
                logger.warning(f"Error reading {path}: {e}")

    res["cache_metrics"] = cache_manager.get_stats()
    return res

@app.post("/api/evaluate")
def api_evaluate():
    """Triggers benchmark evaluation."""
    results = run_full_evaluation()
    return results

@app.get("/api/logs", response_class=PlainTextResponse)
def api_logs():
    """Returns last 100 lines of rag_pipeline.log."""
    log_file = "rag_pipeline.log"
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return "".join(lines[-100:])
        except Exception:
            pass
    return "No logs generated yet."

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
