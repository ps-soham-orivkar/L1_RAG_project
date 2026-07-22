# src/chatbot.py
# Strict University Policy Domain Assistant: Answers university policy and academic queries ONLY.
# Refuses out-of-domain non-policy questions deterministically using cross-encoder relevance thresholding.

import time
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.logger import get_logger

logger = get_logger("Chatbot")

_llm_instance = None

def get_llm():
    """
    Returns a cached connection to local Ollama with optimized low-latency settings.
    """
    global _llm_instance
    if _llm_instance is None:
        logger.info("[LLM INIT] Connecting to local Ollama (model='qwen2.5:3b', temperature=0.0)...")
        _llm_instance = ChatOllama(
            model="qwen2.5:3b",
            temperature=0.0,
            num_ctx=4096
        )
    return _llm_instance

def is_vague_query(prompt):
    """
    Detects short or ambiguous queries that require clarification.
    """
    words = prompt.strip().split()
    if len(words) <= 1 and prompt.strip().lower() in ["help", "info", "what"]:
        return True
    return False

def generate_rag_response(prompt, chat_history, retrieved_chunks, rerank_scores=None):
    """
    Strict Domain RAG Response Generator:
    1. Answers University Policy & Academic queries using retrieved context with inline citations [Source 1], [Source 2].
    2. Deterministically checks cross-encoder rerank scores and refuses out-of-domain non-policy questions (e.g. celebrities, politics, sports).
    """
    start_time = time.time()
    
    # 1. Check for vague query
    if is_vague_query(prompt):
        logger.info(f"[STEP 5: QUERY GUARDRAIL] Query '{prompt}' flagged as ambiguous/too short.")
        yield "I am a University Policy Assistant. Please specify which university policy or academic regulation you would like assistance with."
        return

    # 2. Check for zero retrieved context or extremely low reranker score (out-of-domain query)
    max_score = max(rerank_scores) if rerank_scores else 1.0
    if not retrieved_chunks or max_score < -2.0:
        logger.warning(f"[STEP 5: QUERY GUARDRAIL] Out-of-domain query '{prompt}' (max rerank score: {max_score:.3f}). Triggering domain refusal.")
        yield "I am a specialized University Policy Assistant. I can only assist with questions regarding university policies, academic regulations, examination rules, grading standards, and student conduct. Please ask a policy-related question."
        return

    llm = get_llm()
    
    # 3. Format Context and build Sources Summary
    context_text = ""
    sources_summary = []
    
    for i, chunk in enumerate(retrieved_chunks):
        source_name = chunk.metadata.get('source_name', 'Policy Document')
        page = chunk.metadata.get('page', 0)
        page_str = str(page + 1) if isinstance(page, int) else str(page)
        
        score_info = f", Score: {rerank_scores[i]:.3f}" if rerank_scores and i < len(rerank_scores) else ""
        
        context_text += f"\n[Source {i+1}: {source_name}, Page {page_str}]\n{chunk.page_content}\n"
        sources_summary.append(f"[Source {i+1}] {source_name} (Page {page_str}){score_info}")

    system_prompt = (
        "You are an expert, strict University Policy Assistant.\n"
        "Your ONLY role is to assist users with University Policies, Academic Regulations, Examination Guidelines, Attendance Rules, Grading, and Student Conduct.\n\n"
        "STRICT DOMAIN GUARDRAILS:\n"
        "1. Answer the user's query comprehensively based ONLY on the provided Context below. Include inline bracket citations like [Source 1], [Source 2].\n"
        "2. Do NOT answer out-of-domain non-policy queries.\n"
        "3. Do NOT list a separate 'Sources' section at the end of your response text.\n\n"
        f"Context:\n{context_text}"
    )

    formatted_messages = [SystemMessage(content=system_prompt)]
    for msg in chat_history:
        if msg["role"] == "user":
            formatted_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_messages.append(AIMessage(content=msg["content"]))
            
    formatted_messages.append(HumanMessage(content=prompt))

    logger.info(f"[STEP 5: LLM GENERATION] Streaming response from Qwen 2.5:3B with strict domain guardrails...")
    
    try:
        response_stream = llm.stream(formatted_messages)
        accumulated_text = ""
        
        for chunk in response_stream:
            accumulated_text += chunk.content
            yield accumulated_text
            
        elapsed = time.time() - start_time
        logger.info(f"[STEP 5: LLM GENERATION] Answer generated in {elapsed:.3f}s ({len(accumulated_text)} characters).")
        
        yield accumulated_text

    except Exception as e:
        logger.error(f"[STEP 5: LLM GENERATION] LLM Stream error: {e}")
        yield f"Encountered an error during response generation. Details: {str(e)}"
