# chatbot.py
# Handles the LLM integration (Ollama with qwen2.5:3b) for Standard RAG.

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Singleton LLM instance — avoids re-creating the connection on every request
_llm_instance = None

def get_llm():
    """
    Returns a cached connection to the local Ollama instance.
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOllama(model="qwen2.5:3b", temperature=0.0)
    return _llm_instance

def generate_rag_response(prompt, chat_history, retrieved_chunks):
    """
    Runs the Standard RAG pipeline with streaming.
    Uses a simplified prompt structure optimized for small models.
    """
    # Guardrail: If no relevant policy chunks were retrieved, refuse out-of-domain queries immediately
    if not retrieved_chunks:
        yield "I am a University Policy Assistant. I can only answer questions using the provided university policy documents."
        return

    llm = get_llm()
    
    # Format context from retrieved chunks
    context_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        source = chunk.metadata.get('source_name', 'Unknown Document')
        page = chunk.metadata.get('page', 'Unknown Page')
        if isinstance(page, int):
            page += 1
        context_text += f"\n[Source {i+1}: {source}, Page {page}]\n{chunk.page_content}\n"
    
    # Strict system prompt instructing the model to refuse out-of-context queries
    system_prompt = (
        f"You are a University Policy Assistant. "
        f"Answer the user's question using ONLY the facts explicitly stated in the context below. "
        f"If the context below does not explicitly contain the answer to the question, you MUST reply: 'I am a University Policy Assistant and I can only answer questions using the provided university policy documents.' "
        f"Do NOT use outside general knowledge or answer unrelated topics under any circumstances.\n\n"
        f"Context:\n{context_text}"
    )
    
    # Format chat history
    formatted_chat_history = [SystemMessage(content=system_prompt)]
    for msg in chat_history:
        if msg["role"] == "user":
            formatted_chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_chat_history.append(AIMessage(content=msg["content"]))
            
    formatted_chat_history.append(HumanMessage(content=prompt))
    
    try:
        response_stream = llm.stream(formatted_chat_history)
        accumulated_text = ""
        for chunk in response_stream:
            accumulated_text += chunk.content
            yield accumulated_text
    except Exception as e:
        yield f"Encountered an error during generation. Details: {str(e)}"
