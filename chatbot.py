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
    llm = get_llm()
    
    # Format context from retrieved chunks
    context_text = ""
    if retrieved_chunks:
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk.metadata.get('source_name', 'Unknown Document')
            page = chunk.metadata.get('page', 'Unknown Page')
            if isinstance(page, int):
                page += 1
            context_text += f"\n[Source {i+1}: {source}, Page {page}]\n{chunk.page_content}\n"
    else:
        context_text = "No relevant policy documents were found."
    
    # Simplified prompt — small models follow this much better than numbered instructions
    system_prompt = (
        f"You are a University Policy Assistant. "
        f"Answer the user's question using ONLY the context below. "
        f"Include the source citation at the end of your answer. "
        f"If the context does not contain the answer, say so.\n\n"
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
