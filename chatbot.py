# chatbot.py
# Handles the LLM integration (Ollama with llama3.2) for Standard RAG.

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def get_llm():
    """
    Initializes the connection to the local Ollama instance running the llama3.2 model.
    """
    return ChatOllama(model="llama3.2", temperature=0.0)

def generate_rag_response(prompt, chat_history, retrieved_chunks):
    """
    Runs the Standard RAG pipeline (Fast).
    """
    llm = get_llm()
    
    # Format context
    context_text = ""
    if retrieved_chunks:
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk.metadata.get('source_name', 'Unknown Document')
            page = chunk.metadata.get('page', 'Unknown Page')
            if isinstance(page, int):
                page += 1
            context_text += f"\n--- Source {i+1}: {source}, Page {page} ---\n{chunk.page_content}\n"
    else:
        context_text = "No relevant policy documents were found."
    
    system_prompt = (
        "You are a helpful and intelligent University Policy Assistant.\n\n"
        "INSTRUCTIONS:\n"
        "1. Use the provided context to answer the user's question.\n"
        "2. If the context contains the answer, you MUST include citations (e.g. Source 1: DocName, Page X) at the end of your response.\n"
        "3. If the context does not contain the answer, reply ONLY with: 'I do not have enough information in the uploaded policies to answer this.' Do NOT try to make up or hypothesize an answer.\n\n"
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
