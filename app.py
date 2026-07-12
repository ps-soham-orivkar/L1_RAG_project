# app.py
import gradio as gr
from chatbot import generate_rag_response
from tools import route_query_to_tool
from data_processor import load_uploaded_document, chunk_documents
from retriever import HybridRetriever
import os
import shutil

# 1. Clean ChromaDB cache on startup to prevent duplicate chunks in local database
CHROMA_DIR = "./chroma_db"
if os.path.exists(CHROMA_DIR):
    try:
        shutil.rmtree(CHROMA_DIR)
        print("Cleared persistent vector database to ensure a clean start.")
    except Exception as e:
        print(f"Warning: Could not clear chroma_db cache: {e}")

# Initialize Global Retriever
retriever = HybridRetriever(persist_directory=CHROMA_DIR)

def init_knowledge_base():
    """Load preloaded documents from the data directory on startup."""
    from data_processor import load_documents_from_directory
    docs = load_documents_from_directory("data")
    if docs:
        chunks = chunk_documents(docs)
        retriever.ingest_documents(chunks)
        print(f"Loaded {len(chunks)} chunks into the vector database.")

# Initialize the vector DB
init_knowledge_base()

def chat_with_agent(message, history):
    """
    Standard RAG Pipeline with UI Streaming.
    History is a list of [user_message, bot_message] lists or Gradio messages dict.
    """
    # 1. Fast tool check
    tool_response = route_query_to_tool(message)
    if tool_response:
        yield tool_response
        return
        
    # 2. Format history
    formatted_history = []
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            formatted_history.append({"role": msg["role"], "content": msg["content"]})
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            formatted_history.append({"role": "user", "content": msg[0]})
            formatted_history.append({"role": "assistant", "content": msg[1]})
            
    try:
        # 3. Fast Retrieval
        retrieved_chunks = retriever.retrieve(message)
        
        # 4. Stream Generation
        for response_so_far in generate_rag_response(message, formatted_history, retrieved_chunks):
            yield response_so_far
    except Exception as e:
        yield f"Error connecting to the RAG backend.\n\nDetails: {str(e)}"

def process_uploaded_pdf(file_path):
    """Processes an uploaded PDF and adds it to the knowledge base without duplicates."""
    if file_path is None:
        return "Please upload a file first."
        
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        chunks = chunk_documents(docs)
        
        # Ingest only the new chunks (append support is handled inside ingest_documents)
        retriever.ingest_documents(chunks)
        
        filename = os.path.basename(file_path)
        return f"Successfully processed '{filename}' and added {len(chunks)} chunks to the knowledge base."
    except Exception as e:
        return f"Failed to process document: {str(e)}"

# Build the Gradio UI Layout
with gr.Blocks(title="🎓 University Policy Assistant") as demo:
    gr.Markdown("# 🎓 University Policy Assistant")
    gr.Markdown("Ask me anything about university policies or use my tools (e.g., 'calculate attendance 40 30' or 'check eligibility 80').")
    
    with gr.Row():
        with gr.Column(scale=3):
            # The chat interface
            gr.ChatInterface(
                fn=chat_with_agent,
            )
            
        with gr.Column(scale=1):
            # The document upload interface
            gr.Markdown("### 📄 Document Upload")
            gr.Markdown("Upload a PDF policy document to process.")
            upload_button = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_output = gr.Textbox(label="Upload Status", interactive=False)
            
            upload_button.upload(
                fn=process_uploaded_pdf,
                inputs=upload_button,
                outputs=upload_output
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
