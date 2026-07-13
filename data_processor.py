# data_processor.py
# Handles document loading, text chunking, and metadata extraction.

import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents_from_directory(directory_path="data"):
    """
    Loads all PDF documents from the preloaded specified directory.
    Returns a list of LangChain Document objects.
    """
    documents = []
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        return documents
        
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            # Ensure metadata has document name and page number
            for doc in docs:
                doc.metadata['source_name'] = filename
            documents.extend(docs)
    return documents

def chunk_documents(documents, chunk_size=800, chunk_overlap=100):
    """
    Splits documents into smaller chunks (500-800 tokens) while preserving metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks
