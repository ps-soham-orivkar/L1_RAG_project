# src/data_processor.py
# Document loading, text chunking, metadata extraction, and structured logging.

import os
import time
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.logger import get_logger

logger = get_logger("Ingestion")

def load_documents_from_directory(directory_path="data"):
    """
    Loads all PDF documents from the specified directory.
    Logs progress directly to terminal and log file.
    """
    documents = []
    if not os.path.exists(directory_path):
        logger.warning(f"[STEP 1: INGESTION] Directory '{directory_path}' not found. Creating empty folder.")
        os.makedirs(directory_path)
        return documents
        
    pdf_files = [f for f in os.listdir(directory_path) if f.endswith(".pdf")]
    logger.info(f"[STEP 1: INGESTION] Found {len(pdf_files)} PDF files in '{directory_path}'. Loading pages...")
    
    start_time = time.time()
    for filename in pdf_files:
        file_path = os.path.join(directory_path, filename)
        try:
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata['source_name'] = filename
            documents.extend(docs)
            logger.info(f"[STEP 1: INGESTION]   +-- Loaded '{filename}' ({len(docs)} pages).")
        except Exception as e:
            logger.error(f"[STEP 1: INGESTION] Failed to load '{filename}': {e}")
            
    elapsed = time.time() - start_time
    logger.info(f"[STEP 1: INGESTION] Document loading complete in {elapsed:.3f}s. Total pages loaded: {len(documents)}.")
    return documents

def chunk_documents(documents, chunk_size=800, chunk_overlap=100):
    """
    Splits documents into smaller chunks while preserving metadata.
    Logs chunking stats directly to terminal and log file.
    """
    if not documents:
        logger.warning("[STEP 2: CHUNKING] No documents provided for chunking.")
        return []

    start_time = time.time()
    logger.info(f"[STEP 2: CHUNKING] Splitting {len(documents)} pages into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    elapsed = time.time() - start_time
    
    if chunks:
        avg_len = sum(len(c.page_content) for c in chunks) / len(chunks)
        logger.info(f"[STEP 2: CHUNKING] Created {len(chunks)} text chunks (avg length: {avg_len:.1f} chars) in {elapsed:.3f}s.")
    else:
        logger.warning("[STEP 2: CHUNKING] Text splitter yielded 0 chunks.")
        
    return chunks
