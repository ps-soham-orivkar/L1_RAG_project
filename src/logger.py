# src/logger.py
# Centralized structured logging for the RAG pipeline.

import logging
import sys

LOG_FILE = "rag_pipeline.log"

def get_logger(name="RAGPipeline"):
    """
    Returns a configured logger instance that writes to both stdout (terminal) and rag_pipeline.log.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Stream (console/terminal) handler
        stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1) if hasattr(sys.stdout, 'fileno') else sys.stdout
        stream_handler = logging.StreamHandler(stream)
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)
        
    return logger

# Global default logger instance
pipeline_logger = get_logger("RAGCore")
