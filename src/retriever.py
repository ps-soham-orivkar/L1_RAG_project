# src/retriever.py
# Production Hybrid Retriever (Chroma Vector DB + BM25 Keyword Search + BAAI/bge-reranker Cross-Encoder).

import time
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
from src.reranker import Reranker
from src.logger import get_logger

logger = get_logger("Retriever")

class HybridRetriever:
    def __init__(self, persist_directory="./chroma_db", embedding_model="BAAI/bge-base-en-v1.5"):
        self.persist_directory = persist_directory
        logger.info(f"[EMBEDDINGS] Loading high-accuracy embedding model: '{embedding_model}' on CPU...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
        self.bm25 = None
        self.documents = []
        
        # Load pre-existing persistent ChromaDB if present to avoid re-embedding
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                raw = self.vector_store.get()
                if raw and raw.get('documents'):
                    metas = raw.get('metadatas') or [{}] * len(raw['documents'])
                    self.documents = [
                        Document(page_content=doc, metadata=meta or {})
                        for doc, meta in zip(raw['documents'], metas)
                    ]
                    tokenized_corpus = [doc.page_content.lower().split() for doc in self.documents]
                    self.bm25 = BM25Okapi(tokenized_corpus)
                    logger.info(f"[VECTOR STORE] Loaded existing persistent vector database with {len(self.documents)} chunks from '{self.persist_directory}'.")
            except Exception as e:
                logger.warning(f"[VECTOR STORE] Could not load existing Chroma DB: {e}")
                self.vector_store = None
                self.documents = []
                self.bm25 = None

        # Initialize High-Accuracy Cross-Encoder Reranker
        self.reranker = Reranker(model_name="BAAI/bge-reranker-base")
        
    def reset_index(self):
        """
        Resets and clears ChromaDB vector store and BM25 index.
        """
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
            except Exception as e:
                logger.warning(f"Could not delete collection: {e}")
        self.vector_store = None
        self.documents = []
        self.bm25 = None
        if os.path.exists(self.persist_directory):
            try:
                import shutil
                shutil.rmtree(self.persist_directory)
                logger.info(f"Cleared persistent directory '{self.persist_directory}'.")
            except Exception as e:
                logger.warning(f"Could not remove persist_directory: {e}")

    def ingest_documents(self, chunks):
        """
        Ingest chunks into both ChromaDB and BM25 index with detailed terminal logging.
        """
        if not chunks:
            logger.warning("[VECTOR STORE] Ingest documents called with empty chunks list.")
            return
            
        start_time = time.time()
        logger.info(f"[VECTOR STORE] Ingesting {len(chunks)} chunks into ChromaDB & BM25 index...")

        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=chunks, 
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.documents = list(chunks)
        else:
            self.vector_store.add_documents(chunks)
            self.documents.extend(chunks)
        
        # Tokenize corpus for BM25
        tokenized_corpus = [doc.page_content.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        elapsed = time.time() - start_time
        logger.info(f"[VECTOR STORE] Corpus index ready in {elapsed:.3f}s. Total stored corpus: {len(self.documents)} chunks.")
        
    def retrieve(self, query, top_k_candidates=10, final_top_n=4, use_reranker=True):
        """
        Advanced RAG Retrieval:
        1. Hybrid Search (Chroma Semantic Search + BM25 Keyword Search)
        2. Interleaving & Deduplication
        3. Cross-Encoder Reranking to select top_n most relevant chunks.
        """
        if not self.vector_store or not self.bm25:
            logger.warning("[STEP 3: HYBRID SEARCH] Retrieval attempted before ingesting documents.")
            return [], []
            
        start_time = time.time()
        logger.info(f"[STEP 3: HYBRID SEARCH] Querying index for: '{query}'...")
        
        # 1. Semantic Search (Chroma)
        semantic_results = self.vector_store.similarity_search(query, k=top_k_candidates)
        logger.info(f"[STEP 3: HYBRID SEARCH]   |-- ChromaDB Semantic Search: retrieved {len(semantic_results)} candidates.")
        
        # 2. Keyword Search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(bm25_scores)[::-1][:top_k_candidates]
        keyword_results = [self.documents[i] for i in top_n_indices]
        logger.info(f"[STEP 3: HYBRID SEARCH]   |-- BM25 Keyword Search: retrieved {len(keyword_results)} candidates.")
        
        # 3. Interleave and Deduplicate
        candidate_chunks = []
        seen_content = set()
        
        for sem, keyw in zip(semantic_results, keyword_results):
            if sem.page_content not in seen_content:
                candidate_chunks.append(sem)
                seen_content.add(sem.page_content)
            if keyw.page_content not in seen_content:
                candidate_chunks.append(keyw)
                seen_content.add(keyw.page_content)
            if len(candidate_chunks) >= top_k_candidates:
                break
                
        candidate_chunks = candidate_chunks[:top_k_candidates]
        logger.info(f"[STEP 3: HYBRID SEARCH]   +-- Combined & Deduplicated Candidates: {len(candidate_chunks)} unique chunks.")

        if not use_reranker:
            elapsed = time.time() - start_time
            logger.info(f"[STEP 3: HYBRID SEARCH] Fast hybrid search finished in {elapsed:.3f}s.")
            return candidate_chunks[:final_top_n], []

        # 4. Cross-Encoder Reranking
        final_chunks, scores = self.reranker.rerank(query, candidate_chunks, top_n=final_top_n)
        elapsed = time.time() - start_time
        logger.info(f"[STEP 3 & 4: RETRIEVAL & RERANK] Finished full retrieval in {elapsed:.3f}s. {len(final_chunks)} chunks selected for LLM.")
        
        return final_chunks, scores
