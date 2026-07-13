# retriever.py
# Handles embeddings, Vector DB (Chroma), BM25, and Hybrid Search.

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        # Step 5: Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        
        # Step 6: BM25 setup
        self.bm25 = None
        self.documents = []
        
    def ingest_documents(self, chunks):
        """
        Ingest chunks into both ChromaDB and BM25 index.
        """
        if not chunks:
            return
            
        # Ingest into ChromaDB (Vector Store)
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
        
        # Ingest into BM25 (Keyword Search) - Rebuild full index
        tokenized_corpus = [doc.page_content.split(" ") for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def retrieve(self, query, top_k=10):
        """
        Step 7: Hybrid retrieval logic.
        Combines semantic search and keyword search.
        """
        if not self.vector_store or not self.bm25:
            return []
            
        # 1. Semantic Search (Chroma)
        semantic_results = self.vector_store.similarity_search(query, k=top_k)
        
        # 2. Keyword Search (BM25)
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(bm25_scores)[::-1][:top_k]
        keyword_results = [self.documents[i] for i in top_n_indices]
        
        # 3. Combine and deduplicate
        combined_results = []
        seen_content = set()
        
        # Simple interleaving for hybrid RAG
        for sem, keyw in zip(semantic_results, keyword_results):
            if sem.page_content not in seen_content:
                combined_results.append(sem)
                seen_content.add(sem.page_content)
            if keyw.page_content not in seen_content:
                combined_results.append(keyw)
                seen_content.add(keyw.page_content)
                
            if len(combined_results) >= top_k:
                break
                
        return combined_results[:top_k]
