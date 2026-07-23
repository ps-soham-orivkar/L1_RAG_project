# src/reranker.py
# Cross-Encoder Reranker using BAAI/bge-reranker-base (configured for CPU execution with Singleton Caching).

import time
from sentence_transformers import CrossEncoder
from src.logger import get_logger

logger = get_logger("Reranker")

_reranker_instance = None

def get_reranker(model_name="BAAI/bge-reranker-base"):
    """
    Singleton getter: Loads CrossEncoder model weights EXACTLY ONCE into memory.
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker(model_name=model_name)
    return _reranker_instance

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.model_name = model_name
        logger.info(f"[RERANKER INIT] Loading Cross-Encoder model: '{model_name}' on CPU...")
        try:
            self.model = CrossEncoder(model_name, device="cpu")
            logger.info("[RERANKER INIT] Cross-Encoder model loaded successfully.")
        except Exception as e:
            logger.error(f"[RERANKER INIT] Failed to load model '{model_name}': {e}. Falling back to ms-marco-MiniLM-L-6-v2.")
            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    def rerank(self, query, chunks, top_n=4, score_threshold=-2.0):
        """
        Reranks candidate document chunks based on joint query relevance.
        Runs 100% on CPU.
        """
        if not chunks:
            logger.info("[STEP 4: RERANKING] Empty candidate list passed to Reranker.")
            return [], []

        start_time = time.time()
        logger.info(f"[STEP 4: RERANKING] Re-scoring {len(chunks)} candidate chunks against query...")
        
        pairs = [[query, doc.page_content] for doc in chunks]
        scores = self.model.predict(pairs)
        elapsed = time.time() - start_time
        
        # Pair documents with scores
        doc_score_pairs = list(zip(chunks, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"[STEP 4: RERANKING] Reranking complete in {elapsed:.3f}s. Top candidates after re-scoring:")
        for idx, (doc, score) in enumerate(doc_score_pairs[:top_n]):
            src = doc.metadata.get('source_name', 'Doc')
            pg = doc.metadata.get('page', 0)
            pg_str = str(pg + 1) if isinstance(pg, int) else str(pg)
            snippet = doc.page_content[:55].replace('\n', ' ')
            logger.info(f"   Rank #{idx+1} [Re-rank Score: {score:+.4f}] -> {src} (Pg {pg_str}): \"{snippet}...\"")

        # Filter by threshold and cut to top_n
        filtered = [(doc, score) for doc, score in doc_score_pairs if score >= score_threshold]
        if not filtered:
            logger.warning(f"[STEP 4: RERANKING] All chunks scored below confidence threshold ({score_threshold}).")
            return [], []

        top_pairs = filtered[:top_n]
        reranked_chunks = [p[0] for p in top_pairs]
        reranked_scores = [float(p[1]) for p in top_pairs]
        
        return reranked_chunks, reranked_scores
