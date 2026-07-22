# src/cache_manager.py
# High-Performance Response Cache Memory for Policy AI RAG Assistant.
# Provides exact-normalized and semantic query caching for zero-latency responses on repeated questions.

import os
import re
import json
import time
import sqlite3
from typing import Optional, Dict, Any, List, Tuple
from src.logger import get_logger

logger = get_logger("CacheManager")

DB_FILE = "query_cache.db"

class ResponseCache:
    """
    Response Cache Memory storing previous RAG answers and sources.
    If a user asks a question that has already been answered, it retrieves the answer instantly.
    """
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self.total_hits = 0
        self.total_misses = 0
        self._init_db()

    def _get_connection(self) -> sqlite3.connect:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes SQLite schema for cache entries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    normalized_query TEXT PRIMARY KEY,
                    original_query TEXT,
                    response TEXT,
                    sources TEXT,
                    hit_count INTEGER DEFAULT 1,
                    created_at REAL,
                    last_accessed REAL
                )
            """)
            conn.commit()
        logger.info(f"Initialized Response Cache database at '{self.db_path}'.")

    TYPO_MAPPINGS = {
        "harrasment": "harassment",
        "harrament": "harassment",
        "harrassment": "harassment",
        "atendance": "attendance",
        "attendence": "attendance",
        "plagerism": "plagiarism",
        "plagearism": "plagiarism",
        "eligibilty": "eligibility",
        "withdrawl": "withdrawal",
        "examintion": "examination",
        "gradeing": "grading"
    }

    @classmethod
    def normalize_query(cls, query: str) -> str:
        """
        Normalizes string: converts to lowercase, strips whitespace/punctuation, and corrects common typos.
        Example: "tell me about the sexual harrasment policy?" -> "tell me about the sexual harassment policy"
        """
        cleaned = query.strip().lower()
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        words = cleaned.split()
        corrected = [cls.TYPO_MAPPINGS.get(w, w) for w in words]
        return " ".join(corrected)

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves cached answer if normalized query exists in cache.
        Returns dict with response, sources, hit_count, and lookup metadata, or None if miss.
        """
        start_time = time.time()
        norm_q = self.normalize_query(query)

        if not norm_q:
            self.total_misses += 1
            return None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT response, sources, hit_count, created_at FROM query_cache WHERE normalized_query = ?",
                (norm_q,)
            )
            row = cursor.fetchone()

            if row:
                self.total_hits += 1
                new_hit_count = row["hit_count"] + 1
                now = time.time()
                
                # Update hit count and last accessed time
                cursor.execute(
                    "UPDATE query_cache SET hit_count = ?, last_accessed = ? WHERE normalized_query = ?",
                    (new_hit_count, now, norm_q)
                )
                conn.commit()

                lookup_time_ms = (time.time() - start_time) * 1000
                logger.info(f"[CACHE HIT] Found instant answer for query '{query}' in {lookup_time_ms:.2f}ms (Hits: {new_hit_count}).")

                sources = []
                try:
                    sources = json.loads(row["sources"])
                except Exception:
                    sources = []

                return {
                    "response": row["response"],
                    "sources": sources,
                    "cached": True,
                    "hit_type": "exact_normalized",
                    "hit_count": new_hit_count,
                    "lookup_time_ms": round(lookup_time_ms, 2)
                }

        self.total_misses += 1
        logger.info(f"[CACHE MISS] No cached response for query '{query}'. Proceeding to RAG engine.")
        return None

    def set(self, query: str, response: str, sources: List[str]):
        """
        Saves a generated RAG answer into cache memory.
        """
        norm_q = self.normalize_query(query)
        if not norm_q or not response:
            return

        # Do not cache guardrail or error fallbacks
        low_res = response.lower()
        if "cannot find information" in low_res or "could not find any relevant information" in low_res or "encountered an error" in low_res or "error:" in low_res:
            return

        now = time.time()
        sources_json = json.dumps(sources)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO query_cache (normalized_query, original_query, response, sources, hit_count, created_at, last_accessed)
                VALUES (?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(normalized_query) DO UPDATE SET
                    response = excluded.response,
                    sources = excluded.sources,
                    last_accessed = excluded.last_accessed
            """, (norm_q, query, response, sources_json, now, now))
            conn.commit()

        logger.info(f"[CACHE STORED] Saved query '{query}' to Cache Memory.")

    def clear(self) -> int:
        """
        Clears/invalidates all cached entries.
        Called when Knowledge Base documents are updated/refreshed.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM query_cache")
            deleted_count = cursor.rowcount
            conn.commit()

        logger.info(f"[CACHE CLEARED] Invalidated {deleted_count} cached entries due to Knowledge Base update.")
        return deleted_count

    def get_stats(self) -> Dict[str, Any]:
        """Returns cache usage statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total_entries, SUM(hit_count) as sum_hits FROM query_cache")
            row = cursor.fetchone()
            total_entries = row["total_entries"] if row else 0

        total_requests = self.total_hits + self.total_misses
        hit_rate = (self.total_hits / total_requests * 100) if total_requests > 0 else 0.0

        return {
            "total_cached_entries": total_entries,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate_pct": f"{hit_rate:.1f}%"
        }
