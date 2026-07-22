# tests/test_cache_and_history.py
# Automated verification test script for Response Cache Memory and Persistent Chat History.

import time
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache_manager import ResponseCache
from src.history_manager import ChatHistoryManager

def test_response_cache():
    print("=== Testing Response Cache Memory ===")
    cache = ResponseCache("test_cache.db")
    cache.clear()

    query1 = "What is the minimum passing grade for undergraduate students?"
    query1_normalized = "what is the minimum passing grade for undergraduate students"
    answer1 = "Undergraduate students must maintain a minimum C grade (2.0 GPA)."
    sources1 = ["Academic_Catalog.pdf (Page 5)"]

    # 1. Verify initial lookup is a Cache Miss
    miss_res = cache.get(query1)
    assert miss_res is None, f"Expected None on first query, got {miss_res}"
    print("[PASS] Initial query resulted in expected CACHE MISS.")

    # 2. Save entry to cache
    cache.set(query1, answer1, sources1)

    # 3. Verify exact query lookup is a Cache Hit (< 50ms)
    t0 = time.time()
    hit_res = cache.get(query1)
    t1 = time.time()
    latency_ms = (t1 - t0) * 1000

    assert hit_res is not None, "Expected cache hit, got None"
    assert hit_res["response"] == answer1, f"Expected '{answer1}', got '{hit_res['response']}'"
    assert hit_res["sources"] == sources1, f"Expected {sources1}, got {hit_res['sources']}"
    print(f"[PASS] Exact query CACHE HIT verified! Latency: {latency_ms:.2f}ms (Saved full LLM call).")

    # 4. Verify normalized query lookup (different casing, trailing whitespace/punctuation)
    query1_variant = "  WHAT IS THE MINIMUM PASSING GRADE FOR UNDERGRADUATE STUDENTS?  "
    hit_variant = cache.get(query1_variant)
    assert hit_variant is not None, "Expected cache hit for query variant with different casing & punctuation"
    assert hit_variant["response"] == answer1
    print("[PASS] Case & Punctuation normalized CACHE HIT verified!")

    # 5. Test Cache Stats
    stats = cache.get_stats()
    print(f"[PASS] Cache Stats: {stats}")
    assert stats["total_hits"] >= 2

    # 6. Test Cache Invalidation
    cleared_count = cache.clear()
    assert cleared_count >= 1, "Expected at least 1 entry cleared"
    after_clear = cache.get(query1)
    assert after_clear is None, "Expected None after clearing cache"
    print("[PASS] Cache Invalidation verified successfully!")

    # Clean up test DB file
    if os.path.exists("test_cache.db"):
        os.remove("test_cache.db")

def test_chat_history():
    print("\n=== Testing Chat History Persistence ===")
    history = ChatHistoryManager("test_history.db")

    session_id = history.create_session(title="Academic Policy Session")
    
    history.add_message(session_id, "user", "What is the attendance policy?")
    history.add_message(session_id, "assistant", "Students cannot exceed 3 unexcused absences.", sources=["Attendance.pdf (Page 2)"], cached=False)
    history.add_message(session_id, "user", "What is the attendance policy?")
    history.add_message(session_id, "assistant", "Students cannot exceed 3 unexcused absences.", sources=["Attendance.pdf (Page 2)"], cached=True)

    messages = history.get_session_messages(session_id)
    assert len(messages) == 4, f"Expected 4 stored messages, found {len(messages)}"
    assert messages[3]["cached"] == True, "Expected 4th message to have cached=True"
    print(f"[PASS] Successfully stored and retrieved {len(messages)} messages from session history DB.")

    sessions = history.list_recent_sessions()
    assert len(sessions) >= 1, "Expected at least 1 active session in history"
    print(f"[PASS] Listed recent sessions: {sessions[0]['title']} ({sessions[0]['message_count']} msgs).")

    deleted = history.delete_session(session_id)
    assert deleted == True, "Expected successful session deletion"
    print("[PASS] Session deletion verified!")

    if os.path.exists("test_history.db"):
        os.remove("test_history.db")

if __name__ == "__main__":
    test_response_cache()
    test_chat_history()
    print("\n[SUCCESS] All Cache Memory & Chat History unit tests passed successfully!")
