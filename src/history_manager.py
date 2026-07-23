# src/history_manager.py
# Persistent Chat History and Session Management for Policy AI Assistant.
# Stores chat sessions and messages in SQLite DB to persist conversation history across page reloads.

import json
import time
import uuid
import sqlite3
from typing import List, Dict, Any, Optional
from src.logger import get_logger

logger = get_logger("HistoryManager")

DB_FILE = "chat_history.db"

class ChatHistoryManager:
    """
    Manages persistent chat sessions and message logs in SQLite database.
    """
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.connect:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes SQLite schema for sessions and messages."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    sources TEXT,
                    cached INTEGER DEFAULT 0,
                    timestamp REAL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)
            conn.commit()
        logger.info(f"Initialized Chat History database at '{self.db_path}'.")

    def create_session(self, session_id: Optional[str] = None, title: str = "New Chat") -> str:
        """Creates a new chat session."""
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:10]}"

        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO sessions (session_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (session_id, title, now, now))
            conn.commit()

        logger.info(f"Created chat session '{session_id}' with title '{title}'.")
        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[str]] = None,
        cached: bool = False
    ):
        """Adds a message (user or assistant) to a session."""
        if not content:
            return

        # Ensure session exists
        now = time.time()
        sources_json = json.dumps(sources or [])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if session exists; if not, create it
            cursor.execute("SELECT session_id, title FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                title = content[:35] + "..." if len(content) > 35 else content
                cursor.execute("""
                    INSERT INTO sessions (session_id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (session_id, title, now, now))
            else:
                # Update title if it's the default "New Chat" and message is from user
                if role == "user" and row["title"] in ["New Chat", "Untitled Session"]:
                    title = content[:35] + "..." if len(content) > 35 else content
                    cursor.execute(
                        "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                        (title, now, session_id)
                    )
                else:
                    cursor.execute(
                        "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                        (now, session_id)
                    )

            # Insert message
            cursor.execute("""
                INSERT INTO messages (session_id, role, content, sources, cached, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, role, content, sources_json, 1 if cached else 0, now))
            
            conn.commit()

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves all messages for a given session sorted by timestamp."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, role, content, sources, cached, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
            """, (session_id,))
            rows = cursor.fetchall()

        messages = []
        for r in rows:
            sources = []
            try:
                sources = json.loads(r["sources"]) if r["sources"] else []
            except Exception:
                sources = []

            messages.append({
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "sources": sources,
                "cached": bool(r["cached"]),
                "timestamp": r["timestamp"]
            })
        return messages

    def list_recent_sessions(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Lists recent chat sessions ordered by latest update time."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.session_id, s.title, s.created_at, s.updated_at, COUNT(m.id) as msg_count
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()

        sessions = []
        for r in rows:
            sessions.append({
                "session_id": r["session_id"],
                "title": r["title"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "message_count": r["msg_count"]
            })
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Deletes a session and all its messages."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_all_sessions(self) -> int:
        """Deletes all chat sessions and messages from persistent storage."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages")
            cursor.execute("DELETE FROM sessions")
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info("Cleared all chat sessions and message history.")
            return deleted_count
