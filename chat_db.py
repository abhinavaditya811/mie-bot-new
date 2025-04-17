import sqlite3
import uuid
from typing import List, Dict

DB_FILE = "chat_history.db"

# --- Initialize the DB ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            session_id TEXT,
            role TEXT,
            content TEXT
        )
    """)
    conn.commit()
    conn.close()

# --- Save a single message ---
def save_message(session_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO chats VALUES (?, ?, ?)", (session_id, role, content))
    conn.commit()
    conn.close()

# --- Load all messages for a session ---
def load_chat(session_id: str) -> List[Dict[str, str]]:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM chats WHERE session_id = ?", (session_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]

def get_all_sessions() -> list[str]:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT DISTINCT session_id FROM chats ORDER BY rowid DESC")
    sessions = [row[0] for row in c.fetchall()]
    conn.close()
    return sessions

def get_session_preview(session_id: str, limit=1) -> str:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT content FROM chats WHERE session_id = ? AND role = 'user' LIMIT ?", (session_id, limit))
    row = c.fetchone()
    conn.close()
    return row[0] if row else "(no message)"

def delete_chat(session_id: str) -> bool:
    """Delete all messages for a specific chat session"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM chats WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error deleting chat: {e}")
        return False