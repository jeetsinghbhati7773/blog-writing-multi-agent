from __future__ import annotations

import os
import json
import uuid
import sqlite3
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Environment Secrets
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Initialize Supabase Client if keys are configured
_supabase_client = None
if SUPABASE_URL and SUPABASE_KEY and SUPABASE_URL.startswith("http") and "your-project" not in SUPABASE_URL:
    try:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not initialize Supabase client: {e}")

# Guest Fallback Defaults
GUEST_USER_ID = "00000000-0000-0000-0000-000000000000"
GUEST_USER_EMAIL = "guest@localhost"

# Local SQLite Database Path
# Anchor to the project root (not the current working directory) so chat
# history is always read from and written to the SAME chat_history.db file
# regardless of where the app process was launched from. Using os.getcwd()
# meant launching from a different folder pointed at a different (empty) DB,
# which made saved conversations appear to vanish / not load.
try:
    from src.paths import PROJECT_ROOT
    LOCAL_DB_PATH = str(PROJECT_ROOT / "chat_history.db")
except Exception:
    LOCAL_DB_PATH = os.path.join(os.getcwd(), "chat_history.db")

def get_sqlite_conn():
    """Returns a connection to the local SQLite database."""
    conn = sqlite3.connect(LOCAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_local_db():
    """Initializes local SQLite database tables if Supabase is not active."""
    try:
        conn = get_sqlite_conn()
        cur = conn.cursor()
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_calls TEXT,
            created_at TEXT NOT NULL
        );
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT DEFAULT '',
            note TEXT DEFAULT '',
            created_at TEXT NOT NULL
        );
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_memories (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            memory_text TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)
        
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to initialize local SQLite DB: {e}")

# Auto-initialize local SQLite database
init_local_db()


# ---------------------------------------------------------------------
# 1. Auth Helpers
# ---------------------------------------------------------------------
def login_user(email: str, password: str) -> Dict[str, Any]:
    """Authenticate user against Supabase Auth or fallback to local user session."""
    if not email or not password:
        return {"success": False, "error": "Please provide both Email and Password."}

    if not _supabase_client:
        local_user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, email.strip().lower()))
        return {
            "success": True,
            "user_id": local_user_id,
            "email": email.strip(),
            "message": "Logged in (Local Mode)",
        }
    try:
        res = _supabase_client.auth.sign_in_with_password({"email": email.strip(), "password": password})
        if res.user:
            return {
                "success": True,
                "user_id": res.user.id,
                "email": res.user.email,
                "session": res.session,
            }
        return {"success": False, "error": "Invalid credentials"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def signup_user(email: str, password: str) -> Dict[str, Any]:
    """Register a new user account in Supabase Auth or fallback to local user session."""
    if not email or not password:
        return {"success": False, "error": "Please provide both Email and Password."}

    if not _supabase_client:
        local_user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, email.strip().lower()))
        return {
            "success": True,
            "user_id": local_user_id,
            "email": email.strip(),
            "message": "Account created (Local Mode)!",
        }
    try:
        res = _supabase_client.auth.sign_up({"email": email.strip(), "password": password})
        if res.user:
            return {
                "success": True,
                "user_id": res.user.id,
                "email": res.user.email,
                "message": "User registered successfully!",
            }
        return {"success": False, "error": "Signup failed."}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------
# 2. Chat Session Management (Multiple Chats per User)
# ---------------------------------------------------------------------
def create_session(user_id: str, title: str = "New Conversation") -> Optional[Dict[str, Any]]:
    """Create a new chat session for a specific user."""
    sess_id = str(uuid.uuid4())
    now_iso = datetime.now().isoformat()

    if _supabase_client:
        try:
            res = _supabase_client.table("chat_sessions").insert({
                "user_id": user_id,
                "title": title
            }).execute()
            if res.data:
                return res.data[0]
        except Exception as e:
            logger.error(f"Error creating chat session in Supabase: {e}")

    # Fallback to local SQLite DB
    try:
        conn = get_sqlite_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_sessions (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (sess_id, user_id, title, now_iso, now_iso)
        )
        conn.commit()
        conn.close()
        return {"id": sess_id, "user_id": user_id, "title": title, "created_at": now_iso, "updated_at": now_iso}
    except Exception as e:
        logger.error(f"Error creating local chat session: {e}")
        return {"id": sess_id, "user_id": user_id, "title": title, "created_at": now_iso, "updated_at": now_iso}


def list_user_sessions(user_id: str) -> List[Dict[str, Any]]:
    """Fetch all chat sessions for a specific user."""
    if _supabase_client:
        try:
            res = _supabase_client.table("chat_sessions") \
                .select("*") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .execute()
            if res.data:
                return res.data
        except Exception as e:
            logger.error(f"Error listing chat sessions from Supabase: {e}")

    # Fallback to local SQLite DB
    try:
        conn = get_sqlite_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
        rows = [dict(row) for row in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Error listing local chat sessions: {e}")
        return []


def get_session_messages(session_id: str, user_id: str) -> List[Dict[str, Any]]:
    """Retrieve message history for a specific chat session."""
    if _supabase_client:
        try:
            res = _supabase_client.table("chat_messages") \
                .select("*") \
                .eq("session_id", session_id) \
                .eq("user_id", user_id) \
                .order("created_at", desc=False) \
                .execute()
            if res.data:
                return res.data
        except Exception as e:
            logger.error(f"Error retrieving session messages from Supabase: {e}")

    # Fallback to local SQLite DB
    try:
        conn = get_sqlite_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM chat_messages WHERE session_id = ? AND user_id = ? ORDER BY created_at ASC",
            (session_id, user_id)
        )
        rows = [dict(row) for row in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Error retrieving local session messages: {e}")
        return []


def save_message(session_id: str, user_id: str, role: str, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> bool:
    """Persist a user/assistant message to chat_messages table."""
    msg_id = str(uuid.uuid4())
    now_iso = datetime.now().isoformat()

    if _supabase_client:
        try:
            _supabase_client.table("chat_messages").insert({
                "session_id": session_id,
                "user_id": user_id,
                "role": role,
                "content": content,
                "tool_calls": json.dumps(tool_calls) if tool_calls else None
            }).execute()
            
            _supabase_client.table("chat_sessions") \
                .update({"updated_at": now_iso}) \
                .eq("id", session_id) \
                .execute()
            return True
        except Exception as e:
            logger.error(f"Error saving chat message to Supabase: {e}")

    # Fallback to local SQLite DB
    try:
        conn = get_sqlite_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_messages (id, session_id, user_id, role, content, tool_calls, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (msg_id, session_id, user_id, role, content, json.dumps(tool_calls) if tool_calls else None, now_iso)
        )
        cur.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (now_iso, session_id)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error saving local chat message: {e}")
        return False


def delete_session(session_id: str, user_id: str) -> bool:
    """Delete a chat session and its message history."""
    if _supabase_client:
        try:
            _supabase_client.table("chat_sessions") \
                .delete() \
                .eq("id", session_id) \
                .eq("user_id", user_id) \
                .execute()
            return True
        except Exception as e:
            logger.error(f"Error deleting session from Supabase: {e}")

    try:
        conn = get_sqlite_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM chat_messages WHERE session_id = ? AND user_id = ?", (session_id, user_id))
        cur.execute("DELETE FROM chat_sessions WHERE id = ? AND user_id = ?", (session_id, user_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error deleting local session: {e}")
        return False


# ---------------------------------------------------------------------
# 3. Expense Management (User-Isolated Records)
# ---------------------------------------------------------------------
def add_expense_db(user_id: str, amount: float, category: str, date_val: str, subcategory: str = "", note: str = "") -> Dict[str, Any]:
    """Add a new expense record bound to user_id."""
    exp_id = str(uuid.uuid4())
    now_iso = datetime.now().isoformat()

    if _supabase_client:
        try:
            res = _supabase_client.table("expenses").insert({
                "user_id": user_id,
                "amount": amount,
                "category": category,
                "date": date_val,
                "subcategory": subcategory,
                "note": note
            }).execute()
            if res.data:
                return {"success": True, "expense": res.data[0]}
        except Exception as e:
            logger.error(f"Error adding expense to Supabase: {e}")

    # Fallback to local SQLite DB
    try:
        conn = get_sqlite_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO expenses (id, user_id, date, amount, category, subcategory, note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (exp_id, user_id, date_val, amount, category, subcategory, note, now_iso)
        )
        conn.commit()
        conn.close()
        return {"success": True, "expense": {"id": exp_id, "user_id": user_id, "amount": amount, "category": category, "date": date_val, "note": note}}
    except Exception as e:
        logger.error(f"Error adding local expense: {e}")
        return {"success": False, "error": str(e)}


def list_expenses_db(user_id: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """List expenses for user_id within a date range."""
    if _supabase_client:
        try:
            res = _supabase_client.table("expenses") \
                .select("*") \
                .eq("user_id", user_id) \
                .gte("date", start_date) \
                .lte("date", end_date) \
                .order("date", desc=True) \
                .execute()
            if res.data:
                return res.data
        except Exception as e:
            logger.error(f"Error listing expenses from Supabase: {e}")

    # Fallback to local SQLite DB
    try:
        conn = get_sqlite_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM expenses WHERE user_id = ? AND date >= ? AND date <= ? ORDER BY date DESC",
            (user_id, start_date, end_date)
        )
        rows = [dict(row) for row in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Error listing local expenses: {e}")
        return []


def summarize_expenses_db(user_id: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Summarize expenses by category for user_id within a date range."""
    raw_items = list_expenses_db(user_id, start_date, end_date)
    if not raw_items:
        return []
    
    category_sums: Dict[str, Dict[str, Any]] = {}
    for item in raw_items:
        cat = item.get("category", "General").lower()
        amt = float(item.get("amount", 0.0))
        if cat not in category_sums:
            category_sums[cat] = {"category": cat, "total_amount": 0.0, "count": 0}
        category_sums[cat]["total_amount"] += amt
        category_sums[cat]["count"] += 1
        
    return list(category_sums.values())


# ---------------------------------------------------------------------
# 4. Long-Term Semantic Memory
# ---------------------------------------------------------------------
def save_user_memory(user_id: str, memory_text: str, embedding: Optional[List[float]] = None) -> bool:
    """Save a long-term memory fact for user_id."""
    mem_id = str(uuid.uuid4())
    now_iso = datetime.now().isoformat()

    if _supabase_client:
        try:
            data = {"user_id": user_id, "memory_text": memory_text}
            if embedding:
                data["embedding"] = embedding
            _supabase_client.table("user_memories").insert(data).execute()
            return True
        except Exception as e:
            logger.error(f"Error saving user memory to Supabase: {e}")

    try:
        conn = get_sqlite_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO user_memories (id, user_id, memory_text, created_at) VALUES (?, ?, ?, ?)",
            (mem_id, user_id, memory_text, now_iso)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error saving local user memory: {e}")
        return False


def retrieve_user_memories(user_id: str, query_embedding: Optional[List[float]] = None, limit: int = 3) -> List[str]:
    """Retrieve long-term memory facts for user_id."""
    if _supabase_client:
        try:
            if query_embedding:
                res = _supabase_client.rpc("match_user_memories", {
                    "query_embedding": query_embedding,
                    "filter_user_id": user_id,
                    "match_count": limit
                }).execute()
                if res.data:
                    return [row["memory_text"] for row in res.data]
            
            res = _supabase_client.table("user_memories") \
                .select("memory_text") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
            if res.data:
                return [row["memory_text"] for row in res.data]
        except Exception as e:
            logger.error(f"Error retrieving user memories from Supabase: {e}")

    try:
        conn = get_sqlite_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT memory_text FROM user_memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        )
        rows = [row["memory_text"] for row in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Error retrieving local user memories: {e}")
        return []
