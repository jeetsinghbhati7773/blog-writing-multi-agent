import pytest
from src.db import (
    GUEST_USER_ID,
    GUEST_USER_EMAIL,
    create_session,
    list_user_sessions,
    get_session_messages,
    save_message,
    delete_session,
    add_expense_db,
    list_expenses_db,
    summarize_expenses_db,
    save_user_memory,
    retrieve_user_memories,
)

def test_guest_defaults():
    assert GUEST_USER_ID == "00000000-0000-0000-0000-000000000000"
    assert GUEST_USER_EMAIL == "guest@localhost"

def test_session_management_fallback():
    session = create_session(GUEST_USER_ID, "Test Chat")
    assert session is not None
    assert session.get("user_id") == GUEST_USER_ID
    
    sessions = list_user_sessions(GUEST_USER_ID)
    assert isinstance(sessions, list)
    assert len(sessions) > 0

    msgs = get_session_messages(session["id"], GUEST_USER_ID)
    assert isinstance(msgs, list)

    saved = save_message(session["id"], GUEST_USER_ID, "user", "Hello database")
    assert isinstance(saved, bool)

    deleted = delete_session(session["id"], GUEST_USER_ID)
    assert deleted is True

def test_expense_db_fallback():
    res = add_expense_db(GUEST_USER_ID, 50.0, "Food", "2026-08-23", note="Test")
    assert isinstance(res, dict)

    items = list_expenses_db(GUEST_USER_ID, "2026-08-01", "2026-08-30")
    assert isinstance(items, list)

    summary = summarize_expenses_db(GUEST_USER_ID, "2026-08-01", "2026-08-30")
    assert isinstance(summary, list)

def test_user_memory_fallback():
    saved = save_user_memory(GUEST_USER_ID, "User prefers vegetarian food")
    assert isinstance(saved, bool)

    memories = retrieve_user_memories(GUEST_USER_ID)
    assert isinstance(memories, list)
