# PYTHON 3.12 FIX (must come before any torch/streamlit)
import asyncio
import sys

if sys.platform in ("linux", "darwin") and sys.version_info >= (3, 12):
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# -------------------------------
# Imports
# -------------------------------
import streamlit as st
from chatbot_backend import process_chat
from chat_db import (
    init_db,
    save_message,
    load_chat,
    get_all_sessions,
    get_session_preview
)
import uuid
import time

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="MIE Chatbot — Northeastern University", layout="wide")
st.title("MIE Chatbot — Northeastern University")

# -------------------------------
# Initialize SQLite DB
# -------------------------------
init_db()

# -------------------------------
# Initialize Session State
# -------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = load_chat(st.session_state.session_id)
    st.session_state.chat_history = [
        m["content"] for m in st.session_state.messages if m["role"] == "user"
    ]

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("Chat Sessions")

    # New Chat Button
    if st.button("+ New Chat"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    # Past Sessions with Clickable Preview
    st.subheader("Past Chats")
    all_sessions = get_all_sessions()
    for sid in all_sessions:
        preview = get_session_preview(sid)
        label = preview if len(preview) < 50 else preview[:47] + "..."
        button_label = f"{sid[:8]}... — {label}"
        if st.button(button_label, key=sid):
            st.session_state.session_id = sid
            st.session_state.messages = load_chat(sid)
            st.session_state.chat_history = [
                m["content"] for m in st.session_state.messages if m["role"] == "user"
            ]
            st.rerun()

    # Current Session History
    st.subheader("This Session")
    if st.session_state.chat_history:
        for i, msg in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"**{i}.** {msg}")
    else:
        st.info("Start a conversation above.")

    # Clear current session
    if st.button("Clear this session"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# -------------------------------
# Display Chat History
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# Handle New Message
# -------------------------------
user_input = st.chat_input("Ask about MIE programs, faculty, labs, or policies...")

if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append(user_input)
    save_message(st.session_state.session_id, "user", user_input)

    # Get assistant response
    response = process_chat(user_input, st.session_state.chat_history[:-1])

    # Typing animation for assistant
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for char in response:
            full_response += char
            placeholder.markdown(full_response + "▌")
            time.sleep(0.01)
        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    save_message(st.session_state.session_id, "assistant", response)