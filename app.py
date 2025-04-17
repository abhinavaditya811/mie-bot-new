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
    get_session_preview,
    delete_chat
)
from pdf_qa import process_pdf, answer_question
import uuid
import time
import os

# Set OpenAI API key from environment variable or Streamlit secrets
import openai
openai.api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

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
    st.session_state.pdf_data = None
    st.session_state.pdf_mode = False

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("Chat Sessions")

    # PDF Upload Section
    st.subheader("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            st.session_state.pdf_data = process_pdf(uploaded_file)
            st.session_state.pdf_filename = uploaded_file.name
        
        # Check if the document is related to Northeastern
        if st.session_state.pdf_data.get("is_northeastern_related", False):
            st.success(f"Processed document: {uploaded_file.name}")
            st.session_state.pdf_mode = True
        else:
            st.warning(f"The document '{uploaded_file.name}' does not appear to be related to Northeastern University. I can only answer questions about Northeastern-related documents.")
            st.session_state.pdf_mode = False
            st.session_state.pdf_data = None
    
    # Toggle between regular chat and PDF Q&A
    if st.session_state.pdf_data:
        st.session_state.pdf_mode = st.toggle(
            "Use uploaded document for answers", 
            value=st.session_state.pdf_mode
        )
    
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
        
        # Create a columns layout for each session
        col1, col2 = st.columns([4, 1])
        
        # Load button in first column
        with col1:
            if st.button(f"{sid[:8]}... — {label}", key=f"load_{sid}"):
                st.session_state.session_id = sid
                st.session_state.messages = load_chat(sid)
                st.session_state.chat_history = [
                    m["content"] for m in st.session_state.messages if m["role"] == "user"
                ]
                st.rerun()
        
        # Delete button in second column
        with col2:
            if st.button("Delete", key=f"delete_{sid}"):
                if delete_chat(sid):
                    # If the deleted session was the current one, create a new session
                    if st.session_state.session_id == sid:
                        st.session_state.session_id = str(uuid.uuid4())
                        st.session_state.messages = []
                        st.session_state.chat_history = []
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
mode_indicator = "document" if st.session_state.pdf_mode else "MIE programs"
user_input = st.chat_input(f"Ask about {mode_indicator}...")

if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append(user_input)
    save_message(st.session_state.session_id, "user", user_input)

    # Process message based on mode
    if st.session_state.pdf_mode and st.session_state.pdf_data:
        # PDF Q&A mode with LLM
        with st.spinner("Analyzing document..."):
            response = answer_question(user_input, st.session_state.pdf_data)
    else:
        # Regular chatbot mode
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