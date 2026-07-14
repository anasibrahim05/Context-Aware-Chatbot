"""
app.py
------
Streamlit front-end for the Context-Aware Chatbot.

Run with:
    streamlit run app.py

Features:
- Project title and description
- Sidebar with conversation history and a "Clear chat" button
- Text input box + Send button
- Nicely formatted chat bubbles for user and bot messages
- Shows which source documents were used to answer each question
- Handles missing vector store / empty questions gracefully
"""

import os
import streamlit as st

from chatbot import ChatBot

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Context-Aware RAG Chatbot",
    page_icon="💬",
    layout="wide",
)


# ---------------------------------------------------------------------------
# CACHED CHATBOT LOADER
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_chatbot():
    """
    Loads the ChatBot once and caches it across Streamlit re-runs,
    so the embedding model and LLM are not reloaded on every interaction.
    """
    return ChatBot()


# ---------------------------------------------------------------------------
# SESSION STATE INITIALISATION
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    # Each item is a dict: {"role": "user"/"bot", "content": str, "sources": list}
    st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📚 About this project")
    st.markdown(
        """
        **Context-Aware Chatbot** built with:
        - LangChain
        - FAISS (vector database)
        - Sentence-Transformers embeddings
        - HuggingFace Transformers (flan-t5-base)
        - Streamlit

        Upload documents into the `documents/` folder and run
        `python ingest.py` before chatting.
        """
    )

    st.divider()
    st.subheader("🕑 Conversation History")
    if len(st.session_state.chat_history) == 0:
        st.caption("No messages yet.")
    else:
        for msg in st.session_state.chat_history:
            role_label = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
            st.caption(f"**{role_label}:** {msg['content'][:60]}")

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        # Also clear the LangChain memory buffer, not just the UI history
        if "chatbot_instance" in st.session_state:
            st.session_state.chatbot_instance.clear_memory()
        st.rerun()


# ---------------------------------------------------------------------------
# MAIN AREA
# ---------------------------------------------------------------------------
st.title("💬 Context-Aware Chatbot (LangChain + RAG)")
st.markdown(
    "Ask questions about the documents in the `documents/` folder. "
    "This chatbot remembers previous questions, so you can ask natural "
    "follow-ups like *'Where was he born?'* after asking about a person."
)

# Try to load the chatbot; handle the case where ingest.py hasn't been run yet
try:
    if "chatbot_instance" not in st.session_state:
        with st.spinner("Loading models and vector store... this can take a minute the first time."):
            st.session_state.chatbot_instance = load_chatbot()
    bot = st.session_state.chatbot_instance
    load_error = None
except FileNotFoundError as e:
    bot = None
    load_error = str(e)
except Exception as e:
    bot = None
    load_error = f"Unexpected error while loading the chatbot: {e}"

if load_error:
    st.error(load_error)
    st.info("Run `python ingest.py` in your terminal or Colab cell, then restart this app.")
    st.stop()

# ---------------------------------------------------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------------------------------------------------
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            if msg.get("sources"):
                st.caption(f"📄 Source(s): {', '.join(msg['sources'])}")

# ---------------------------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------------------------
user_question = st.chat_input("Type your question here and press Enter...")

if user_question is not None:
    # Handle empty / whitespace-only questions gracefully
    if not user_question.strip():
        st.warning("Please enter a question before sending.")
    else:
        # Show the user's message immediately
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        # Generate and show the bot's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = bot.ask(user_question)
            st.write(response["answer"])
            if response["sources"]:
                st.caption(f"📄 Source(s): {', '.join(response['sources'])}")

        st.session_state.chat_history.append(
            {
                "role": "bot",
                "content": response["answer"],
                "sources": response["sources"],
            }
        )
