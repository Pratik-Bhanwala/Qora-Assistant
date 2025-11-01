"""Streamlit UI for a LangGraph + Ollama powered conversational assistant."""

from __future__ import annotations

from datetime import datetime
import time
from html import escape
from uuid import uuid4

import streamlit as st

from backend.config import settings
from backend.langgraph_agent import PersonaAgent, format_history_for_display
from backend.memory_manager import load_memory, save_memory


st.set_page_config(page_title="Qora Assistant", page_icon="💬", layout="wide")


@st.cache_resource(show_spinner=False)
def load_agent() -> PersonaAgent:
    return PersonaAgent()


agent = load_agent()

def create_chat() -> dict:
    now = datetime.utcnow().isoformat()
    return {
        "id": str(uuid4()),
        "title": "New chat",
        "messages": [],
        "diagnostics": [],
        "created": now,
        "updated": now,
    }


def get_chat(chat_id: str) -> dict:
    for chat in st.session_state.chats:
        if chat["id"] == chat_id:
            return chat
    if st.session_state.chats:
        return st.session_state.chats[0]
    chat = create_chat()
    st.session_state.chats.append(chat)
    return chat


def persist_session() -> None:
    chat = get_chat(st.session_state.chat_id)
    chat["messages"] = list(st.session_state.conversation)
    chat["diagnostics"] = list(st.session_state.diagnostics)
    if chat["messages"]:
        first_user = next(
            (m["content"].strip() for m in chat["messages"] if m["role"] == "user" and m["content"].strip()),
            "",
        )
        if first_user:
            title = first_user[:48]
            if len(first_user) > 48:
                title += "…"
            chat["title"] = title
    else:
        chat["title"] = "New chat"
    chat["updated"] = datetime.utcnow().isoformat()
    save_memory(st.session_state.chats)


def select_chat(chat_id: str) -> None:
    if chat_id == st.session_state.chat_id:
        return
    persist_session()
    st.session_state.chat_id = chat_id
    chat = get_chat(chat_id)
    st.session_state.conversation = list(chat.get("messages", []))
    st.session_state.diagnostics = list(chat.get("diagnostics", []))


def new_chat() -> None:
    persist_session()
    chat = create_chat()
    st.session_state.chats.insert(0, chat)
    st.session_state.chat_id = chat["id"]
    st.session_state.conversation = []
    st.session_state.diagnostics = []
    save_memory(st.session_state.chats)


if "chats" not in st.session_state:
    st.session_state.chats = load_memory()
    if not st.session_state.chats:
        st.session_state.chats = [create_chat()]
        save_memory(st.session_state.chats)

# Always start the session on a brand-new chat (only once per launch)
if "launched_initialized" not in st.session_state:
    chat = create_chat()
    st.session_state.chats.insert(0, chat)
    st.session_state.chat_id = chat["id"]
    st.session_state.conversation = []
    st.session_state.diagnostics = []
    st.session_state.launched_initialized = True

if "chat_id" not in st.session_state:
    st.session_state.chat_id = st.session_state.chats[0]["id"]

if "conversation" not in st.session_state:
    current_chat = get_chat(st.session_state.chat_id)
    st.session_state.conversation = list(current_chat.get("messages", []))

if "diagnostics" not in st.session_state:
    current_chat = get_chat(st.session_state.chat_id)
    st.session_state.diagnostics = list(current_chat.get("diagnostics", []))

if "model_name" not in st.session_state:
    st.session_state.model_name = settings.ollama_model

if "ollama_host" not in st.session_state:
    st.session_state.ollama_host = settings.ollama_host

if "ollama_api_key" not in st.session_state:
    st.session_state.ollama_api_key = settings.ollama_api_key or ""

if "show_thinking" not in st.session_state:
    st.session_state.show_thinking = False

connection_details = agent.set_connection(
    st.session_state.ollama_host, st.session_state.ollama_api_key
)
st.session_state.ollama_host = connection_details["host"] or st.session_state.ollama_host
agent.set_model(st.session_state.model_name)


def apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f5f7ff 0%, #ffffff 45%, #f3f4f6 100%);
                color: #0f172a;
            }
            .stApp:before {
                content:"";
                position: fixed; inset: 0; pointer-events: none;
                background: radial-gradient(900px 600px at 92% 8%, rgba(59,130,246,0.10), transparent 60%),
                            radial-gradient(700px 480px at 8% 90%, rgba(99,102,241,0.10), transparent 60%);
            }
            section[data-testid="stSidebar"] {
                background: rgba(247, 249, 255, 0.92);
                border-right: 1px solid #e2e8f0;
                backdrop-filter: blur(12px);
            }
            section[data-testid="stSidebar"] .block-container {
                padding-top: 1.6rem;
                padding-bottom: 2rem;
            }
            section[data-testid="stSidebar"] button {
                border-radius: 12px;
                border: 1px solid #d7defe;
                background: #ffffff;
                color: #1f2937;
                font-weight: 600;
                box-shadow: 0 4px 14px rgba(79, 70, 229, 0.08);
            }
            section[data-testid="stSidebar"] button:hover {
                background: #eef2ff;
                border-color: #c7d2fe;
            }
            .sidebar-title {
                font-size: 1.05rem;
                font-weight: 700;
                color: #1f2a44;
                margin-bottom: 0.75rem;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .sidebar-card {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid #e0e7ff;
                border-radius: 18px;
                padding: 16px 18px;
                margin-top: 1.2rem;
                box-shadow: 0 18px 36px rgba(99,102,241,0.10);
            }
            .sidebar-card-title {
                font-size: 0.78rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #475569;
                margin-bottom: 0.65rem;
            }
            .sidebar-tags {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }
            .sidebar-tags span {
                background: #eef2ff;
                color: #1e3a8a;
                border-radius: 999px;
                padding: 4px 12px;
                font-size: 0.75rem;
                font-weight: 600;
            }
            .sidebar-metric {
                display: flex;
                justify-content: space-between;
                font-size: 0.88rem;
                color: #475569;
                margin-bottom: 8px;
            }
            .sidebar-metric strong {
                color: #1f2937;
            }
            div[data-testid="stChatMessage"] {
                max-width: 880px;
                margin-bottom: 20px;
                border-radius: 20px;
                border: 1px solid #e2e8f0;
                padding: 20px 24px;
                box-shadow: 0 18px 36px rgba(15, 23, 42, 0.05);
                background: #ffffff;
            }
            div[data-testid="stChatMessageUser"] {
                background: linear-gradient(135deg, #2563eb, #1d4ed8);
                color: #ffffff;
                border: none;
                box-shadow: 0 16px 30px rgba(37, 99, 235, 0.18);
            }
            div[data-testid="stChatMessageUser"] p {
                color: #ffffff !important;
            }
            div[data-testid="stChatMessageAssistant"] {
                background: #ffffff;
            }
            .thinking-card {
                background: #f8f9fb !important;
                border: 1px dashed #d1d9e6 !important;
                border-radius: 16px !important;
                padding: 14px 16px !important;
                color: #8595a8 !important;
                font-size: 0.9rem !important;
                font-style: italic !important;
                font-weight: 400 !important;
                line-height: 1.5 !important;
            }
            .thinking-card .thinking-label {
                display: block;
                font-style: normal !important;
                font-weight: 700 !important;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 0.72rem !important;
                margin-bottom: 6px;
                color: #64748b !important;
            }
            .thinking-card .thinking-body {
                font-style: italic !important;
                color: #8595a8 !important;
            }
            .chat-header {
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 0.15rem;
                color: #111827;
            }
            .chat-subheader {
                color: #5b6785;
                margin-bottom: 1.5rem;
                font-size: 0.95rem;
            }
            .stChatInput textarea {
                border-radius: 999px;
                border: 1px solid #d1d5db;
                padding: 12px 16px;
                font-size: 1rem;
                background: #ffffff;
                box-shadow: 0 20px 36px rgba(15, 23, 42, 0.08);
            }
            [data-testid="stChatInputSubmitButton"] {
                border-radius: 999px;
                background: linear-gradient(135deg, #2563eb, #4338ca);
                color: #ffffff;
                border: none;
                width: 46px; height: 46px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 14px 28px rgba(37, 99, 235, 0.15);
            }
            [data-testid="stChatInputSubmitButton"]:hover {
                background: linear-gradient(135deg, #1d4ed8, #312e81);
            }
            .qora-badge {
                position: fixed;
                right: 18px;
                bottom: 16px;
                background: #0f172a;
                color: #ffffff;
                border-radius: 999px;
                padding: 8px 14px;
                font-weight: 600;
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.25);
                letter-spacing: 0.08em;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
            .qora-badge { position: fixed; right: 18px; bottom: 16px; background: #111827; color: #fff; border-radius: 999px; padding: 8px 12px; font-weight: 600; box-shadow: 0 10px 24px rgba(0,0,0,0.15); }
        </style>
        <div class="qora-badge">Qora Assistant · Built by Pratik</div>
        """,
        unsafe_allow_html=True,
    )


apply_custom_styles()


with st.sidebar:
    st.markdown("<div class='sidebar-title'>Chats</div>", unsafe_allow_html=True)
    st.button("＋ New chat", use_container_width=True, on_click=new_chat, type="primary")

    chat_options = [chat["id"] for chat in st.session_state.chats]
    current_index = chat_options.index(st.session_state.chat_id)
    selected_chat = st.radio(
        "Chat history",
        options=chat_options,
        index=current_index,
        format_func=lambda chat_id: get_chat(chat_id).get("title", "New chat"),
        label_visibility="collapsed",
        key="chat-history",
    )
    if selected_chat != st.session_state.chat_id:
        select_chat(selected_chat)
        st.rerun()

    if st.button("🗑 Delete this chat", use_container_width=True, key="delete-chat"):
        if st.session_state.chats:
            target_id = st.session_state.chat_id
            st.session_state.chats = [chat for chat in st.session_state.chats if chat["id"] != target_id]
            if st.session_state.chats:
                st.session_state.chat_id = st.session_state.chats[0]["id"]
                st.session_state.conversation = list(st.session_state.chats[0].get("messages", []))
                st.session_state.diagnostics = list(st.session_state.chats[0].get("diagnostics", []))
            else:
                chat = create_chat()
                st.session_state.chats = [chat]
                st.session_state.chat_id = chat["id"]
                st.session_state.conversation = []
                st.session_state.diagnostics = []
            save_memory(st.session_state.chats)
            st.toast("Chat deleted")
            st.rerun()

    if st.button("🧹 Clear messages", use_container_width=True, key="clear-chat"):
        st.session_state.conversation = []
        st.session_state.diagnostics = []
        persist_session()
        st.toast("Current chat cleared")
        st.rerun()

    stop_disabled = not st.session_state.get("is_generating")
    if st.button("⏹ Stop generation", use_container_width=True, key="force-stop", disabled=stop_disabled):
        st.session_state.stop_requested = True
        st.toast("Stopping response…")

    model_card = f"""
    <div class='sidebar-card'>
        <div class='sidebar-card-title'>Model in use</div>
        <div class='sidebar-metric'><span>Model</span><strong>{escape(st.session_state.model_name)}</strong></div>
        <div class='sidebar-metric'><span>Host</span><strong>{escape(st.session_state.ollama_host)}</strong></div>
        <div class='sidebar-metric'><span>Chats saved</span><strong>{len(st.session_state.chats)}</strong></div>
    </div>
    """
    st.markdown(model_card, unsafe_allow_html=True)

    tech_items = ["Ollama Cloud", "LangGraph", "Streamlit", "Python 3.11", "Pydantic"]
    tech_tags = "".join(f"<span>{escape(item)}</span>" for item in tech_items)
    tech_card = f"""
    <div class='sidebar-card'>
        <div class='sidebar-card-title'>Tech stack</div>
        <div class='sidebar-tags'>{tech_tags}</div>
    </div>
    """
    st.markdown(tech_card, unsafe_allow_html=True)

    help_card = """
    <div class='sidebar-card'>
        <div class='sidebar-card-title'>Tip</div>
        <p style="margin:0; color:#475569; font-size:0.87rem;">
            Use “＋ New chat” to start a fresh conversation. Chats are saved locally and you can remove any thread with the delete button above.
        </p>
    </div>
    """
    st.markdown(help_card, unsafe_allow_html=True)


current_chat = get_chat(st.session_state.chat_id)
header_title = current_chat.get("title") or "New chat"
if header_title.lower() == "new chat":
    header_title = "Ready when you are."

st.markdown(f"<div class='chat-header'>{header_title}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='chat-subheader'>Qora Assistant · Built by Pratik</div>",
    unsafe_allow_html=True,
)


def send_suggestion(prompt_text: str) -> None:
    handle_user_message(prompt_text)
    st.rerun()


def _render_message_with_code(container, text: str) -> None:
    """Render a chat message with markdown and code blocks exactly once."""
    import re

    pattern = re.compile(r"```(\w+)?\n([\s\S]*?)```", re.MULTILINE)
    pos = 0
    for match in pattern.finditer(text):
        start, end = match.span()
        if start > pos:
            container.markdown(text[pos:start])
        lang = (match.group(1) or "").strip() or None
        code = match.group(2)
        container.code(code, language=lang)
        container.download_button(
            label="Download code",
            data=code,
            file_name=f"snippet.{lang or 'txt'}",
            mime="text/plain",
        )
        pos = end
    if pos < len(text):
        container.markdown(text[pos:])


def _sanitize_tags(text: str) -> str:
    """Remove <thinking> and <answer> tags from text."""
    import re
    return re.sub(r"</?(thinking|answer)>", "", text)


def _strip_reasoning_overlap(reasoning: str, answer: str) -> str:
    """Trim duplicated answer text from the end of the reasoning block."""

    reasoning_clean = reasoning.strip()
    answer_clean = answer.strip()
    if not answer_clean or not reasoning_clean:
        return reasoning_clean

    if reasoning_clean.endswith(answer_clean):
        reasoning_clean = reasoning_clean[: -len(answer_clean)].rstrip()

    return reasoning_clean


def _thinking_html(text: str, *, completed: bool) -> str:
    label = "Thought process" + (" (complete)" if completed else " (in progress)")
    body = escape(text).replace("\n", "<br>") if text else "<em>Thinking…</em>"
    return (
        "<div class='thinking-card'>"
        f"<span class='thinking-label'>{label}</span>"
        f"<div class='thinking-body'>{body}</div>"
        "</div>"
    )


suggestions = [
    "Explain a complex topic in simple terms",
    "Summarise where we left off",
    "Draft a polite follow-up email",
    "Brainstorm project ideas",
]
cols = st.columns(len(suggestions))
for idx, suggestion in enumerate(suggestions):
    if cols[idx].button(suggestion, key=f"suggestion-{idx}"):
        send_suggestion(suggestion)


chat_container = st.container()

with chat_container:
    formatted_history = format_history_for_display(st.session_state.conversation)
    if not formatted_history:
        st.info("Ask me anything to get started.", icon="💬")
    for message in formatted_history:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.markdown(content)


def handle_user_message(prompt: str) -> None:
    with st.chat_message("user"):
        st.markdown(prompt)

    assistant_container = st.chat_message("assistant")
    thinking_card = assistant_container.container()
    thinking_body = thinking_card.empty()
    text_placeholder = assistant_container.empty()

    thinking_body.markdown(_thinking_html("", completed=False), unsafe_allow_html=True)

    previous_history = list(st.session_state.conversation)
    assembled = ""
    st.session_state.stop_requested = False
    answer_started = False

    with st.spinner("Thinking…"):
        try:
            stream = agent.stream_reply(previous_history, prompt)
            thinking_buffer = ""
            for event in stream:
                if event.get("thinking"):
                    lines = event.get("thinking") or []
                    if lines:
                        thinking_buffer = "\n".join(lines)
                        reasoning = _sanitize_tags(thinking_buffer)
                        if answer_started:
                            reasoning = _strip_reasoning_overlap(reasoning, _sanitize_tags(assembled))
                        thinking_body.markdown(
                            _thinking_html(reasoning, completed=answer_started),
                            unsafe_allow_html=True,
                        )
                if event.get("thinking_delta"):
                    thinking_buffer += event["thinking_delta"]
                    reasoning = _sanitize_tags(thinking_buffer)
                    if answer_started:
                        reasoning = _strip_reasoning_overlap(reasoning, _sanitize_tags(assembled))
                    thinking_body.markdown(
                        _thinking_html(reasoning, completed=answer_started),
                        unsafe_allow_html=True,
                    )
                if event.get("delta"):
                    if not answer_started:
                        time.sleep(0.25)
                        answer_started = True
                        reasoning = _strip_reasoning_overlap(
                            _sanitize_tags(thinking_buffer), _sanitize_tags(assembled)
                        )
                        thinking_body.markdown(
                            _thinking_html(reasoning, completed=True),
                            unsafe_allow_html=True,
                        )
                    assembled += event["delta"]
                    text_placeholder.markdown(_sanitize_tags(assembled))
                if st.session_state.get("stop_requested"):
                    break
                if event.get("final"):
                    st.session_state.conversation = event["history"]
                    metadata = event.get("metadata") or {}
                    st.session_state.diagnostics.append(metadata)
                    final_answer = _sanitize_tags(assembled) or _sanitize_tags(thinking_buffer)
                    # Freeze thinking card in completed state
                    reasoning = _strip_reasoning_overlap(
                        _sanitize_tags(thinking_buffer), final_answer
                    )
                    thinking_body.markdown(
                        _thinking_html(reasoning, completed=True),
                        unsafe_allow_html=True,
                    )
                    # Replace the streamed text with a single rendered message
                    text_placeholder.empty()
                    final_container = assistant_container.container()
                    _render_message_with_code(final_container, final_answer)
                    persist_session()
                    break
        except ConnectionError:
            error_message = (
                "⚠️ Unable to reach Ollama right now. Please check the host and API key, then try again."
            )
            text_placeholder.markdown(error_message)
            st.session_state.conversation = previous_history + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": error_message},
            ]
            st.session_state.diagnostics.append(
                {
                    "error": "connection",
                    "host": st.session_state.ollama_host,
                    "model": st.session_state.model_name,
                }
            )
            st.error(
                f"Could not reach the Ollama host ({st.session_state.ollama_host}). Double-check the URL and that your API key can access {st.session_state.model_name}.",
                icon="🚫",
            )
            persist_session()
            return
        except Exception as exc:  # noqa: BLE001 - surface error to user
            error_message = "⚠️ The agent ran into an error."
            text_placeholder.markdown(error_message)
            st.session_state.conversation = previous_history + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": error_message},
            ]
            st.session_state.diagnostics.append({"error": str(exc)})
            st.error(f"Agent failed: {exc}")
            persist_session()
            return


user_prompt = st.chat_input("Ask anything …")

if user_prompt:
    handle_user_message(user_prompt)

