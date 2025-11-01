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
            .stApp { background: #f7f8fa; color: #1f2937; }
            .stApp:before { content:""; position: fixed; inset: 0; pointer-events: none;
                background: radial-gradient(1200px 600px at 90% 5%, rgba(59,130,246,0.08), transparent 60%),
                            radial-gradient(800px 400px at 10% 100%, rgba(99,102,241,0.08), transparent 60%); }
            section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
            section[data-testid="stSidebar"] .stButton>button { border-radius: 12px; border: 1px solid #e2e8f0; background: #ffffff; color: #1f2937; }
            section[data-testid="stSidebar"] .stButton>button:hover { border-color: #cbd5f5; }
            .sidebar-header { font-weight: 600; margin-top: 0.5rem; margin-bottom: 0.75rem; }
            div[data-testid="stChatMessage"] { max-width: 820px; margin-bottom: 20px; border-radius: 18px; border: 1px solid #e2e8f0; padding: 18px 22px; box-shadow: 0 12px 20px rgba(15,23,42,.05); background: #fff; }
            div[data-testid="stChatMessageUser"] { background: #2563eb; color:#fff; margin-left:auto; }
            div[data-testid="stChatMessageUser"] p { color:#fff !important; }
            /* Make thinking lighter so it is visually distinct */
            .thinking-card { background:#f8f9fb !important; border:1px dashed #d1d9e6 !important; border-radius:12px !important; padding:12px !important; color:#8595a8 !important; font-size:0.9rem !important; font-style: italic !important; font-weight: 400 !important; line-height: 1.5 !important; }
            .thinking-card * { color:#8595a8 !important; font-style: italic !important; font-weight: 400 !important; }
            .thinking-card .thinking-label { display:block; font-style: normal !important; text-transform: uppercase; letter-spacing: .08em; font-size:0.72rem !important; font-weight: 600 !important; margin-bottom:6px; color:#64748b !important; }
            .thinking-card .thinking-body { font-style: italic !important; color:#8595a8 !important; }
            .chat-header { font-size: 1.8rem; font-weight: 600; margin-bottom: 0.1rem; }
            .chat-subheader { color: #64748b; margin-bottom: 1.2rem; }
            /* Reduce chat input height and keep it very close to bottom */
            div[data-testid="stChatInput"] { position: sticky; bottom: 6px; z-index: 2; padding-top: 0; padding-bottom: 0; }
            .stChatInput textarea { border-radius:999px; border:1px solid #d1d5db; padding:10px 14px; font-size:.98rem; background:#ffffff; box-shadow: 0 12px 20px rgba(15,23,42,.06); }
            [data-testid="stChatInputSubmitButton"] { border-radius:999px; background:#2563eb; color:#fff; border:none; width:40px; height:40px; display:inline-flex; align-items:center; justify-content:center; }
            [data-testid="stChatInputSubmitButton"]:hover { background:#1d4ed8; }
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
    st.markdown("<div class='sidebar-header'>Chats</div>", unsafe_allow_html=True)
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

    delete_choice = st.selectbox(
        "Delete chat",
        options=["None"] + [get_chat(chat_id).get("title", "New chat") for chat_id in chat_options],
        index=0,
    )
    if delete_choice != "None":
        titles = [get_chat(chat_id).get("title", "New chat") for chat_id in chat_options]
        chat_id_to_delete = chat_options[titles.index(delete_choice)]
        st.session_state.chats = [chat for chat in st.session_state.chats if chat["id"] != chat_id_to_delete]
        if chat_id_to_delete == st.session_state.chat_id:
            if st.session_state.chats:
                st.session_state.chat_id = st.session_state.chats[0]["id"]
            else:
                new_chat()
        save_memory(st.session_state.chats)
        select_chat(st.session_state.chat_id)
        st.rerun()

    with st.expander("Settings", expanded=True):
        host_input = st.text_input(
            "Ollama host",
            value=st.session_state.ollama_host,
            help="Use http://localhost:11434 for local runtime or https://ollama.com for cloud.",
        )
        if host_input != st.session_state.ollama_host:
            details = agent.set_connection(host_input, st.session_state.ollama_api_key)
            st.session_state.ollama_host = details["host"] or host_input
            st.toast(f"Host set to {st.session_state.ollama_host}")

        api_key_input = st.text_input(
            "Ollama API key",
            value=st.session_state.ollama_api_key,
            type="password",
            help="Only required for Ollama Cloud.",
        )
        if api_key_input != st.session_state.ollama_api_key:
            st.session_state.ollama_api_key = api_key_input
            details = agent.set_connection(st.session_state.ollama_host, api_key_input)
            st.session_state.ollama_host = details["host"] or st.session_state.ollama_host
            st.toast("Updated API key")

        selected_model = st.text_input(
            "Ollama model",
            value=st.session_state.model_name,
            help="Pick a model reachable by the configured host (e.g. glm-4.6:cloud).",
        )
        if selected_model and selected_model != st.session_state.model_name:
            st.session_state.model_name = selected_model
            agent.set_model(selected_model)
            st.toast(f"Switched model to {selected_model}")

        if st.button("Clear current chat", use_container_width=True):
            st.session_state.conversation = []
            st.session_state.diagnostics = []
            persist_session()
            st.rerun()

        if st.button("■ Stop generation", help="Stop the current response"):
            st.session_state.stop_requested = True

    st.markdown("<div class='controls-section'></div>", unsafe_allow_html=True)

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

