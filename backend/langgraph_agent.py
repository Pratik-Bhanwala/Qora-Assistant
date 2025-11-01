"""General-purpose LangGraph/Ollama agent (no persona injection)."""

from __future__ import annotations

import json
from typing import Literal, TypedDict
from langgraph.graph import END, START, StateGraph
from ollama import Client

from .config import settings


class AgentState(TypedDict, total=False):
    """State that flows through the LangGraph pipeline for each turn."""

    history: list[dict[str, str]]
    user_input: str
    intent: Literal[
        "persona",
        "skills",
        "projects",
        "fun",
        "casual",
        "other",
    ]
    context_query: str
    context: str
    agent_reply: str


class PersonaAgent:
    """Agent wrapper that exposes a friendly interface for the Streamlit app."""

    def __init__(self) -> None:
        self._host = self._normalize_host(settings.ollama_host)
        self._api_key = settings.ollama_api_key
        self._client = self._create_client()
        self._graph = self._build_graph()
        self._model = settings.ollama_model

    @property
    def profile(self):  # retained for compatibility
        return None

    def profile_summary(self) -> str:
        return ""

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def generate_reply(
        self,
        history: list[dict[str, str]],
        user_input: str,
    ) -> dict[str, str]:
        """Run the LangGraph pipeline and return the assistant reply."""

        initial_state: AgentState = {
            "history": history,
            "user_input": user_input,
        }

        reasoning_steps: list[str] = []

        result_state = self._graph.invoke(initial_state)
        reply = result_state["agent_reply"]
        updated_history = result_state["history"]
        metadata = {"model": self._model, "host": self._host}

        return {"reply": reply, "history": updated_history, "metadata": metadata}

    # ------------------------------------------------------------------
    # Streaming API (token-by-token)
    # ------------------------------------------------------------------
    def stream_reply(
        self,
        history: list[dict[str, str]],
        user_input: str,
    ) -> "typing.Generator[dict[str, typing.Any], None, None]":
        """Yield reasoning first, then assistant tokens as they stream in, then final history.

        Events yielded:
        - { 'thinking': [lines], 'metadata': {...} }
        - { 'delta': token }
        - { 'final': True, 'history': [...], 'metadata': {...} }
        """

        import typing as _t

        # Build generic system message with reasoning tags
        system_prompt = (
            "You are Qora, a helpful general-purpose assistant. "
            "First write your reasoning inside <thinking>...</thinking> using clear steps, "
            "then write the final answer inside <answer>...</answer>."
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        metadata = {"model": self._model, "host": self._host}
        yield {"thinking": ["Thinking in progress"], "metadata": metadata}

        reply_text = ""
        try:
            stream = self._client.chat(
                model=self._model,
                messages=messages,
                options={"temperature": 0.6, "top_p": 0.9},
                stream=True,
            )
            in_thinking = False
            for chunk in stream:  # type: ignore[assignment]
                token = (chunk or {}).get("message", {}).get("content", "")
                if not token:
                    continue
                # Strip answer tags if model emits them
                token = token.replace("<answer>", "").replace("</answer>", "")
                if "<thinking>" in token:
                    in_thinking = True
                    token = token.replace("<thinking>", "")
                if "</thinking>" in token:
                    in_thinking = False
                    token = token.replace("</thinking>", "")

                if in_thinking:
                    if token:
                        yield {"thinking_delta": token}
                else:
                    if token:
                        reply_text += token
                        yield {"delta": token}
        except Exception:  # Propagate to caller to handle
            raise

        updated_history = history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": reply_text},
        ]

        yield {"final": True, "history": updated_history, "metadata": metadata}

    def set_model(self, model: str) -> None:
        """Allow callers to change the Ollama model dynamically."""

        if model:
            self._model = model

    def set_connection(self, host: str | None = None, api_key: str | None = None) -> dict[str, str | None]:
        """Reconfigure the Ollama client with new host and/or API key."""

        if host:
            self._host = self._normalize_host(host)
        if api_key is not None:
            self._api_key = api_key or None
        self._client = self._create_client()
        return self.connection_details()

    def connection_details(self) -> dict[str, str | None]:
        return {"host": self._host, "model": self._model}

    def _create_client(self) -> Client:
        client_kwargs: dict[str, object] = {}
        if self._api_key:
            client_kwargs["headers"] = {
                "Authorization": f"Bearer {self._api_key}",
            }
        return Client(host=self._host, **client_kwargs)

    def _normalize_host(self, host: str) -> str:
        cleaned = host.strip()
        if not cleaned:
            return self._host
        if cleaned.endswith("/api"):
            cleaned = cleaned[: -len("/api")]
        cleaned = cleaned.rstrip("/")
        return cleaned

    # ------------------------------------------------------------------
    # LangGraph definition
    # ------------------------------------------------------------------
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("respond", self._generate_response)
        workflow.add_edge(START, "respond")
        workflow.add_edge("respond", END)
        return workflow.compile()

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------
    def _classify_intent(self, state: AgentState) -> AgentState:
        return {"intent": "other", "context_query": state.get("user_input", "")}

    def _needs_context(self, state: AgentState) -> bool:
        return False

    def _fetch_context(self, state: AgentState) -> AgentState:
        return {"context": ""}

    def _generate_response(self, state: AgentState) -> AgentState:
        history = state.get("history", [])
        user_input = state["user_input"]
        context = ""
        system_prompt = (
            "You are Qora, a helpful, concise, and safe general-purpose AI assistant. "
            "Respond clearly and helpfully."
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        response = self._client.chat(
            model=self._model,
            messages=messages,
            options={"temperature": 0.6, "top_p": 0.9},
        )
        reply_content = response.get("message", {}).get("content", "I'm sorry, something went wrong.")

        updated_history = history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": reply_content},
        ]

        return {
            "agent_reply": reply_content,
            "history": updated_history,
        }


def format_history_for_display(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Ensure messages are serialisable and filtered for the Streamlit chat UI."""

    formatted: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        if role in {"user", "assistant"}:
            formatted.append({"role": role, "content": content})
    return formatted

