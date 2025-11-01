"""Utility helpers for persisting chat history locally."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from .config import BASE_DIR


MEMORY_FILE = BASE_DIR / "data" / "chat_memory.json"


def load_memory() -> List[Dict[str, Any]]:
    """Load persisted conversations from disk."""

    if not MEMORY_FILE.exists():
        return []

    try:
        with MEMORY_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, list):
                return data
    except json.JSONDecodeError:
        # Corrupted file – reset to empty conversations.
        return []

    return []


def save_memory(chats: List[Dict[str, Any]]) -> None:
    """Persist the provided chat list to disk."""

    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MEMORY_FILE.open("w", encoding="utf-8") as handle:
        json.dump(chats, handle, ensure_ascii=False, indent=2)

