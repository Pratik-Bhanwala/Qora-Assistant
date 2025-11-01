"""Quick sanity check for Ollama Cloud connectivity and model availability."""

from __future__ import annotations

import os
import sys

from ollama import Client


def main() -> int:
    host = os.environ.get("CHATBOT_OLLAMA_HOST", "https://cloud.ollama.com")
    model = os.environ.get("CHATBOT_OLLAMA_MODEL", "glm-4.6:cloud")
    api_key = os.environ.get("CHATBOT_OLLAMA_API_KEY")

    if not api_key:
        print("ERROR: CHATBOT_OLLAMA_API_KEY is not set.", file=sys.stderr)
        return 2

    client = Client(host=host, headers={"Authorization": f"Bearer {api_key}"})

    try:
        response = client.generate(model=model, prompt="Say hello in one short sentence.")
    except Exception as exc:  # noqa: BLE001
        print("FAILED to reach Ollama Cloud")
        print(f"Host: {host}")
        print(f"Model: {model}")
        print(f"Error: {exc}")
        return 1

    print("SUCCESS")
    print(f"Model: {response.get('model')}")
    print(f"Output: {response.get('response')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

