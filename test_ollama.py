#!/usr/bin/env python3
"""Test Ollama for Russian meeting summarization.

Usage:
    uv run test_ollama.py                    # test default model
    uv run test_ollama.py --list-models
    uv run test_ollama.py -m qwen2.5:14b

Prerequisites:
    brew install ollama
    ollama serve                  # or: brew services start ollama
    ollama pull gemma3:27b        # ~16 GB, excellent Russian
    ollama pull qwen2.5:14b       # ~8 GB, very good Russian
"""

import argparse
import json
import sys
import time
import urllib.request
from urllib.error import URLError

OLLAMA_URL = "http://localhost:11434"

SAMPLE_TRANSCRIPT = """\
SPEAKER_00: Добрый день. Начнём с задач на следующий спринт.
SPEAKER_01: Есть несколько срочных вопросов по бэкенду.
SPEAKER_00: Иван, что у тебя с API авторизации?
SPEAKER_01: Закончу до пятницы — осталось написать тесты.
SPEAKER_02: Мне нужна помощь с интеграцией. Можете посмотреть мой PR?
SPEAKER_00: Посмотрю сегодня. Итого: Иван — тесты до пятницы, Мария — PR на ревью сегодня.
"""

PROMPT = """\
Прочитай транскрипт встречи и создай резюме.

Транскрипт:
{transcript}
Ответь строго в JSON:
{{
  "summary": "краткое описание (2-3 предложения)",
  "decisions": ["решение 1"],
  "action_items": [
    {{"what": "задача", "who": "кто", "when": "когда"}}
  ]
}}"""


def ollama_is_running() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            return bool(r.status == 200)
    except (URLError, Exception):
        return False


def list_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            data = json.loads(r.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def extract_json(response: str) -> str:
    """Extract JSON from a response that may be wrapped in markdown code fences."""
    json_str = response
    for marker in ("```json", "```"):
        if marker in json_str:
            json_str = json_str.split(marker)[1].split("```")[0].strip()
            break
    return json_str


def generate(model: str, prompt: str) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return str(json.loads(r.read()).get("response", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Ollama for Russian summarization")
    parser.add_argument("-m", "--model", default="gemma3:27b")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    print("Checking Ollama... ", end="", flush=True)
    if not ollama_is_running():
        print("NOT RUNNING")
        print("\nStart with:   ollama serve")
        print("Install with: brew install ollama")
        sys.exit(1)
    print("OK")

    models = list_models()

    if args.list_models:
        print(f"\nAvailable models: {models or '(none pulled)'}")
        print("\nRecommended:")
        print("  ollama pull gemma3:27b    # ~16 GB, excellent Russian")
        print("  ollama pull qwen2.5:14b   # ~8 GB, very good Russian")
        return

    if not models:
        print(f"\nNo models found. Pull one with:  ollama pull {args.model}")
        sys.exit(1)

    if args.model not in models:
        print(f"\nModel '{args.model}' not available. Pulled: {models}")
        print(f"Pull with:  ollama pull {args.model}")
        sys.exit(1)

    print(f"Testing: {args.model}")
    t0 = time.time()
    response = generate(args.model, PROMPT.format(transcript=SAMPLE_TRANSCRIPT))
    elapsed = time.time() - t0
    print(f"\nResponse ({elapsed:.1f}s):\n{response}")

    # Try to parse JSON from response (may be wrapped in markdown code fences)
    json_str = extract_json(response)

    try:
        parsed = json.loads(json_str)
        print("\nParsed OK:")
        print(f"  summary:      {parsed.get('summary', '')[:80]}")
        print(f"  decisions:    {len(parsed.get('decisions', []))}")
        print(f"  action_items: {len(parsed.get('action_items', []))}")
        print("\nOllama test PASSED")
    except json.JSONDecodeError:
        print("\nJSON parse failed — check response above")


if __name__ == "__main__":
    main()
