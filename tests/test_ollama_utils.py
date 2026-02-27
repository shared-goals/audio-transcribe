"""Tests for test_ollama.py utilities — extract_json, ollama_is_running, list_models."""

import json
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from test_ollama import extract_json, list_models, ollama_is_running

# --- extract_json ---


def test_extract_json_plain_json():
    assert extract_json('{"key": "val"}') == '{"key": "val"}'


def test_extract_json_markdown_json_fence():
    response = '```json\n{"key": "val"}\n```'
    assert extract_json(response) == '{"key": "val"}'


def test_extract_json_plain_code_fence():
    response = '```\n{"key": "val"}\n```'
    assert extract_json(response) == '{"key": "val"}'


def test_extract_json_prefers_json_fence():
    """json fence takes priority over plain fence."""
    response = '```json\n{"a": 1}\n```'
    result = extract_json(response)
    assert result == '{"a": 1}'


def test_extract_json_valid_after_extraction():
    response = '```json\n{"summary": "ok", "decisions": []}\n```'
    parsed = json.loads(extract_json(response))
    assert parsed["summary"] == "ok"


# --- ollama_is_running ---


def test_ollama_not_running_connection_refused():
    with patch("urllib.request.urlopen", side_effect=URLError("connection refused")):
        assert ollama_is_running() is False


def test_ollama_not_running_generic_exception():
    with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
        assert ollama_is_running() is False


def test_ollama_running():
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_resp):
        assert ollama_is_running() is True


# --- list_models ---


def test_list_models_returns_names():
    data = {"models": [{"name": "gemma3:27b"}, {"name": "qwen2.5:14b"}]}
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_resp):
        models = list_models()
    assert models == ["gemma3:27b", "qwen2.5:14b"]


def test_list_models_empty_on_error():
    with patch("urllib.request.urlopen", side_effect=URLError("refused")):
        assert list_models() == []


def test_list_models_empty_list():
    data: dict[str, list[object]] = {"models": []}
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_resp):
        assert list_models() == []
