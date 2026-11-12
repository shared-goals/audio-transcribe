import json
import os
import shutil
import socket
import tempfile
import threading

from audio_transcribe import asr_worker


def _call(worker, request):
    """Serve one request through the real socket loop, return the parsed response."""
    # AF_UNIX paths are capped at ~104 bytes, so not pytest's tmp_path.
    short_dir = tempfile.mkdtemp(prefix="asr-t-")
    socket_path = os.path.join(short_dir, "s.sock")
    server = asr_worker._Server(socket_path, worker)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(socket_path)
    client.sendall((json.dumps(request) + "\n").encode())
    chunks = b""
    while not chunks.endswith(b"\n"):
        data = client.recv(65536)
        if not data:
            break
        chunks += data
    client.close()
    thread.join(timeout=5)
    server.server_close()
    shutil.rmtree(short_dir, ignore_errors=True)
    return json.loads(chunks.decode())


def _worker(tmp_path, whisper, tone=None, tone_prompt=False, failures=()):
    audio = tmp_path / "voice.webm"
    audio.write_bytes(b"audio")
    calls = {"tone": 0}

    def tone_engine(path):
        calls["tone"] += 1
        if "tone" in failures:
            raise RuntimeError("tone model is not installed")
        return tone or ""

    engine = asr_worker.Worker(
        tone_engine if tone is not None or failures else None,
        whisper,
        prepare=lambda path: path,
        tone_prompt=tone_prompt,
    )
    return engine, audio, calls


def test_final_pass_is_the_answer_and_tone_is_not_run_by_default(tmp_path):
    seen = {}

    def whisper(path, language, prompt):
        seen["prompt"] = prompt
        seen["language"] = language
        return "Раз, два, три."

    worker, audio, calls = _worker(tmp_path, whisper, tone="раз два три")
    response = _call(worker, {"id": 7, "audio": str(audio), "language": "ru", "prompt": "Thunder Forge"})

    assert response["ok"] is True
    assert response["text"] == "Раз, два, три."
    # An unpunctuated prompt makes whisper drop punctuation, so the tone text is
    # not fed to the final pass unless it is explicitly requested.
    assert seen["prompt"] == "Thunder Forge"
    assert calls["tone"] == 0
    assert "tone_text" not in response


def test_tone_prompt_is_opt_in(tmp_path):
    seen = {}

    def whisper(path, language, prompt):
        seen["prompt"] = prompt
        return "текст"

    worker, audio, calls = _worker(tmp_path, whisper, tone="текст", tone_prompt=True)
    response = _call(worker, {"audio": str(audio), "prompt": "MLX"})

    assert seen["prompt"] == "текст MLX"
    assert calls["tone"] == 1
    assert response["tone_text"] == "текст"


def test_failed_final_pass_degrades_to_tone_text(tmp_path):
    def whisper(path, language, prompt):
        raise RuntimeError("mlx is unavailable")

    worker, audio, _calls = _worker(tmp_path, whisper, tone="раз два три")
    response = _call(worker, {"audio": str(audio)})

    assert response["ok"] is True
    assert response["degraded"] is True
    assert response["text"] == "раз два три"
    assert "mlx is unavailable" in response["tone_error"]


def test_total_failure_is_reported_when_tone_also_fails(tmp_path):
    def whisper(path, language, prompt):
        raise RuntimeError("mlx is unavailable")

    worker, audio, _calls = _worker(tmp_path, whisper, failures=("tone",))
    response = _call(worker, {"audio": str(audio)})

    assert response["ok"] is False
    assert "mlx is unavailable" in response["error"]
    assert "tone model is not installed" in response["error"]


def test_missing_audio_is_a_clean_error(tmp_path):
    worker, _audio, _calls = _worker(tmp_path, lambda p, lang, prompt: "текст")
    response = _call(worker, {"id": 3, "audio": "/nope/missing.wav"})

    assert response["ok"] is False
    assert response["id"] == 3
    assert "audio file not found" in response["error"]


def test_merge_prompt_keeps_only_present_parts():
    assert asr_worker.merge_prompt("", "словарь") == "словарь"
    assert asr_worker.merge_prompt("текст", "  ") == "текст"
    assert asr_worker.merge_prompt("", "") == ""


def test_a_broken_request_does_not_kill_the_connection(tmp_path):
    def whisper(path, language, prompt):
        raise RuntimeError("boom")

    def prepare(path):
        raise FileNotFoundError("ffmpeg")

    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"audio")
    worker = asr_worker.Worker(None, whisper, prepare=prepare)
    response = _call(worker, {"id": 11, "audio": str(audio)})

    assert response["ok"] is False
    assert response["id"] == 11
