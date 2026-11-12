"""Resident ASR worker for the short-request service (issue #4).

One process holds both models warm and answers JSON-lines requests over a Unix
socket; the Rust HTTP front in ``service/`` owns queueing, timeouts and the
worker lifecycle.

Two passes per request:
  1. streaming T-one Russian (``sherpa-onnx``, CPU) on 8 kHz audio - fast, lower
     case, no punctuation;
  2. ``mlx-whisper large-v3`` on 16 kHz audio with the T-one text as
     ``initial_prompt`` - it adds punctuation, case and fixes word forms, but
     does not have to re-derive the words.

The second pass is the answer: the client gets one text, so returning the raw
T-one pass would be a readability regression.

Run as ``python -m audio_transcribe.asr_worker --socket <path> --tone-model-dir <dir>``.
"""

from __future__ import annotations

import argparse
import json
import os
import socketserver
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np

from audio_transcribe.stages.preprocess import preprocess
from audio_transcribe.stages.transcribe import MlxVadOptions, transcribe_mlx_vad

DEFAULT_WHISPER_MODEL = "large-v3"
TONE_SAMPLE_RATE = 8000
TONE_CHUNK_SECONDS = 0.1
TONE_TAIL_SECONDS = 0.6


def merge_prompt(tone_text: str, prompt: str) -> str:
    """Build the ``initial_prompt`` for the final pass.

    The T-one text is the strongest available context (it is what was actually
    said); a caller-supplied vocabulary/context string is appended after it so
    names and abbreviations still reach the decoder.
    """
    parts = [part.strip() for part in (tone_text, prompt) if part and part.strip()]
    return " ".join(parts)


def _prepare_audio(path: str) -> str:
    """Convert any recorded container to 16 kHz mono WAV next to the source."""
    output = str(Path(path).with_suffix("")) + ".16k.wav"
    return preprocess(path, output, remove_silence=False)


def _decode_8k(source: str) -> str:
    """Resample to the 8 kHz mono WAV that T-one expects."""
    import subprocess
    import tempfile

    handle, path = tempfile.mkstemp(suffix=".wav")
    os.close(handle)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", source, "-ar", str(TONE_SAMPLE_RATE), "-ac", "1", path],
        check=True,
    )
    return path


class ToneEngine:
    """Streaming T-one Russian recognizer, loaded on first use."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self._recognizer: Any = None

    def _load(self) -> Any:
        if self._recognizer is None:
            import sherpa_onnx

            self._recognizer = sherpa_onnx.OnlineRecognizer.from_t_one_ctc(
                tokens=str(self.model_dir / "tokens.txt"),
                model=str(self.model_dir / "model.onnx"),
                num_threads=2,
                sample_rate=TONE_SAMPLE_RATE,
                feature_dim=80,
                decoding_method="greedy_search",
            )
        return self._recognizer

    def __call__(self, audio_16k: str) -> str:
        recognizer = self._load()
        wav_path = _decode_8k(audio_16k)
        try:
            with wave.open(wav_path, "rb") as wav:
                raw = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
            samples = raw.astype(np.float32) / 32768.0
            stream = recognizer.create_stream()
            chunk = int(TONE_SAMPLE_RATE * TONE_CHUNK_SECONDS)
            for start in range(0, len(samples), chunk):
                stream.accept_waveform(TONE_SAMPLE_RATE, samples[start : start + chunk])
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
            stream.accept_waveform(
                TONE_SAMPLE_RATE,
                np.zeros(int(TONE_SAMPLE_RATE * TONE_TAIL_SECONDS), dtype=np.float32),
            )
            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            return str(recognizer.get_result(stream)).strip()
        finally:
            os.unlink(wav_path)


class WhisperEngine:
    """Production VAD + ``mlx-whisper`` final pass, model kept warm."""

    def __init__(self, model: str = DEFAULT_WHISPER_MODEL) -> None:
        self.model = model

    def __call__(self, audio_16k: str, language: str | None, prompt: str) -> str:
        result, _audio = transcribe_mlx_vad(
            audio_16k,
            self.model,
            language or "",
            MlxVadOptions(initial_prompt=prompt or None),
            keep_model=True,
        )
        segments = result.get("segments") or []
        return " ".join(str(seg.get("text", "")).strip() for seg in segments).strip()


class Worker:
    """Executes one transcription request; engines are injectable for tests.

    The final pass is the answer. T-one runs before it only when
    ``tone_prompt`` is enabled, because an unpunctuated, lower-case prompt makes
    whisper mirror that style and drop punctuation - measured on 12.11.2026:
    the same clip came back as ``раз два три четыре пять`` with the T-one text
    as prompt and as ``Раз, два, три, четыре, пять.`` without it. By default the
    tone engine is the degraded answer when the final pass fails.
    """

    def __init__(
        self,
        tone: Callable[[str], str] | None,
        whisper: Callable[[str, str | None, str], str],
        prepare: Callable[[str], str] = _prepare_audio,
        tone_prompt: bool = False,
    ) -> None:
        self.tone = tone
        self.whisper = whisper
        self.prepare = prepare
        self.tone_prompt = tone_prompt
        self.lock = threading.Lock()

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("id")
        audio = request.get("audio")
        if not isinstance(audio, str) or not Path(audio).is_file():
            return {"id": request_id, "ok": False, "error": f"audio file not found: {audio!r}"}
        language = request.get("language") or None
        prompt = str(request.get("prompt") or "")
        started = time.monotonic()
        # shortcut: requests are serialized because both engines are stateful and
        # the model host has 16 GB; raise this only with a measured memory budget.
        with self.lock:
            prepared = self.prepare(audio)
            tone_text = ""
            tone_ms = 0
            if self.tone is not None and self.tone_prompt:
                tone_started = time.monotonic()
                tone_text = self.tone(prepared)
                tone_ms = round((time.monotonic() - tone_started) * 1000)
            whisper_started = time.monotonic()
            try:
                text = self.whisper(prepared, language, merge_prompt(tone_text, prompt))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if self.tone is None or tone_text:
                    return {
                        "id": request_id,
                        "ok": False,
                        "error": f"final pass failed: {error}",
                        "tone_text": tone_text,
                        "timings_ms": {"tone": tone_ms, "total": round((time.monotonic() - started) * 1000)},
                    }
                # Degraded answer: the unpunctuated T-one text beats no text at all.
                tone_started = time.monotonic()
                try:
                    tone_text = self.tone(prepared)
                except Exception as tone_exc:
                    return {
                        "id": request_id,
                        "ok": False,
                        "error": f"final pass failed: {error}; tone fallback failed: {tone_exc}",
                    }
                tone_ms = round((time.monotonic() - tone_started) * 1000)
                return {
                    "id": request_id,
                    "ok": True,
                    "text": tone_text,
                    "tone_text": tone_text,
                    "degraded": True,
                    "tone_error": f"final pass failed: {error}",
                    "timings_ms": {"tone": tone_ms, "total": round((time.monotonic() - started) * 1000)},
                }
            whisper_ms = round((time.monotonic() - whisper_started) * 1000)
        response: dict[str, Any] = {
            "id": request_id,
            "ok": True,
            "text": text,
            "timings_ms": {
                "tone": tone_ms,
                "whisper": whisper_ms,
                "total": round((time.monotonic() - started) * 1000),
            },
        }
        if tone_text:
            response["tone_text"] = tone_text
        return response


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        worker: Worker = self.server.worker  # type: ignore[attr-defined]
        for line in self.rfile:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                response = {"id": None, "ok": False, "error": f"invalid json: {exc}"}
            else:
                # One broken request must not kill the connection for the next one.
                try:
                    response = worker.handle(request)
                except Exception as exc:
                    response = {"id": request.get("id"), "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            self.wfile.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, socket_path: str, worker: Worker) -> None:
        self.worker = worker
        super().__init__(socket_path, _Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resident ASR worker for the audio-transcribe service.")
    parser.add_argument("--socket", required=True, help="Unix socket path to listen on")
    parser.add_argument(
        "--tone-model-dir",
        help="sherpa-onnx T-one Russian model directory; without it only the final pass runs",
    )
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument(
        "--tone-prompt",
        action="store_true",
        help="use the T-one text as initial_prompt for the final pass (drops punctuation; see Worker)",
    )
    args = parser.parse_args(argv)

    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()

    tone = ToneEngine(args.tone_model_dir) if args.tone_model_dir else None
    worker = Worker(tone, WhisperEngine(args.whisper_model), tone_prompt=args.tone_prompt)
    server = _Server(str(socket_path), worker)
    print(f"asr worker listening on {socket_path}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if socket_path.exists():
            socket_path.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
