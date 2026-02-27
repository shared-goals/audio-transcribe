#!/usr/bin/env python3
"""FFmpeg audio preprocessing for WhisperX.

Converts any audio format to 16kHz mono PCM WAV.

Usage:
    uv run preprocess.py input.m4a -o output.wav
    uv run preprocess.py input.wav          # outputs input.16k.wav
    uv run preprocess.py input.mp4 --no-silence-removal
"""

import argparse
import subprocess
import sys
from pathlib import Path


def preprocess(
    input_path: str,
    output_path: str | None = None,
    remove_silence: bool = True,
    silence_threshold_db: str = "-35dB",
    silence_duration: float = 0.3,
) -> str:
    input_p = Path(input_path)
    if not input_p.exists():
        raise FileNotFoundError(f"Not found: {input_path}")

    if output_path is None:
        output_path = str(input_p.with_name(input_p.stem + ".16k.wav"))

    filters = []
    if remove_silence:
        filters.append(
            f"silenceremove=start_periods=1"
            f":start_silence={silence_duration}"
            f":start_threshold={silence_threshold_db}"
            f":detection=peak"
        )
    filters.append("aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono")

    cmd = ["ffmpeg", "-y", "-i", input_path, "-af", ",".join(filters), output_path]
    print(f"Preprocessing: {input_path} → {output_path}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("FFmpeg failed")

    size_mb = Path(output_path).stat().st_size / 1_048_576
    print(f"Done: {output_path} ({size_mb:.1f} MB)", file=sys.stderr)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess audio to 16kHz mono WAV")
    parser.add_argument("input", help="Input file (WAV, M4A, MP3, FLAC, MP4, WebM)")
    parser.add_argument("-o", "--output", help="Output path (default: <input>.16k.wav)")
    parser.add_argument("--no-silence-removal", action="store_true")
    parser.add_argument("--silence-threshold", default="-35dB")
    parser.add_argument("--silence-duration", type=float, default=0.3)
    args = parser.parse_args()

    preprocess(
        args.input,
        args.output,
        remove_silence=not args.no_silence_removal,
        silence_threshold_db=args.silence_threshold,
        silence_duration=args.silence_duration,
    )


if __name__ == "__main__":
    main()
