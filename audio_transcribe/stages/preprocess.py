"""FFmpeg audio preprocessing for WhisperX.

Converts any audio format to 16kHz mono PCM WAV.
"""

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def preprocess(
    input_path: str,
    output_path: str | None = None,
    remove_silence: bool = True,
    silence_threshold_db: str = "-35dB",
    silence_duration: float = 0.3,
) -> str:
    """Preprocess audio to 16kHz mono WAV with optional silence removal."""
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
    logger.info("Preprocessing: %s -> %s", input_path, output_path)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        logger.error("FFmpeg error: %s", proc.stderr)
        raise RuntimeError("FFmpeg failed")

    out_size = Path(output_path).stat().st_size
    # silenceremove can strip everything if the file is all silence/noise -
    # a 16kHz mono WAV needs at least ~44 bytes header + some samples.
    # 1 KB (~63 samples) is the safety floor.
    if out_size < 1024:
        logger.error(
            "Preprocessed audio is empty (%d bytes) - "
            "silenceremove removed all content. "
            "Source file may be silence-only, corrupted, or contain no speech-like audio.",
            out_size,
        )
        raise RuntimeError(
            f"Preprocessed audio is empty ({out_size} bytes) - "
            "silenceremove removed all content. "
            "Source file may be silence-only, corrupted, or contain no speech-like audio. "
            "Try with --no-silence-remove or raise silence_duration threshold."
        )

    size_mb = out_size / 1_048_576
    logger.info("Done: %s (%.1f MB)", output_path, size_mb)
    return output_path
