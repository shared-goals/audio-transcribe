# audio-transcribe

Local audio transcription pipeline: WhisperX + Ollama.

## Setup

```bash
uv sync
export HF_TOKEN=hf_...  # required for diarization
```

HuggingFace token requires accepting the pyannote license at:
https://huggingface.co/pyannote/speaker-diarization-3.1

External dependency: `ffmpeg` (`brew install ffmpeg`).

## Usage

Run via `uv` from the project directory:

```bash
uv run audio-transcribe process input.m4a -o result.json
```

Or add the venv to your PATH (e.g. in `~/.zshrc`):

```bash
export PATH="/Users/gnezim/_projects/gnezim/audio-transcribe/.venv/bin:$PATH"
```

Then you can run directly:

```bash
audio-transcribe process input.m4a -o result.json
```

## Options

```bash
# Specify language and model
uv run audio-transcribe process input.m4a -l ru -m large-v3 -o result.json

# Skip alignment or diarization
uv run audio-transcribe process input.m4a --no-align --no-diarize -o result.json

# View run statistics
uv run audio-transcribe stats --last 5

# Get backend recommendations
uv run audio-transcribe recommend input.m4a

# Learn corrections from an edited transcript
uv run audio-transcribe learn corrected-transcript.md
```

