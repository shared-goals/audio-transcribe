# Security Policy

## Supported versions

Only the latest tagged release is supported. Report vulnerabilities privately through GitHub Security Advisories rather than opening a public issue.

## Dependency policy

CI audits the installed core environment on every change. ML dependencies are updated as a coordinated stack and require an Apple Silicon smoke test because WhisperX, PyTorch, Torchaudio, Pyannote, Transformers, and MLX have tightly coupled compatibility constraints.

WhisperX 3.8.6 requires PyTorch and Torchaudio 2.8.x and Hugging Face Hub below 1.0. Known advisories that require a newer PyTorch or Transformers 5.3+ cannot currently be resolved: secure Transformers releases require Hugging Face Hub 1.x. This application processes local, trusted recordings and trusted model artifacts only; do not load untrusted PyTorch checkpoints, serialized models, or audio supplied by an untrusted party. Re-evaluate these exceptions whenever WhisperX releases a compatible stack.

TorchCodec is pinned to the upstream-supported `0.7.x` line for PyTorch 2.8 so fresh installations satisfy WhisperX metadata. TorchCodec 0.7 cannot load against the newer Homebrew FFmpeg libraries on the deployment Mac; Pyannote catches this condition, and the application decodes audio through FFmpeg or supplies preloaded tensors instead. Every ML-stack update must verify that Pyannote initialization still degrades to this supported fallback.
