"""Shared application exceptions."""


class PipelineError(Exception):
    """Pipeline failure with optional stage and timing context."""

    def __init__(self, message: str, stage: str | None = None, elapsed_s: float = 0.0) -> None:
        self.stage = stage
        self.elapsed_s = elapsed_s
        super().__init__(message)
