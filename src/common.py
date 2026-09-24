"""Shared helpers for the local-model tools."""

import itertools
import sys
import threading
import time
from typing import Self

import ollama

# ollama wraps an unreachable server in the builtin ConnectionError.
OLLAMA_ERRORS = (
    ollama.ResponseError,
    ollama.RequestError,
    ConnectionError,
)


class Spinner:
    """Animate a status line while a blocking call runs, so it doesn't look hung."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, message: str) -> None:
        self.message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        # Only animate on a terminal; carriage returns would litter piped output.
        if sys.stdout.isatty():
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def stop(self) -> None:
        """Clear the status line early, e.g. once the first token arrives."""
        if self._thread:
            self._stop.set()
            self._thread.join()
            self._thread = None
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    def _spin(self) -> None:
        start = time.monotonic()
        for frame in itertools.cycle(self.FRAMES):
            elapsed = time.monotonic() - start
            sys.stdout.write(f"\r  {frame} {self.message} ({elapsed:.0f}s)")
            sys.stdout.flush()
            if self._stop.wait(0.1):
                return


def read_input(path: str | None, tool: str) -> str:
    """Read from a file, or from a pipe when no file is given."""
    if path:
        try:
            return open(path, errors="replace").read()
        except OSError as err:
            sys.exit(f"{tool}: {err}")
    if sys.stdin.isatty():
        sys.exit(f"{tool}: nothing on stdin; pipe text in or pass a file path")
    return sys.stdin.buffer.read().decode(errors="replace")


def pick_model(
    client: ollama.Client, requested: str | None, preference: list[str], tool: str
) -> str:
    """The requested model, else the first installed one in preference order."""
    installed = [m.model for m in client.list().models]
    if requested:
        if requested not in installed:
            sys.exit(
                f"{tool}: {requested} is not installed (have: {', '.join(installed)})"
            )
        return requested
    for model in preference:
        if model in installed:
            return model
    if not installed:
        sys.exit(f"{tool}: no models installed; pull one with `ollama pull`")
    return installed[0]
