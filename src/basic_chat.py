#!/usr/bin/env python3

import argparse
import itertools
import sys
import threading
import time
from dataclasses import dataclass
from typing import Self

import ollama

SIZES = {
    "small": ["phi4-mini:3.8b", "gemma3:4b"],
    "medium": ["gemma4:12b", "gpt-oss:20b", "phi4:14b"],
}
DEFAULT_SIZE = "medium"
DEFAULT_PROMPT = "Why is the sky blue?"

NS_PER_S = 1_000_000_000
NUM_PREDICT = 128
UNLOAD_TIMEOUT_S = 30

# ollama wraps an unreachable server in the builtin ConnectionError.
OLLAMA_ERRORS = (
    ollama.ResponseError,
    ollama.RequestError,
    ConnectionError,
    TimeoutError,
)


@dataclass
class Timing:
    model: str
    wall: float
    load: float
    eval_tokens: int
    eval_rate: float
    truncated: bool


def ns_to_s(value: int | None) -> float:
    return (value or 0) / NS_PER_S


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
        if self._thread:
            self._stop.set()
            self._thread.join()
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


def unload_all(client: ollama.Client) -> None:
    """Evict every loaded model so the next one has the GPU to itself."""
    for loaded in client.ps().models:
        client.generate(model=loaded.model, keep_alive=0)

    # Eviction finishes asynchronously; wait until the VRAM is actually free.
    deadline = time.monotonic() + UNLOAD_TIMEOUT_S
    while client.ps().models:
        if time.monotonic() > deadline:
            raise TimeoutError("models still loaded after unload request")
        time.sleep(0.1)


def warmup(client: ollama.Client, model: str) -> None:
    """Load the model into memory so the timed call is not charged for it."""
    client.chat(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        options={"num_predict": 1},
    )


def run(client: ollama.Client, model: str, prompt: str) -> Timing | None:
    print(f"Model: {model}")

    try:
        with Spinner("loading model"):
            unload_all(client)
            warmup(client, model)

        # Time inside the spinner so its thread start/join isn't counted.
        with Spinner("generating"):
            start = time.perf_counter()
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": NUM_PREDICT},
            )
            wall = time.perf_counter() - start
    except OLLAMA_ERRORS as err:
        print(f"  failed: {err}\n")
        return None

    # Reasoning models can spend the whole token budget in `thinking`,
    # leaving `content` empty; show whichever the model produced.
    print(response.message.content or response.message.thinking or "")

    eval_tokens = response.eval_count or 0
    eval_seconds = ns_to_s(response.eval_duration)
    timing = Timing(
        model=model,
        wall=wall,
        load=ns_to_s(response.load_duration),
        eval_tokens=eval_tokens,
        eval_rate=eval_tokens / eval_seconds if eval_seconds else 0.0,
        truncated=response.done_reason == "length",
    )

    print(
        f"\n  {timing.wall:.2f}s wall "
        f"({timing.load:.2f}s load, "
        f"{timing.eval_tokens} tokens"
        f"{' capped' if timing.truncated else ''}, "
        f"{timing.eval_rate:.1f} tok/s)\n"
    )
    return timing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local Ollama models on the same prompt."
    )
    parser.add_argument(
        "--size",
        choices=sorted(SIZES),
        default=DEFAULT_SIZE,
        help=f"model set to benchmark (default: {DEFAULT_SIZE})",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help=f"prompt to send (default: {DEFAULT_PROMPT!r})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = " ".join(args.prompt) or DEFAULT_PROMPT

    client = ollama.Client()
    timings = [t for model in SIZES[args.size] if (t := run(client, model, prompt))]

    try:
        unload_all(client)
    except OLLAMA_ERRORS as err:
        print(f"unload failed: {err}\n")

    if not timings:
        return

    print(f"{'model':<16}{'wall':>9}{'load':>9}{'tokens':>9}{'tok/s':>9}")
    for t in sorted(timings, key=lambda t: t.eval_rate, reverse=True):
        print(
            f"{t.model:<16}{t.wall:>8.2f}s{t.load:>8.2f}s"
            f"{t.eval_tokens:>9}{t.eval_rate:>9.1f}"
        )


if __name__ == "__main__":
    main()
