#!/usr/bin/env python3

import argparse
import time
from dataclasses import dataclass

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


@dataclass
class Timing:
    model: str
    wall: float
    ttft: float
    load: float
    eval_tokens: int
    eval_rate: float
    truncated: bool


def ns_to_s(value: int | None) -> float:
    return (value or 0) / NS_PER_S


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
        unload_all(client)
        warmup(client, model)

        start = time.perf_counter()
        ttft = None
        final = None
        for chunk in client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": NUM_PREDICT},
            stream=True,
        ):
            # Reasoning models stream `thinking` before (or instead of)
            # `content`; either counts as the first token.
            text = chunk.message.content or chunk.message.thinking or ""
            if text:
                if ttft is None:
                    ttft = time.perf_counter() - start
                print(text, end="", flush=True)
            if chunk.done:
                final = chunk
        wall = time.perf_counter() - start
    # ollama wraps an unreachable server in the builtin ConnectionError.
    except (
        ollama.ResponseError,
        ollama.RequestError,
        ConnectionError,
        TimeoutError,
    ) as err:
        print(f"\n  failed: {err}\n")
        return None

    if final is None:
        print("\n  failed: stream ended without a final chunk\n")
        return None

    eval_tokens = final.eval_count or 0
    eval_seconds = ns_to_s(final.eval_duration)
    timing = Timing(
        model=model,
        wall=wall,
        ttft=ttft if ttft is not None else wall,
        load=ns_to_s(final.load_duration),
        eval_tokens=eval_tokens,
        eval_rate=eval_tokens / eval_seconds if eval_seconds else 0.0,
        truncated=final.done_reason == "length",
    )

    print(
        f"\n\n  {timing.wall:.2f}s wall "
        f"({timing.ttft:.2f}s to first token, "
        f"{timing.load:.2f}s load, "
        f"{timing.eval_tokens} tokens"
        f"{' capped' if timing.truncated else ''}, "
        f"{timing.eval_rate:.1f} tok/s)\n"
    )
    return timing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local Ollama models on the same prompt, streaming."
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

    if not timings:
        return

    print(f"{'model':<16}{'wall':>9}{'ttft':>9}{'load':>9}{'tokens':>9}{'tok/s':>9}")
    for t in sorted(timings, key=lambda t: t.eval_rate, reverse=True):
        print(
            f"{t.model:<16}{t.wall:>8.2f}s{t.ttft:>8.2f}s{t.load:>8.2f}s"
            f"{t.eval_tokens:>9}{t.eval_rate:>9.1f}"
        )


if __name__ == "__main__":
    main()
