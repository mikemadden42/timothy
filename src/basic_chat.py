#!/usr/bin/env python3

import sys
import time
from dataclasses import dataclass

import ollama

models = ["gemma4:12b", "gpt-oss:20b", "phi4:14b"]

NS_PER_S = 1_000_000_000
NUM_PREDICT = 128


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
        warmup(client, model)

        start = time.perf_counter()
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": NUM_PREDICT},
        )
        wall = time.perf_counter() - start
    except ollama.ResponseError as err:
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


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "Why is the sky blue?"

    client = ollama.Client()
    timings = [t for model in models if (t := run(client, model, prompt))]

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
