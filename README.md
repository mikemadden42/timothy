# timothy

Benchmark local [Ollama](https://ollama.com) models by asking them all the same
question and comparing how fast they answer.

Named after timothy hay, a favorite food of llamas — the connection to Ollama,
which runs the models.

Two scripts run the same benchmark:

- `src/streaming_chat.py` — streams tokens as they arrive, and also reports
  time to first token and time to answer
- `src/basic_chat.py` — waits for the whole response, then prints it

A third puts the models to work: `src/triage.py` reads an error log and
summarizes what failed. See [Log triage](#log-triage).

## Requirements

- Python 3.14+ and [uv](https://docs.astral.sh/uv/)
- A running Ollama server with the models you want to test already pulled

## Usage

```sh
uv run src/streaming_chat.py            # medium model set (default)
uv run src/streaming_chat.py --size small
uv run src/basic_chat.py --size small
```

Each run asks for the question interactively, so it never lands in shell
history:

```
Question [Why is the sky blue?]: How many raccoons exist in the world?
```

Press Enter on a blank line to use the default question, or Ctrl+C / Ctrl+D to
quit before any model loads.

## Output

Each model streams (or prints) its answer, followed by a per-model summary, and
the run ends with a table sorted by tokens per second:

```
model                    wall     ttft      tta     load   tokens    tok/s
phi4-mini:3.8b          0.15s    0.07s    0.07s    0.00s        6     71.7
gemma3:4b               0.16s    0.07s    0.07s    0.00s        6     64.9
nemotron-3-nano:4b      1.33s    0.13s    1.19s    0.00s       59     49.1
qwen3:4b                6.77s    0.03s    6.67s    0.00s      326     48.3
```

| Column | Meaning |
| --- | --- |
| `wall` | Total time for the request. Reflects how much a model writes, not just its speed. |
| `ttft` | Time to the first token, whether that token is reasoning or answer (streaming only). |
| `tta` | Time to the first *answer* token; `-` if the model only ever produced reasoning (streaming only). |
| `load` | Model load time reported by Ollama. Should be `0.00s`, since warmup loads the model first. |
| `tokens` | Tokens generated, marked `capped` if the model hit the token limit. |
| `tok/s` | Generation rate — the fairest speed comparison between models. |

Reasoning models stream their thinking first; `streaming_chat.py` prints an
`--- answer ---` separator where the answer begins.

## Model sets

`SIZES` in each script defines the sets `--size` selects:

```python
SIZES = {
    "small": ["phi4-mini:3.8b", "gemma3:4b", "nemotron-3-nano:4b", "qwen3:4b"],
    "medium": ["gemma4:26b", "gpt-oss:20b", "qwen3:30b"],
}
```

Edit those lists to benchmark other models. A model that is not pulled fails
with "model not found" and is skipped; the rest of the set still runs.

## How the benchmark stays fair

- **One model at a time.** Ollama keeps a model in memory for 5 minutes by
  default, so leftover models can push the next one onto the CPU. Every run
  evicts all loaded models and waits until none are loaded, and the last model
  is unloaded when the run finishes.
- **Warmup before timing.** A throwaway request loads the model, so load time
  is not charged to the timed request.
- **Same context for both requests.** Warmup and the timed request use the same
  `NUM_CTX`; otherwise Ollama reloads the model and the reload lands in the
  timing.
- **Room to finish.** `NUM_PREDICT = 4096` with `NUM_CTX = 8192` lets reasoning
  models think and still answer within the context.

## Log triage

Pipe an error log in, or pass a file, and get four lines back: summary, likely
cause, a quoted log line as evidence, and what to check next.

```sh
pytest 2>&1 | uv run src/triage.py
journalctl -p err -b | uv run src/triage.py
uv run src/triage.py build.log --note "started after a system upgrade"
```

```
Summary: Linker failed due to missing libssl.
Cause: The system could not find the libssl library for linking; guess: wrong library path or missing installation.
Evidence: `error: linker failed /usr/bin/ld: cannot find -lssl: No such file or directory`
Next: Check if libssl is installed and in the linker path using `ldconfig -v | grep ssl`.
```

- Picks the first installed model from `MODEL_PREFERENCE`; override with
  `--model`.
- Thinking is left to each model's own default, which is what the bigger
  reasoning models want. `--no-think` forces it off for a model that thinks
  without ever answering (`qwen3:4b` does this), and `--think` forces it on.
- Terminal colors are stripped and repeated lines collapsed. A log too big to
  send whole is filtered down to the error-ish lines and their context, capped
  at a few copies of each repeated message, plus the tail. On a 1.4 MB syslog
  that is 125 error lines instead of the 10 a plain head/tail slice caught.
- Reading stops at the end of the `Next:` line, so a chatty model can't run on
  past the answer (this took `qwen3:4b` from 75s to 9s).
- The most repeated failure lines are counted in Python, shown after the answer
  and handed to the model as a hint. A build log often holds several unrelated
  failures, and the four-line format can only name one:

  ```
  repeated failures:
      12x error: linking with `clang` failed: exit status: 1
       2x error: failed to run custom build command for `onig_sys v69.8.1`
       2x error: failed to run custom build command for `openssl-sys v0.9.61`
  ```

- The model name and the counts go to stderr, so `triage.py build.log | pbcopy`
  copies only the answer.
- `--timeout` (default 120s) gives up on a model that is still thinking. An
  offloaded model generating at 4 tok/s would otherwise take 15+ minutes to
  spend its token budget.

Small models sometimes stitch together or slightly misquote the "Evidence" line,
so check it against the log before trusting it. A log containing chat
transcripts (an Ollama server log, say) can also pull a small model into
answering the conversation instead of triaging it; the prompt warns against
this, but `gemma3:4b` follows that far better than `nemotron-3-nano:4b` does.

## Development

```sh
uv sync            # install dependencies
uv run ruff check  # lint
uv run ruff format # format
```

See [TODO.md](TODO.md) for planned improvements.
