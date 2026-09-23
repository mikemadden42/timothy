# Ideas

Things to build with local Ollama models, beyond the chat benchmarks.

## Daily-use tools

Ordered by how often they would earn their keep. Start with the commit message
writer: smallest one, used several times a day, and the git hook makes it
automatic.

### 1. Commit message writer

Pipe `git diff --cached` to a model and get a subject line plus bullets. Wire it
into `.git/hooks/prepare-commit-msg` so the message is pre-filled on every
`git commit` and you just edit it.

- Diffs never leave the machine
- A 4B model is good enough at summarizing a diff

### 2. Shell command helper

Type what you want in plain words, get the command. Bind it to a zsh key so it
drops the command on the prompt line without running it; press Enter if it looks
right.

- For `rsync`, `ffmpeg`, `find -exec`, `awk`, `jq` — the ones re-looked-up every
  time

### 3. Pre-push diff review

Run the working-tree or branch diff past a model before pushing. Ask only for
things that look wrong: debug prints, hardcoded paths, a flag flipped and
forgotten.

- Cap it at three notes or "looks fine" so it stays a 10-second habit

### 4. Error and log triage — built, see `src/triage.py`

Pipe a stack trace, failing test log, or `journalctl -p err -b` in; get a summary
with the likely cause.

- Logs are the thing least worth pasting into a web form
- Where a bigger model on the Mac pays off

### 5. Clipboard rewriter

Take the clipboard (`wl-paste` on Linux, `pbpaste` on macOS), tighten or re-tone
it, put it back. For emails, PR descriptions, Slack messages.

### 6. Release notes from git log

Feed it `git log v1.2..HEAD`, get grouped human-readable notes. Tag-time rather
than daily, but turns a chore into one command.

### 7. Regex and jq generator with a check

Describe the goal plus a sample line, get the expression, then run it against the
sample and show what matched. The verification step is what makes it
trustworthy — a wrong regex is worse than no regex.

### 8. Screenshot to text

Point a vision model (`gemma3`, `muse-glimmer`) at the newest screenshot and
extract the text. For error dialogs and images of code.

### 9. Man page and `--help` summarizer

`tldr`, but generated on demand from the actual `--help` on this machine, so it
matches the installed version.

### 10. Dependency update summarizer

Feed it the `uv.lock` diff after `uv sync -U`; get a plain-language summary of
what moved and whether anything looks risky.

## Benchmark scripts

Moving from "how fast?" to "can it do the job?"

### Structured output conformance

Ask every model for JSON matching a schema and check whether it validates.
`client.chat()` takes `format=` (accepts a JSON schema), and `pydantic` is
already installed as an `ollama` dependency.

- Measures: valid-JSON rate, correct field types, correct values, and how much
  the schema slows generation
- Smallest step beyond the current scripts: same loop, `format=` added,
  validation instead of a timer

### Tool-calling benchmark

`client.chat()` takes `tools=`. Give each model two or three fake tools
(`get_weather(city)`, `add(a, b)`, `search(query)`) and prompts where the right
move is obvious — including a couple where the right move is to call nothing.

- Measures: right tool picked, arguments parse, no invented tools
- `phi4-mini`, `qwen3`, `nemotron` advertise tool support; `gemma3` does not

### Quality scoring with a judge model

Keep a small file of prompts with expected answers, run them across the set, and
have a larger model (e.g. `gpt-oss:20b`) grade each response 1-5 with a reason.

- Measures answer quality, which the speed tables cannot show
- Reuses `SIZES` and the unload/warmup discipline; output is a scoreboard

### Others

- **Local RAG:** `client.embed()` exists. Index a folder of notes, retrieve top
  matches, answer from them. Measures retrieval hit rate and end-to-end latency.
- **Long-context accuracy:** hide a fact mid-document and ask for it at 2K/8K/32K
  tokens. Tests `NUM_CTX` directly; models degrade differently.
- **Quantization comparison:** same model at `q4_K_M`, `q8_0`, `bf16`. Shows what
  VRAM buys in quality — matters far more on the 8 GB laptop than the 36 GB Mac.
- **Concurrency and throughput:** fire N requests at once, watch total tok/s.
  Answers "can this serve more than one user?" and stresses memory.
- **Vision:** `gemma3` and `muse-glimmer` accept images. Score captions or text
  extraction over a folder of screenshots.
