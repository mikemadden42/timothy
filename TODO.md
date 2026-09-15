# TODO

Ordered by priority. Items 1 and 5 are quick fixes; do 2 before adding
bigger features so they only need to be written once.

## 1. Name the model sets by whether they fit in VRAM

The `medium` set is now `gemma4:26b`, `gpt-oss:20b`, and `qwen3:30b` — all
installed, so the default run works. But at 18/13/18 GB they offload to CPU on
8 GB VRAM, which makes "medium" the wrong label. Meanwhile none of the `small`
models are installed, so `--size small` fails on every model with
"model not found".

- [ ] Rename the sets by whether they fit in 8 GB VRAM — `small` (fits) and
      `large` (offloads to CPU: `gemma4:26b`, `gpt-oss:20b`, `qwen3:30b`)
- [ ] Pull `phi4-mini:3.8b`, `gemma3:4b`, and `qwen3:4b`, or point the small
      set at models that are actually installed

## 2. Move shared code into one module

About 100 lines are duplicated between the scripts: `SIZES`, the constants,
`OLLAMA_ERRORS`, `Spinner`, `unload_all`, `warmup`, and `parse_args`. They
have already drifted twice (the token limit and the `qwen3:4b` entry).

- [ ] Create a shared module (e.g. `src/bench.py`)
- [ ] Leave each script with only its own `run()` and results table

## 3. Run each model several times

A single run is noisy: identical `phi4-mini` runs ranged from 44.4 to
47.9 tok/s.

- [ ] Add a `--runs N` option
- [ ] Report the median per model

## 4. Check for missing models up front

- [ ] Compare the selected set against `client.list()` before starting
- [ ] List missing models with the `ollama pull` commands to fix them

## 5. Clean up on Ctrl+C

Interrupting a long run skips the final unload, leaving the model in VRAM
for Ollama's 5-minute keep-alive.

- [ ] Wrap `main()` in `try`/`finally` so `unload_all()` always runs

## 6. Save results to a file

- [ ] Add `--csv` and/or `--json` output to compare runs over time and
      across model sets

## 7. Project housekeeping

- [ ] Fill in `README.md` (currently empty)
- [ ] Fix the `timothy` script in `pyproject.toml`: it points at the
      placeholder `src/timothy/__init__.py`, which only prints
      "Hello from timothy!" — wire it to the benchmarks or remove it
