# TODO

Ordered by priority. Items 1 and 5 are quick fixes; do 2 before adding
bigger features so they only need to be written once.

## 1. Fix the default model set

`DEFAULT_SIZE` is `medium`, and none of `gemma4:26b`, `gpt-oss:20b`, or
`qwen3:30b` are installed on this machine, so a plain run fails on every model
with "model not found". The whole `small` set is installed and runs fine.

On the 8 GB laptop those medium models also offload to CPU (18/13/18 GB), which
makes "medium" the wrong label; on the 36 GB MacBook they fit.

- [ ] Point the default at a set that is actually installed
- [ ] Rename the sets by whether they fit in VRAM — `small` (fits) and `large`
      (offloads to CPU), or pull the medium models

## 2. Move shared code into one module

Now duplicated across three scripts: `Spinner`, `OLLAMA_ERRORS`, and in the two
benchmarks also `SIZES`, the constants, `unload_all`, `warmup`, `ask_prompt`,
and `parse_args`. They have already drifted twice (the token limit and the
`qwen3:4b` entry), and `triage.py` carries a third copy of `Spinner`.

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

- [x] Fill in `README.md`
- [ ] Fix the `timothy` script in `pyproject.toml`: it points at the
      placeholder `src/timothy/__init__.py`, which only prints
      "Hello from timothy!" — wire it to the scripts or remove it

## 8. Triage follow-ups

`src/triage.py` works (see the README), but the small models vary a lot on the
same log. Two things would make that measurable instead of anecdotal:

- [ ] Keep a few saved logs with known answers (the mold linker failure, the
      sssd socket failures, a log with no errors) and score models against them
- [ ] Consider dropping `nemotron-3-nano:4b` from `MODEL_PREFERENCE`: it
      repeatedly ignores the four-line format and answers chat transcripts
      found inside logs
