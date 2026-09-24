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

Model choice is now measured rather than guessed. Two corpora with checkable
answers, scored by whether the reply claims a failure:

- `~/rust2/logs/*.log` — 67 cargo builds, 10 broken (ground truth: a line
  starting with `error`)
- `~/etc/*.log` — 39 command logs, 1 broken (`update-src.log`, a git pull
  refused over unstaged changes)

318 runs over both, after the `Summary:` prefill landed:

| model | accuracy | avg | worst mistake |
| --- | --- | --- | --- |
| `qwen3:4b` | 99% (105/106) | 5.6s | missed `update-src`, but the counted-failures warning fired |
| `phi4-mini:3.8b` | 97% (103/106) | 5.8s | true false negative on `logs.log` (`could not find Cargo.toml`) |
| `gemma3:4b` | 96% (102/106) | 6.2s | invented "Service failed to respond to heartbeat" from a file of RPM package names |

- [x] Score models against saved logs with known answers
- [x] Demote `nemotron-3-nano:4b` (3 outright failures in 17 `/var/log` runs,
      plus a fabricated finding on `gpu-manager.log`)
- [ ] Commit the two corpora, or a trimmed copy, so the scores can be
      reproduced from the repo instead of from this machine's home directory
- [ ] Script the scoring (`--csv` from item 6 would feed it) rather than
      keeping it in a throwaway shell script
- [ ] Both `gemma3` and `phi4-mini` fail on `ffsend.log`: a build that
      succeeds while printing 62 warnings, including `enum Error is never
      used`. Detecting "ends in Finished/Build succeeded" would fix the one
      case every model gets wrong
- [ ] Re-score on the MacBook once `gemma4:26b`, `gpt-oss:20b` and
      `qwen3:30b` are available — the ordering above only covers 4B models
