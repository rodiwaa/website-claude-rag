# Evals on Every Commit (pre-commit hook)

## Intent

Automatically run the eval suite before every `git commit` to catch regressions as they happen, rather than letting quality drift accumulate unnoticed. The hook blocks the commit if evals fail, providing a lightweight quality gate with no extra CI infrastructure required.

## What Was Added

A pre-commit hook that runs `make eval` on every commit, with an escape hatch for fast commits when needed.

## How It Works

### 1. Pre-commit hook (`.git/hooks/pre-commit`)

Git invokes this script before writing the commit. If `make eval` exits non-zero the commit is blocked. If `SKIP_EVALS=1` is set in the environment, the hook exits immediately with a skip message.

```
git commit:
  1. Git stages snapshot
  2. pre-commit hook fires
     ├── SKIP_EVALS=1? → skip, exit 0
     └── else → make eval
          ├── Requires existing LangSmith dataset (errors if missing)
          └── LLM-as-judge evaluation (5 pairs × 3 metrics)
  3. Commit written (or blocked on non-zero exit)
```

### 2. Version-controlled hook source (`scripts/pre-commit`)

Since `.git/hooks/` is not committed to the repo, the canonical hook source lives at `scripts/pre-commit`. Teammates install it after cloning via `make install-hooks`.

### 3. Makefile targets

| Target | Command | Purpose |
|---|---|---|
| `make install-hooks` | `cp scripts/pre-commit .git/hooks/pre-commit && chmod +x` | Install hook after fresh clone |
| `make skip-commit` | `SKIP_EVALS=1 git commit $(ARGS)` | Commit without running evals |
| `make eval` | `uv run python run_evals.py` | Run evals against existing stable dataset |
| `make eval-fresh` | `uv run python run_evals.py --fresh` | Delete dataset, regenerate QA pairs, then evaluate |

## Usage

```bash
# First-time setup — generate the stable golden dataset
make eval-fresh

# Normal commit — evals run automatically against stable dataset
git commit -m "update retrieval logic"

# Fast commit — skip evals
SKIP_EVALS=1 git commit -m "fix typo"

# Or via Makefile
make skip-commit ARGS='-m "fix typo"'

# After cloning fresh
make install-hooks

# Regenerate dataset (e.g. after adding new Qdrant docs)
make eval-fresh
```

## Stable Golden Dataset

`make eval` requires a pre-existing LangSmith dataset (`career-rag-golden`). It will error with a clear message if the dataset is missing — run `make eval-fresh` first to generate it. This ensures eval scores are comparable across commits because the questions never change between runs unless you explicitly regenerate them.

`make eval-fresh` is intentionally separate: use it only when the underlying Qdrant documents change or you want to reset the baseline.

## Committing Without Running Evals

Set `SKIP_EVALS=1` to bypass the pre-commit hook:

```bash
# Inline env var
SKIP_EVALS=1 git commit -m "fix typo"

# Or via Makefile shortcut
make skip-commit ARGS='-m "fix typo"'
```

The hook checks for this variable first and exits 0 immediately — no evals run, commit proceeds normally.

Use this when iterating on non-logic changes: docs, formatting, config, comments.

## Caveat: Eval Duration

The eval suite takes ~60–80 seconds (5 QA pairs × 3 LLM judge calls each). This adds latency to every commit. Use `SKIP_EVALS=1` when iterating quickly on non-logic changes (docs, formatting, config).

## Files Changed

- `scripts/pre-commit` — version-controlled hook source (new file)
- `.git/hooks/pre-commit` — active hook, installed and executable (not committed)
- `Makefile` — added `install-hooks`, `skip-commit`, and `eval-fresh` targets
- `run_evals.py` — added `--fresh` flag; normal path reuses existing dataset instead of auto-regenerating
