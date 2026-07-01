---
name: kaggle-kernel
description: Publish a Kaggle notebook (kernel) for a model — build an .ipynb with markdown + code cells, attach it to the competition, push it PRIVATE for review. Use when the human says "publish/upload a kernel/notebook", "make a Kaggle notebook for this model", or wants to share a node's solution as a runnable Kaggle notebook.
argument-hint: <slug> <node_id>   e.g. playground-series-s6e6 node_0140
allowed-tools: Bash, Read, Write, Edit
---

# /kaggle-kernel — publish a model as a private Kaggle notebook

Turn a built node's solution into a clean, self-contained Kaggle **notebook**,
**attached to the competition** and **always private** (the human reviews before
making anything public). Two rules never change: **`is_private: true`** and
**`competition_sources: [<slug>]`**.

## Steps

**1. Start from VERIFIED code.** Use the node's `src/solution.py` (or the
champion's). **Do not rewrite the model/feature/training logic** — slice it
verbatim so the score is preserved. Only the *orchestration* (load → build
features → loop → write `submission.csv`) and presentation may change. Replace
repo-only paths with Kaggle ones: read from `/kaggle/input/<slug>/`, write
`submission.csv` to the working dir. Regenerate folds in-kernel with the SAME
scheme as `validation.md` (e.g. `StratifiedKFold(n_splits, shuffle=True,
random_state=<seed>)`).

**2. Build the notebook** as a real `.ipynb` (JSON: `cells`, `nbformat: 4`).
Alternate **markdown above each code cell** (plain language, what the next cell
does + why) and **a print after each code cell** where a result is worth seeing
(data shapes, feature counts, per-fold scores, final CV, submission head). Keep
language simple. A small generator script that reads the verified `.py` and
slices it into cells is the safest way (no retyping the model).

**3. Verify before pushing** — combined code cells must compile:
```bash
uv run --no-sync python -c "import json,ast; nb=json.load(open('<dir>/<name>.ipynb')); \
ast.parse('\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')); print('cells OK')"
```

**4. Write `kernel-metadata.json`** next to the `.ipynb` (private + comp always):
```json
{
  "id": "<kaggle_username>/<slug>-<short-name>",
  "title": "<Short Title>",
  "code_file": "<name>.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": false,
  "dataset_sources": [],
  "competition_sources": ["<slug>"],
  "kernel_sources": []
}
```
Get the username from `$KAGGLE_USERNAME`. `enable_gpu`: true only if the model
needs it. `enable_internet`: false (turn on only if a cell pip-installs).

**5. Push** (auth first; use `python -m kaggle`, the bare `kaggle` binary isn't on PATH):
```bash
set -a && . ./.env && set +a && : "${KAGGLE_KEY:=$KAGGLE_TOKEN}" && export KAGGLE_KEY
uv run --no-sync python -m kaggle kernels push -p <dir>
uv run --no-sync python -m kaggle kernels status <username>/<id-slug>   # RUNNING → COMPLETE
```
Report the URL (`https://www.kaggle.com/code/<username>/<id-slug>`) and that it's
private + attached.

## Gotchas
- **`no kernel image is available for execution` (CUDA arch error):** the session's
  GPU is newer than the env's torch build. Tell the human to switch Accelerator to
  **T4 / P100** (or use the latest environment). Not a code bug.
- **Submitting:** the human submits from a **committed version's Output tab**
  (Save Version first), not the interactive editor. The file must be `submission.csv`.
- One-time human gates (accept rules, phone-verify) and the daily submission limit
  still apply — see `kaggle-io`.
