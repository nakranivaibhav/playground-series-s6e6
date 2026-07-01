# Why TabPFN / TabICL underperform on this dataset — and the one lever that changes it

**TL;DR:** Tabular foundation models (TabPFN, TabICL) are *in-context* learners with a hard context budget of roughly 10k rows. This dataset has **577k** training rows. Run them frozen and they only ever "see" ~2% of your data per prediction — which is why they land *below* a tuned LightGBM/TabM here. The lever that unlocks them is **fine-tuning**, not a bigger ensemble of frozen calls.

## The frozen numbers (so we're concrete)

Out of the box, on this competition's features, here's where the foundation models sit versus ordinary tuned models (Balanced Accuracy, same CV):

| model | mode | BA |
|---|---|---|
| TabPFN v2 | frozen, subsampled context | ~0.943 |
| TabPFN v2.5 (large-samples) | frozen | ~0.949 |
| TabICL | frozen | ~0.959 |
| tuned LightGBM / TabM | trained on full data | **~0.965–0.968** |

People see this and conclude "foundation models don't work for tabular." That's the wrong conclusion. Look at *why*.

## The mechanism: in-context learning is context-starved at scale

TabPFN/TabICL don't train on your data — they take a chunk of it as **context** (like a prompt) and predict the rest in a single forward pass. The architecture caps that context at ~10k rows (TabPFN v2). With 577k training rows:

- a frozen call conditions on **≤2%** of the available signal at a time;
- subsample-ensembling (averaging many 10k-row draws) helps a little but can't recover information that no single forward pass ever sees together;
- the rare classes (QSO, STAR here) are the ones most starved, which is brutal under **Balanced Accuracy**.

So the ceiling isn't the model's capacity — it's the **context window** vs **dataset size** mismatch. Frozen foundation models are built for *small* tables (hundreds–few thousand rows), where they're genuinely SOTA. At 577k rows you're using them out of their design regime.

## The lever: fine-tune the pretrained transformer

Fine-tuning runs gradient descent on the pretrained weights with *your* data, so the model's learned prior actually adapts to this distribution instead of re-deriving it from a 10k-row prompt each time. Two findings from recent work worth knowing:

1. **Gains scale with dataset size.** Fine-tuning helps *most* on large tables — exactly our situation. On small datasets the improvement over frozen ICL is often statistically insignificant; on large ones it's material. ([On Finetuning Tabular Foundation Models, arXiv:2506.08982](https://arxiv.org/abs/2506.08982))
2. **For TabPFN specifically, full fine-tuning beats LoRA / parameter-efficient tuning** — comparable accuracy but faster convergence, so reach for full-FT first, not LoRA. (same paper)

The official package makes it a drop-in:

```python
from tabpfn.finetuning import FinetunedTabPFNClassifier

clf = FinetunedTabPFNClassifier(
    device="cuda",
    epochs=30,
    learning_rate=3e-5,                       # 3e-5–1e-4 for 10k+ row datasets (default 1e-5 is too low)
    n_finetune_ctx_plus_query_samples=10000,  # chunk size per gradient step
    n_inference_subsample_samples=50000,      # support rows used at prediction time
)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)
```

### Practical gotcha: the offline/license gate

On Kaggle (no interactive terminal, internet often off) the fine-tuner will try to download weights and hit a license prompt. If the v2 checkpoint is already cached (e.g. from a prior frozen run, or added as a dataset), pass it straight through and it skips the gate:

```python
ckpt = "/kaggle/input/tabpfn-v2-weights/tabpfn-v2-classifier.ckpt"   # wherever it lives
clf = FinetunedTabPFNClassifier(..., extra_classifier_kwargs={"model_path": ckpt})
```

GPU memory is modest — a fold of this size fine-tunes in a few GB of VRAM with activation checkpointing on, so a single T4/P100 is enough.

## What I'm *not* claiming

I'm not posting a "fine-tuned TabPFN = new SOTA here" number — that's still cooking, and on a dataset this thoroughly modeled the honest expectation is a *useful base*, not a miracle. The point of this post is the **diagnosis**: if you tried a foundation model frozen, saw ~0.95, and shelved it — that result is an artifact of context-starvation, not the model's ceiling. Fine-tuning is the missing step, and it's one constructor away.

If you want the other SOTA options: **Mitra** (Amazon, ships inside AutoGluon, fine-tune built in, trained on a tree-ensemble + SCM prior mix) and **TabICLv2** (natively scales to ~500k rows) are both worth a look. Happy to compare notes in the comments. ⚡
