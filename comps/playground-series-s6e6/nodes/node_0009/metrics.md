# node_0009 metrics
metric: Balanced Accuracy Score (maximize)
model: TabM (library tabm==0.0.3, k=32 ensemble) + PiecewiseLinearEmbeddings (rtdl), CUDA
per_fold: [0.964402, 0.962802, 0.965039, 0.964408, 0.964424]
cv: 0.964215 ± 0.000374   (oof_metric=0.964215)
shuffled_cv: 0.33333   (baseline=0.333, control=PASS)
change: TabM via official library; target-aware bins + standardization fit-inside-fold; native cat embeddings. Saves oof+test_probs.
