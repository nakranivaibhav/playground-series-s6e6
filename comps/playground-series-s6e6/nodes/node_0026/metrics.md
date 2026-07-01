# node_0026 metrics
metric: Balanced Accuracy Score (maximize)
model: TabICL v2 (tabicl, n_estimators=8, context_size=100000, kv_cache=repr), CUDA
per_fold: [0.959432, 0.959371, 0.958361, 0.958788, 0.959005]
cv: 0.958991 +/- 0.000197   (oof_metric=0.958991)
change: TabICL foundation model; class-balanced context subsample from train-fold only; kv_cache=repr for efficient query prediction.
