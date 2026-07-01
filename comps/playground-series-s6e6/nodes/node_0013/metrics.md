# node_0013 metrics
metric: Balanced Accuracy Score (maximize)
per_fold: [0.963893, 0.964049, 0.963510, 0.964540, 0.963890]
cv: 0.963977 ± 0.000167   (oof_metric=0.963977)   parent node_0006=0.965004
n_features: 37
shuffled_cv: 0.33233   (baseline=0.333, control=PASS)
change: node_0006 LightGBM + leak-safe positional features (sin/cos RA, unit-sphere xyz,
delta×redshift, sky_cell native categorical, knn_dist5 sky-density). All label-free.
