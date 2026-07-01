# node_0008 metrics
metric: Balanced Accuracy Score (maximize)
model: PyTorch MLP [256,128,64]+BN+Dropout0.2, class-weighted CE, Adam, early-stop, CUDA
per_fold: [0.955425, 0.955669, 0.954312, 0.954113, 0.955325]
cv: 0.954969 ± 0.000315   (oof_metric=0.954969)
input_dim: 32
shuffled_cv: 0.33333   (baseline=0.333, control=PASS)
change: non-GBDT diversity arm (MLP on CUDA). Standardization fit-inside-fold. Saves oof+test_probs.
