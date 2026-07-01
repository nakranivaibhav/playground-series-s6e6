# node_0010 metrics
metric: Balanced Accuracy Score (maximize)
per_fold: [0.966421, 0.965800, 0.965836, 0.965576, 0.965811]
cv: 0.965889 ± 0.000141   (HONEST nested fold weights)
full_oof_balacc: 0.965963
final_weights: {node_0006:0.250, node_0004:0.150, node_0001:0.250, node_0009:0.350}
per_fold_weights: [(0.2, 0.2, 0.25, 0.35), (0.25, 0.2, 0.25, 0.3), (0.25, 0.15, 0.25, 0.35), (0.25, 0.15, 0.25, 0.35), (0.3, 0.2, 0.2, 0.3)]
shuffled_cv: 0.33298   (baseline=0.333, control=PASS)
change: combine — fold-honest weighted blend of ['node_0006', 'node_0004', 'node_0001', 'node_0009'] (adds TabM to node_0007). No retrain.
