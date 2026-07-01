# node_0007 metrics
metric: Balanced Accuracy Score (maximize)
per_fold: [0.966235, 0.965247, 0.965202, 0.965577, 0.965391]
cv: 0.965530 ± 0.000188   (HONEST nested fold weights)
full_oof_balacc: 0.965580   (optimistic — weights fit on all OOF)
final_weights: {node_0006:0.400, node_0004:0.400, node_0001:0.200}
per_fold_weights: [(0.4, 0.35, 0.25), (0.45, 0.35, 0.2), (0.4, 0.4, 0.2), (0.4, 0.4, 0.2), (0.4, 0.4, 0.2)]
reference: champion_n6=0.965004, n6+n4_50/50=0.965498, uniform=0.965460
shuffled_cv: 0.33303   (baseline=0.333, control=PASS)
change: combine — fold-honest weighted-probability-average of ['node_0006', 'node_0004', 'node_0001']. No model retrain.
