# champion — node_0091 (full-pool L2 LogReg mega-stack)
metric: Balanced Accuracy Score (maximize)
CV: 0.970355 (sem 0.000249) · public LB: 0.97121 · promoted 2026-06-14 (prev champ node_0063 cv 0.970153 / lb 0.97073)

What it is: balanced multinomial LogReg meta on clipped log-probs over the FULL pool
(Deotte public bank-17 + FT-Transformer external OOF + ~54 in-house base OOFs),
with the inverse-strength C tuned IN-FOLD (nested) — winner C=0.003. Strong L2
shrinkage absorbs the weak/correlated in-house bases instead of diluting, netting
the best honest CV AND the best honest LB (beats the lean bank17+FT-T node_0070
0.97087 on LB by +0.00034).

Reproduce:
  uv run --no-sync python comps/playground-series-s6e6/nodes/node_0091/src/solution.py
  # reads the frozen folds.json + every base node's oof.npy/test_probs.npy + refs bank-17 + FT-T;
  # writes oof.npy, test_probs.npy, submission.csv. Full detail: nodes/node_0091/node.md.
