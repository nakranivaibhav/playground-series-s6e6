---
id: node_0060
desc: LightGBM + SDSS17 provenance flags (C5)
op: improve
parents: [node_0030]
uses_data: [fs_realmlp_fe]
family: gbdt
status: valid
stage: scored
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966623
sem: null
folds: [0.967302, 0.967073, 0.966800, 0.966011, 0.965929]
baseline_cv: 0.966952
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-10
decided: 2026-06-10
tags: [data-centric, provenance, generator-forensics, null]
---

## plan
built on:   node_0030 LightGBM-richFE, byte-identical except 3 added stateless features.
change:     add _alpha_in_orig / _delta_in_orig / _both_in_orig — exact-value membership of alpha/delta in the real SDSS17 catalog (label-free: only feature values read, never orig class).
hypothesis: generator reuses real coordinate VALUES (32.5% alpha / 38.1% delta match orig vs ~1.5% mags); membership carries class signal (both_in STAR 17.7% vs neither 12.5%).
target:     beat champion 0.969808 by >=2sem in a restack.

## notes
Solo REGRESSED -0.000329 (folds 0/1 up, 2/3/4 down — overfits like dead positional FE). Restack all worse (bank17+new -0.000055). NULL. Kept datum: coordinate-value reuse is a real generator artifact, not exploitable. fs note: provenance flags are a new stateless sub-set of fs_realmlp_fe (membership of base alpha/delta), no data.md feature-set promoted since null.
