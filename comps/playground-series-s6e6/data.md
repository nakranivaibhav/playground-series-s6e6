# playground-series-s6e6 — data lineage
sources: train.csv, test.csv · cleaning: eda.md (data already clean — type hygiene only) · updated 2026-06-07T06:06Z

Engineered feature-sets and which nodes consume them. Recipes live in
`src/clean.py` (one function per group). Experiment lineage is in `graph.md`;
this map is the **data** side (updated 2026-06-22T06:13Z). Leak-safety: **stateless** = row-wise, no fit, safe
to share; **fit_in_fold** = needs a train-only reference (target-encode, scaler,
or a cross-row stat), so it must be (re)built inside each train fold.

```mermaid
graph LR
    raw[(raw train/test)] --> base[base · type hygiene]
    base --> fs_realmlp_fe[fs_realmlp_fe · ref-recipe FE · stateless]
    raw --> fs_sed_image[fs_sed_image · 7-ch rest-frame SED-texture image · mostly stateless +MTF/std fit_in_fold]
    fs_sed_image --> node_0057
    fs_sed_image --> node_0058
    base --> fs_colors[fs_colors · adjacent colors · stateless]
    fs_colors --> fs_research[fs_research · ext-colors+redshift+qso+galactic · stateless]
    fs_research --> fs_positional[fs_positional · sky geometry + kNN density · fit_in_fold]
    fs_research --> fs_tgt_enc[fs_tgt_enc · smoothed P(class|cat) + redshift-band TE · fit_in_fold]
    fs_colors --> node_0001
    fs_research --> node_0006
    fs_research --> node_0009
    fs_research --> node_0011
    fs_research --> node_0014
    fs_research --> node_0015
    fs_research --> node_0016
    fs_research --> node_0018
    fs_research --> node_0019
    fs_research --> node_0021
    fs_research --> node_0022
    fs_research --> node_0024
    fs_research --> node_0025
    fs_research --> node_0026
    fs_research --> node_0027
    fs_realmlp_fe --> node_0028
    fs_realmlp_fe --> node_0030
    fs_realmlp_fe --> node_0031
    fs_realmlp_fe --> node_0032
    fs_realmlp_fe --> node_0033
    fs_realmlp_fe --> node_0034
    fs_realmlp_fe --> node_0035
    fs_realmlp_fe --> node_0036
    fs_realmlp_fe --> node_0037
    fs_realmlp_fe --> node_0038
    fs_realmlp_fe --> node_0042
    fs_realmlp_fe --> node_0043
    fs_realmlp_fe --> node_0049
    fs_realmlp_fe --> node_0050
    fs_realmlp_fe --> node_0051
    fs_realmlp_fe --> node_0055
    fs_realmlp_fe --> node_0056
    fs_realmlp_fe --> node_0065
    fs_realmlp_fe --> node_0066
    fs_realmlp_fe --> node_0067
    fs_realmlp_fe --> node_0079
    fs_realmlp_fe --> node_0081
    fs_realmlp_fe --> node_0082
    fs_realmlp_fe --> node_0083
    fs_realmlp_fe --> node_0085
    fs_realmlp_fe --> node_0090
    fs_realmlp_fe --> node_0115
    base --> fs_sdss_labels[fs_sdss_labels · external SDSS DR17 coord-matched real class · fit_in_fold]
    fs_sdss_labels --> node_0083
    base --> fs_physics_locus[fs_physics_locus · physics redshift/color-locus residuals · stateless]
    fs_physics_locus --> node_0085
    base --> fs_zresid[fs_zresid · z-conditional color/mag residual z-scores · fit_in_fold]
    fs_zresid --> node_0086
    fs_zresid --> node_0087
    base --> fs_errpocket_w[fs_errpocket_w · error-pocket up-weighting sample weights · fit_in_fold]
    fs_errpocket_w --> node_0094
    fs_realmlp_fe --> node_0094
    base --> fs_zresid_strict[fs_zresid_strict · STRICT z-color-residuals + STAR-flag ONLY · fit_in_fold]
    fs_zresid_strict --> node_0095
    fs_realmlp_fe --> fs_rbf_nystroem[fs_rbf_nystroem · Nystroem RBF random-feature map of core photometric vector · fit_in_fold]
    fs_rbf_nystroem --> node_0096
    fs_colors --> node_0023
    fs_positional --> node_0013
    fs_tgt_enc --> node_0018
    base --> fs_genfp[fs_genfp · per-feature generator decimal/mantissa fingerprints · stateless]
    fs_genfp --> node_0118
    fs_realmlp_fe --> node_0118
    fs_realmlp_fe --> fs_synthpre[fs_synthpre · per-class tabular-generator synthetic rows for NN pretrain · fit_in_fold]
    fs_synthpre --> node_0119
    fs_realmlp_fe --> node_0120
    base --> fs_sdr[fs_sdr · SDR sharpened-manifold embedding · fit_in_fold]
    fs_sdr --> node_0121
    base --> fs_zsoft[fs_zsoft · redshift-error-aware z-warp + soft STAR-likelihood · stateless]
    fs_zsoft --> node_0140
    fs_realmlp_fe --> node_0140
    base --> fs_ambig[fs_ambig · bank-OOF-derived per-row ambiguity aux-target · fit_in_fold]
    fs_ambig --> node_0139
    fs_realmlp_fe --> node_0139
    fs_realmlp_fe --> node_0133
    fs_realmlp_fe --> node_0135
    fs_realmlp_fe --> node_0137
    fs_realmlp_fe --> node_0141
    fs_realmlp_fe --> node_0142
    fs_realmlp_fe --> node_0143
    classDef warn fill:#fee,stroke:#c00;
    class fs_positional warn;
    class fs_tgt_enc warn;
    class fs_errpocket_w warn;
    class fs_zresid_strict warn;
    class fs_rbf_nystroem warn;
    class fs_synthpre warn;
    class fs_ambig warn;
```

## feature-sets
| id | what it is | derived from | recipe (`src/clean.py`) | leak-safety | produced by | consumed by |
|----|------------|--------------|--------------------------|-------------|-------------|-------------|
| fs_colors | adjacent-band colors u_g, g_r, r_i, i_z, u_z | base | `add_color_features` | stateless | node_0001 | n1, n3, n4, n5, n6, n8, n9, n11, n12, n13, n14, n15, n16, n23 |
| fs_research | full pairwise + curvature colors, log1p/regime-flag redshift, QSO color-box, galactic (l, b) | fs_colors | `add_extended_colors` + `add_redshift_features` + `add_qso_colorbox` + `add_galactic_coords` | stateless | node_0006 | n6, n8, n9, n11, n12, n13, n14, n15, n16, n18, n19, n21, n22, n24, n25, n26, n27 |
| fs_positional | sin/cos RA, unit-sphere xyz, delta×redshift, sky_cell(10°×5°, native cat), **knn_dist5** | fs_research | `add_positional_features` + node-local cKDTree | **fit_in_fold** (knn_dist5 is a train-reference cross-row stat) | node_0013 | n13 |
| fs_tgt_enc | smoothed (m=100) P(class\|spectral_type), P(class\|galaxy_population), + P(class\|redshift-band) over ~10 train-fold quantile bands | fs_research | `add_target_encoding` (smoothed posteriors + quantile-band TE, all fit train-fold-only) | **fit_in_fold** (target-encoded posteriors + fold-local redshift band edges) | node_0018 | n18 |
| fs_realmlp_fe | public RealMLP reference FE: redshift ratios (g/redshift, i/redshift), log1p(redshift), all 7 color pairs (u-g, g-r, r-i, i-z, u-r, g-i, r-z), mag_mean, mag_range, integer-floor categorical views of every base numeric, category cross-combos | base | FE in `refs/realmlp-v5-for-s6e6.py` (ports to `src/clean.py`); all row-wise deterministic, no fit/target/cross-row stats | stateless | node_0028 | n28, n30, n31, n32, n33, n34, n35, n36, n37, n38, n42, n43, n49, n50, n51, n55, n56, n65, n66, n67, n79, n81, n82, n83, n85, n90, n94, n115, n118, n119 (base for synth-gen), n120, n133, n135, n137, n139, n140, n141, n142, n143 |
| fs_zresid | z-conditional residual z-scores: for each color (u-g, g-r, r-i, i-z, u-z) and each magnitude, (value − mean_zbin)/std_zbin over ~40 redshift quantile bins; PLUS raw redshift kept (STAR z≈0); raw colors DROPPED (residual-dominated, not additive) | base (u,g,r,i,z,redshift) | new `add_zconditional_residuals` — redshift quantile-bin EDGES + per-bin color/mag MEAN/STD fit on train fold only, applied to val+test; global mean/std fallback for sparse bins (research.md lines 83-105). **NOTE: implemented independently in EACH consumer node's own src/ (identical recipe) — no cross-node import; this is the single canonical recipe** | **fit_in_fold** (bin edges + per-bin mean/std are train-fold cross-row references) | node_0086 | n86, n87 |
| fs_sdss_labels | external SDSS DR17 spectroscopic class attached to train/test rows coordinate-matched on (alpha,delta) within a small tol; matched real class as feature/sample-weight prior + match-coverage stats | base + external SDSS DR17 specObj/photoObj catalog | new `add_sdss_dr17_labels` (coord crossmatch); match-confidence/coverage stats computed train-fold-only | **fit_in_fold** (match-coverage stats are train-fold cross-row references; matched label itself external) | node_0083 | n83 |
| fs_physics_locus | physically-motivated residuals: distance from STAR locus (z≈0 color track), QSO UV-excess color-box residual, GALAXY red-sequence/blue-cloud color-magnitude track residual | base (u,g,r,i,z,redshift); reuses fs_research QSO color-box + redshift regime formulas | new stateless `add_physics_locus` in `src/clean.py`; all row-wise deterministic, no fit/target/cross-row stats | stateless | node_0085 | n85 |
| fs_sed_image | per-row 7-channel 32x32 SED-texture image: rest-frame-warped GAF/GADF/RP/MTF(s_rest) + GASF(s_obs) + zmod + support-mask, with 2 side scalars [mag_mean, redshift] | raw u,g,r,i,z,redshift | flux-shape f_b=10^(-0.4(mag_b−mag_mean)) → (1+z) rest-frame warp lam_rest=lam_obs/(1+max(z,−0.009)) over SDSS λ_eff [3543,4770,6231,7625,9134] → PCHIP resample(24) on common log-λ grid → pyts GAF/RP/MTF encode, pad 24→32, float16 memmap cache (adapts `viz_sed_images.py`) | **mostly stateless** EXCEPT MTF quantile bin edges (pooled train-fold s_rest only) + channel/side-scalar standardization, which are **fit_in_fold** | node_0057 | node_0057, node_0058 |
| fs_errpocket_w | per-train-row LightGBM sample_weight that UP-WEIGHTS rows in the bank's hardest error pockets (NOT a feature — used only as `sample_weight`). Error map = where node_0070's fold-honest OOF mispredicts, binned over redshift-quantile × magnitude × true-class cells; a train row's weight scales with its cell's error density (floor 1.0). Applied to TRAIN rows only; val/test scored unweighted | base (redshift, mags, true-class) + node_0070 fold-honest OOF (nodes/node_0070/oof.npy) | new `add_errpocket_weights`: read node_0070 OOF errors → redshift quantile-bin EDGES + per-cell (z-bin × mag-bin × class) error densities fit on the **train fold only**, applied to that fold's train rows; global density fallback for sparse cells. **NOTE: implemented in node_0094's own src/ (single canonical recipe)** | **fit_in_fold** (bin edges + per-cell error densities are train-fold cross-row references; weight is label-derived via the row's true class — train-fold-only, never val/test/full-train) | node_0094 | node_0094 |
| fs_zresid_strict | STRICT residual-only view: ONLY the z-conditional per-COLOR residual z-scores ((color − mean_zbin)/std_zbin over ~40 redshift quantile bins) + a single binary STAR-flag (1 if row in the z≈0 lowest bin). DROPS everything fs_zresid kept: raw redshift-as-continuous, raw magnitudes, raw colors, AND the per-magnitude z-scores. A stricter sibling of fs_zresid built to force a residual-only decision geometry | base (u,g,r,i,z,redshift) | z-binned color residuals: redshift quantile-bin EDGES + per-bin per-color MEAN/STD fit train-fold-only, global mean/std fallback for sparse bins (research.md lines 83-105); STAR-flag = z≈0 bin membership. **NOTE: STRICTER sibling of fs_zresid (drops raw-z-continuous + magnitudes + raw colors + mag z-scores); implemented in node_0095's own src/ (single canonical recipe)** | **fit_in_fold** (bin edges + per-bin color mean/std are train-fold cross-row references) | node_0095 | node_0095 |
| fs_rbf_nystroem | sklearn Nystroem RBF kernel-approximation map (n_components ~1000-2000, gamma fold-0 micro-sweep) of the ~13-dim standardized core photometric vector, into a high-dim random-feature space for a balanced linear/shallow-MLP head | the ~13-dim standardized core photometric vector: u,g,r,i,z,redshift + the 7 fs_realmlp_fe color pairs (u-g, g-r, r-i, i-z, u-r, g-i, r-z) | `StandardScaler` + `Nystroem(kernel="rbf")`: landmarks (sampled train rows) + scaler fit on the **train fold only**, fitted transform applied to val+test; gamma chosen by a cheap fold-0 micro-sweep then frozen. Implemented in node_0096's own src/ | **fit_in_fold** (Nystroem landmarks + StandardScaler are train-fold references) | node_0096 | node_0096 |

## notes
- **fs_research is the value feature-set** — it is the atomic change that made
  node_0006 the best single, and it propagates into the champion lineage (node_0007,
  node_0010 blend node_0006's OOF) and into n9/n11.
- **fs_positional is flagged `fit_in_fold`** because `knn_dist5` builds its KDTree
  reference from training positions. node_0013 built that reference on the **whole**
  train (not fold-local); it is label-free, so the shuffled-label control can't catch
  it — a small density leak the leak-safety class surfaces that the control misses.
  node_0013 also regressed (−6σ), so it's `valid (REGRESSED)`, not in the champion.
- Combine nodes (node_0007, node_0010) consume **OOF**, not feature-sets — that
  lineage is the `combine` edges in `graph.md`, so their `uses_data` is `[]`.
| fs_flux | linear FLUX-space representation: f_b = 10^(−0.4·(mag_b − mag_mean)) for b∈{u,g,r,i,z}; pairwise flux RATIOS f_b/f_b'; flux vector normalized to unit sum (SED-shape simplex); + raw redshift. NO magnitudes, NO log colors. (Hypothesis: linear-ratio geometry decorrelates TabM errors vs log-color space; KILL gate — flux ratio = exp(log-color), so if err-corr ≥0.75 it is just re-encoded color space) | base (u,g,r,i,z,redshift) | new stateless `add_flux_features` in node_0103's own src/; all row-wise deterministic, no fit/target/cross-row stats | stateless | node_0103 | node_0103 |
| fs_flux_rich | RICH flux-space FE (the flux analogue of fs_realmlp_fe): linear fluxes f_b=10^(−0.4(mag_b−mag_mean)), ALL pairwise flux ratios f_b/f_b', unit-sum SED simplex, flux aggregates (flux mean, range, brightest/faintest-band one-hot), flux×redshift + flux-ratio×redshift interactions, raw redshift + log1p(redshift), AND the two engineered categoricals (spectral_type, galaxy_population). NO raw magnitudes, NO log colors. (Follow-up to n103: restores the breadth n103 dropped to recover BA while keeping flux-geometry decorrelation 0.485) | base (u,g,r,i,z,redshift) + 2 categoricals | new stateless `add_flux_rich_features` in node_0107's own src/; all row-wise deterministic, no fit/target/cross-row stats | stateless | node_0107 | node_0107 |
| fs_luptitude | arcsinh-flux (luptitude, Lupton 1999) FE: mu_b=arcsinh(f_b/(2*soft)) with f_b=10^(-0.4(mag_b-mag_mean)) and a FIXED softening constant; 5 luptitudes + all 10 pairwise luptitude differences + luptitude mean/range + raw redshift + log1p(redshift) + 2 categoricals. Well-conditioned flux geometry interpolating linear-flux (faint) to magnitude (bright). NO raw magnitudes, NO log-colors | base (u,g,r,i,z,redshift) + 2 cats | new stateless add_luptitude_features in node_0108 src/; softening is a fixed constant (NOT data-fit) so row-wise deterministic | stateless | node_0108 | node_0108 |
| fs_genfp | per-feature synthetic-generator quantization fingerprints: for each base numeric (u, g, r, i, z, redshift, alpha, delta) — number of significant decimals, fractional/mantissa residual after rounding to k places, last-digit value, trailing-zero count. A NON-photometric provenance representation forensically probing the tabular generator's class-conditional float-quantization artifacts. Fed alongside fs_realmlp_fe to a dedicated LightGBM base | base (u, g, r, i, z, redshift, alpha, delta) | new stateless `add_genfp_fingerprints` in node_0118's own src/; per-row decimal/mantissa/last-digit/trailing-zero arithmetic, NO target/fit/cross-row stat | stateless | node_0118 | node_0118 |
| fs_sdr | SDR sharpened-manifold embedding: iteratively shift points toward higher-density regions (mean-shift-like local-gradient density sharpening) BEFORE a DR projection (UMAP/LMDS), then kNN-classify in the sharpened embedding. Source: "Supervised star/galaxy/QSO classification with sharpened dimensionality reduction," A&A 2024. Libraries-first pySDR/SHARC, else mean-shift loop over umap-learn + sklearn kNN. A representation-CLASS change (manifold geometry, not axis-aligned partitioning of the 26-D space) — aims to decorrelate from the entire bank | base photometric numerics (u, g, r, i, z, redshift, colors) | sharpening (mean-shift density-gradient) + UMAP/SDR DR projection + kNN-classify in the embedding | **fit_in_fold** (the embedding, density sharpening, and kNN are all cross-row stats → fit on the TRAIN FOLD ONLY each fold; val/test embedded via the fitted train-fold embedding; never fit on full train or test) | node_0121 | node_0121 |
| fs_synthpre | per-class tabular-generator SYNTHETIC ROWS used ONLY to PRE-TRAIN a TabM (NOT a feature view): fit a fast per-class generator (Gaussian-copula or small CTGAN/TVAE, libraries-first via sdv/ctgan) on the TRAIN-FOLD rows only, sample ~2-4M labelled synthetic rows, pretrain TabM on them, then fine-tune on the real fold. Val/test are NEVER generated or seen by the generator | fs_realmlp_fe (the FE the generator is fit over) | new generator-fit + sample in node_0119's own src/: per-class generator fitted on **train-fold rows only**, sampled rows feed pretraining only; verify no val/test row is ever generated/seen | **fit_in_fold** (the generator is a train-fold cross-row reference; sampled rows feed pretraining only, val/test never generated) | node_0119 | node_0119 |
| fs_zsoft | redshift-ERROR-aware re-expression of the z≈0 bottleneck (research.md L244): z_snr = z/3e-4 (z signal-to-noise vs the SDSS error floor), asinh(z/3e-4) (well-conditioned z-warp), log10(z+3e-4) (log z-warp), and a soft STAR-likelihood = a smooth membership score of the z≈0 stellar-noise band (logistic/Gaussian bump at z≈0, FIXED centre/width). Expands the crushed sub-0.01 stellar-noise↔smallest-galaxy interval into a wide stable margin; fed alongside fs_realmlp_fe to the n028 RealMLP recipe. NO label smoothing (split out to a later node) | base (redshift) | new stateless `add_zsoft_features` in node_0140's own src/; all four row-wise deterministic on raw z with a FIXED constant 3e-4 (NOT data-fit); soft STAR-likelihood uses a fixed centre/width | **stateless** (no target/fit/cross-row; fixed constants only) | node_0140 | node_0140 |
| fs_ambig | per-row AMBIGUITY auxiliary TARGET (NOT a feature view): a binary/soft label = "is this row in the entangled GAL↔STAR / GAL↔QSO confusion zone" derived from the champion-bank fold-honest OOF disagreement (pooled base argmax-vote entropy / top-2-class split) + the row's true class. Used as a SECOND head's target in a multi-task NN whose primary head is the 3-class task; shapes the shared trunk representation. The output written to oof/test is the 3-class prediction, never the ambiguity label | champion-bank OOF (nodes/node_0091|0070/oof.npy) + base train labels | new `add_ambig_target` in node_0139's own src/: ambiguity label computed from the **train-fold rows' fold-honest OOF + their true labels**, applied within that fold only; val/test rows are NEVER assigned a label from their own labels or a full-train refit | **fit_in_fold** (the ambiguity label is a train-fold cross-row + label-derived reference; must be rebuilt per fold from that fold's OOF, never full-train/test) | node_0139 | node_0139 |
