"""Render the PROPOSED node_0057 input images: rest-frame-warped SED -> GAF/RP spectrogram.
One card: 3 samples x 3 classes (GALAXY/QSO/STAR), picked at the 25/50/75th redshift
percentile within each class so the redshift-warp behaviour is visible. Shows the exact
features used + the SED signal + the GASF/GADF/RP channels the CNN would read."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

COMP = Path("/home/vaibhav/projects/upwork/ak/kaggleforge/comps/playground-series-s6e6")
BANDS = ["u", "g", "r", "i", "z"]
LAM_OBS = np.array([3543., 4770., 6231., 7625., 9134.])   # SDSS effective wavelengths (A)
CLASSES = ["GALAXY", "QSO", "STAR"]
NGRID = 24
GRID = np.exp(np.linspace(np.log(2500.), np.log(9200.), NGRID))   # common rest-frame log-lambda grid
CLASS_COLOR = {"GALAXY": "#d6604d", "QSO": "#4393c3", "STAR": "#f1a340"}

def build_signals(row):
    """Return (sed_obs_curve, s_rest_resampled, s_obs_resampled, lam_rest)."""
    mags = np.array([row[b] for b in BANDS], float)
    mag_mean = mags.mean()
    flux = 10.0 ** (-0.4 * (mags - mag_mean))            # per-row flux SHAPE (brightness removed)
    z = float(row["redshift"]); zc = max(z, -0.009)
    lam_rest = LAM_OBS / (1.0 + zc)                       # the rest-frame warp (the load-bearing trick)
    # PCHIP through anchors; edge-clamp outside support
    def resample(lam):
        order = np.argsort(lam)
        p = PchipInterpolator(lam[order], flux[order], extrapolate=False)
        out = p(GRID)
        if np.isnan(out).any():                          # edge-clamp extrapolated region
            lo, hi = flux[order][0], flux[order][-1]
            out[GRID < lam[order][0]] = lo
            out[GRID > lam[order][-1]] = hi
        return out
    return flux, resample(lam_rest), resample(LAM_OBS), lam_rest

def gaf_rp(s):
    """Hand-rolled GASF / GADF / Recurrence-Plot of a 1D signal (faithful to pyts)."""
    s = np.asarray(s, float)
    smin, smax = s.min(), s.max()
    sc = (s - smin) / (smax - smin + 1e-9) * 2 - 1       # rescale to [-1,1]
    sc = np.clip(sc, -1, 1)
    phi = np.arccos(sc)
    gasf = np.cos(phi[:, None] + phi[None, :])
    gadf = np.sin(phi[:, None] - phi[None, :])
    rp = np.abs(s[:, None] - s[None, :])                 # continuous recurrence
    rp = rp / (rp.max() + 1e-9)
    return gasf, gadf, rp

# ---- pick 3 representative rows per class (25/50/75th redshift pct) ----
tr = pd.read_csv(COMP / "data/train.csv")
for b in BANDS:
    tr = tr[(tr[b] > 5) & (tr[b] < 35)]                  # drop any placeholder/outlier mags
picks = []
rng = np.random.RandomState(0)
for c in CLASSES:
    sub = tr[tr["class"] == c]
    qs = sub["redshift"].quantile([0.25, 0.50, 0.75]).values
    for q in qs:
        idx = (sub["redshift"] - q).abs().idxmin()
        picks.append(sub.loc[idx])

# ---- render the card ----
nrows = len(picks)                                       # 9
fig = plt.figure(figsize=(15.5, 2.05 * nrows))
gs = GridSpec(nrows, 6, figure=fig, width_ratios=[1.35, 1.5, 1, 1, 1, 1.15],
              hspace=0.55, wspace=0.32, left=0.012, right=0.99, top=0.945, bottom=0.025)
fig.suptitle("Proposed node_0057 input — rest-frame-warped SED  →  GAF/RP 'spectrogram'  (3 samples × 3 classes)",
             fontsize=15, fontweight="bold", y=0.985)

for k, row in enumerate(picks):
    cls = row["class"]; col = CLASS_COLOR[cls]
    flux, s_rest, s_obs, lam_rest = build_signals(row)
    gasf, gadf, rp = gaf_rp(s_rest)
    comp = np.stack([(gasf + 1) / 2, (gadf + 1) / 2, rp], axis=-1)  # RGB composite

    # col0: features text
    axf = fig.add_subplot(gs[k, 0]); axf.axis("off")
    feat = (f"$\\bf{{{cls}}}$\n"
            f"u={row['u']:.2f}  g={row['g']:.2f}\n"
            f"r={row['r']:.2f}  i={row['i']:.2f}\n"
            f"z(band)={row['z']:.2f}\n"
            f"redshift={row['redshift']:.4f}\n"
            f"(1+z) warp ×{1/(1+max(row['redshift'],-0.009)):.3f}")
    axf.text(0.0, 0.5, feat, va="center", ha="left", fontsize=10.5,
             family="monospace", color="black",
             bbox=dict(boxstyle="round,pad=0.4", fc=col, alpha=0.18, ec=col, lw=1.6))

    # col1: SED signal (observed anchors + warped resampled curve)
    axs = fig.add_subplot(gs[k, 1])
    axs.plot(LAM_OBS, flux, "o-", color="gray", ms=5, lw=1.2, label="observed [u,g,r,i,z]")
    axs.plot(GRID, s_rest, "-", color=col, lw=2.0, label="rest-frame warped")
    axs.set_xlabel("wavelength Å", fontsize=8); axs.tick_params(labelsize=7)
    axs.set_title("SED signal (input)", fontsize=9)
    if k == 0: axs.legend(fontsize=6.5, loc="upper right")

    # col2-4: the three channels
    for j, (img, name, cmap) in enumerate([(gasf, "GASF", "RdBu_r"),
                                           (gadf, "GADF", "PuOr"),
                                           (rp, "RP", "viridis")]):
        ax = fig.add_subplot(gs[k, 2 + j])
        ax.imshow(img, cmap=cmap, vmin=(-1 if name != "RP" else 0), vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if k == 0: ax.set_title(name, fontsize=9)

    # col5: RGB composite = what the CNN "sees"
    axc = fig.add_subplot(gs[k, 5])
    axc.imshow(comp, interpolation="nearest")
    axc.set_xticks([]); axc.set_yticks([])
    for s in axc.spines.values(): s.set_color(col); s.set_linewidth(2.5)
    if k == 0: axc.set_title("RGB composite\n(R=GASF G=GADF B=RP)", fontsize=8.5)

out = COMP / "sed_image_card.png"
fig.savefig(out, dpi=115, facecolor="white")
print("WROTE", out)
print("\nFEATURES USED TO BUILD EACH IMAGE: u, g, r, i, z (5 bands) + redshift")
print("  -> flux-shape = 10^(-0.4*(mag - mag_mean_row))   [brightness removed]")
print("  -> rest-frame warp: lambda_rest = lambda_obs / (1+redshift)")
print("  -> PCHIP resample to 24 pts on common log-lambda grid -> GASF/GADF/RP 24x24")
print("\nSAMPLES (class | redshift):")
for row in picks:
    print(f"  {row['class']:7s} z={row['redshift']:+.4f}  u={row['u']:.2f} g={row['g']:.2f} r={row['r']:.2f} i={row['i']:.2f} z={row['z']:.2f}")
