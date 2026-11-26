"""Does the alignment offset track pose error better than AP and mIoU?

Success criteria, fixed BEFORE looking at results (change them only before a run):
  S1  For lateral, longitudinal and heading error, the pooled |Spearman rho| with the matching
      alignment offset has a 95% CI (bootstrap over scenes) whose lower bound is above the
      upper bounds of both the AP and the mIoU intervals.
  S2  Inside five equal-count score bins, the typical spread of pose error is smaller for the
      alignment offset than for AP and for mIoU.
      spread = (p90 + floor) / (p10 + floor) of |pose error| per bin; reported as the
      geometric mean over bins (1 = a perfect score, larger = worse).
Runs on the development scenes only. Held-out scenes are used once, at the end.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hdloc.metrics.stats import spearman_ci

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--files", nargs="*", default=["transfer_sweep", "map_ablations"])
args = ap.parse_args()

df = pd.concat([pd.read_csv(REPO / "results" / f"{f}.csv") for f in args.files], ignore_index=True)
assert "align_lat_med" in df.columns, "re-run the sweeps: these CSVs have no alignment columns"

# comparable population: moving scenes, no odometry sweeps, no range cut (AP is 1 by
# construction there), no empty map (nothing to align)
df = df[(df.path_len > 10.0) & ~df.kind.str.startswith("drift_") & (df.kind != "range_m")].copy()
df = df[df.align_lat_med.notna()].copy()
ident = {"lateral_shift": 0, "longitudinal_shift": 0, "rotation_deg": 0, "jitter_sigma": 0,
         "element_dropout": 0, "drop_transverse": 0, "false_dividers": 0, "false_stop_lines": 0,
         "drop_dividers": 0}
is_id = np.array([ident.get(k) == v for k, v in zip(df.kind, df.value)])
base = df[is_id].groupby(["scene", "seed"]).pf_lon_med.median().rename("lon_base")
df = df.join(base, on=["scene", "seed"])
df["lon_excess"] = df.pf_lon_med - df.lon_base
df = df[df.lon_base.notna()].copy()
print(f"{len(df)} runs, {df.scene.nunique()} scenes, {df.kind.nunique()} degradation families")

COMPONENTS = (  # name, alignment column, pose-error column, floor for the spread ratio
    ("lateral", "align_lat_med", "pf_lat_med", 0.02),
    ("longitudinal", "align_lon_med", "lon_excess", 0.02),
    ("heading", "align_yaw_med_deg", "pf_yaw_med_deg", 0.10),
)


def abs_ci(x, y, groups):
    rho, lo, hi = spearman_ci(x, y, groups=groups)
    return (abs(rho), -hi, -lo) if rho < 0 else (rho, lo, hi)


def spread(score, err, floor, bins=5):
    b = pd.qcut(score.rank(method="first"), bins, labels=False)
    ratios = []
    for _, g in err.abs().groupby(b):
        p10, p90 = g.quantile(0.1), g.quantile(0.9)
        ratios.append(np.log((p90 + floor) / (p10 + floor)))
    return float(np.exp(np.mean(ratios)))


print("\nS1: pooled |Spearman rho| with 95% CI (bootstrap over scenes)")
print(f"{'error':13s}{'alignment':>22s}{'AP':>22s}{'mIoU':>22s}   verdict")
s1 = {}
for name, acol, ecol, _ in COMPONENTS:
    res = {s: abs_ci(df[col], df[ecol], df.scene)
           for s, col in (("align", acol), ("ap", "map_ap"), ("miou", "map_miou"))}
    fmt = {k: f"{v[0]:.2f} [{v[1]:.2f},{v[2]:.2f}]" for k, v in res.items()}
    s1[name] = res["align"][1] > max(res["ap"][2], res["miou"][2])
    print(f"{name:13s}{fmt['align']:>22s}{fmt['ap']:>22s}{fmt['miou']:>22s}   "
          f"{'PASS' if s1[name] else 'FAIL'}")

print("\nS2: spread of pose error inside equal-count score bins (1 = perfect, lower is better)")
print(f"{'error':13s}{'alignment':>12s}{'AP':>10s}{'mIoU':>10s}   verdict")
for name, acol, ecol, floor in COMPONENTS:
    sp = {s: spread(df[col], df[ecol], floor)
          for s, col in (("align", acol), ("ap", "map_ap"), ("miou", "map_miou"))}
    ok = sp["align"] < min(sp["ap"], sp["miou"])
    print(f"{name:13s}{sp['align']:12.1f}{sp['ap']:10.1f}{sp['miou']:10.1f}   "
          f"{'PASS' if ok else 'FAIL'}")

print("\nWhere the offset misses: median (pose error - alignment offset) per family")
miss = pd.DataFrame({
    name: (df[ecol] - df[acol]).groupby(df.kind).median() for name, acol, ecol, _ in COMPONENTS
})
print(miss.round(2).to_string())

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for a, (name, acol, ecol, _) in zip(ax, COMPONENTS):
    for kind, g in df.groupby("kind"):
        a.scatter(g[acol], g[ecol], s=12, alpha=0.55, label=kind)
    lim = max(df[acol].max(), df[ecol].max()) * 1.05
    a.plot([0, lim], [0, lim], "k--", lw=0.8)
    a.set_xlabel(f"alignment offset ({name})")
    a.set_ylabel(f"pose error ({name})")
    a.set_title(name)
ax[0].legend(fontsize=7)
fig.tight_layout()
out = REPO / "results" / "figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "align_vs_error.png", dpi=130)
print("saved", out / "align_vs_error.png")
