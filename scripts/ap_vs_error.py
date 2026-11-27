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

frames = []
for f in args.files:
    p = REPO / "results" / f"{f}.csv"
    if p.exists():
        frames.append(pd.read_csv(p))
    else:
        print("missing", p)
df = pd.concat(frames, ignore_index=True)
df = df.drop_duplicates(["scene", "kind", "value", "seed"]).reset_index(drop=True)
assert "map_ap" in df.columns, "re-run the sweeps: these CSVs have no AP columns yet"
df = df[(df.path_len > 10.0) & ~df.kind.str.startswith("drift_")].copy()

# longitudinal error beyond what the same scene, seed and drift give with an undamaged map
identity = {"lateral_shift": 0, "longitudinal_shift": 0, "rotation_deg": 0, "jitter_sigma": 0,
            "element_dropout": 0, "drop_transverse": 0, "false_dividers": 0,
            "false_stop_lines": 0, "drop_dividers": 0, "drop_all": 0, "range_m": 50}
is_id = np.array([identity.get(k) == v for k, v in zip(df.kind, df.value)])
base = df[is_id].groupby(["scene", "seed"]).pf_lon_bias.median().rename("lon_base")
df = df.join(base, on=["scene", "seed"])
df["lon_excess"] = (df.pf_lon_bias - df.lon_base).abs()
print(f"{len(df)} runs, {df.scene.nunique()} scenes, {df.kind.nunique()} degradation families")

cols = ["map_ap", "map_miou", "pf_lat_med", "lon_excess"]
print("\nDamage ladder (median over scenes and seeds):")
print(df.groupby(["kind", "value"])[cols].median().round(2).to_string())

print("\nPooled Spearman over all families and sizes (95% CI, bootstrap over scenes):")
for score in ("map_ap", "map_miou"):
    for err in ("pf_lat_med", "lon_excess", "pf_yaw_med_deg"):
        rho, lo, hi = spearman_ci(df[score], df[err], groups=df.scene)
        print(f"  {score:9s} vs {err:15s} rho {rho:+.2f}  [{lo:+.2f}, {hi:+.2f}]")

df["ap_bin"] = pd.cut(df.map_ap, [-0.01, 0.4, 0.7, 0.9, 0.99, 1.01])
spread = df.groupby("ap_bin", observed=True).agg(
    n=("kind", "size"), families=("kind", "nunique"),
    lat_min=("pf_lat_med", "min"), lat_med=("pf_lat_med", "median"), lat_max=("pf_lat_med", "max"),
    lon_min=("lon_excess", "min"), lon_med=("lon_excess", "median"), lon_max=("lon_excess", "max"),
)
print("\nSpread of pose error at similar AP:")
print(spread.round(2).to_string())

fig, ax = plt.subplots(2, 3, figsize=(17, 9), sharex="row")
for r, score in enumerate(("map_ap", "map_miou")):
    for a, (err, title) in zip(ax[r], (("pf_lat_med", "lateral error (m)"),
                                       ("lon_excess", "longitudinal error beyond baseline (m)"),
                                       ("pf_yaw_med_deg", "heading error (deg)"))):
        for kind, g in df.groupby("kind"):
            a.scatter(g[score], g[err], s=12, alpha=0.55, label=kind)
        a.set_xlabel(score)
        a.set_ylabel(title)
ax[0][0].legend(fontsize=7)
fig.tight_layout()
out = REPO / "results" / "figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "ap_vs_error.png", dpi=130)
print("saved", out / "ap_vs_error.png")
