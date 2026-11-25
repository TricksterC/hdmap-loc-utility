import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hdloc.experiments.class_observability import ABLATIONS, coverage_table, run_ablation
from hdloc.experiments.common import load_scenes
from hdloc.localization.particle_filter import FilterConfig
from hdloc.utils.io import load_yaml

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "nuscenes" / "v1.0-mini"
FIG = REPO / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--scales", type=float, nargs="*", default=[0.01, 0.03, 0.06])
ap.add_argument("--scenes", type=int, nargs="*")
ap.add_argument("--coverage-only", action="store_true")
args = ap.parse_args()

cfg = load_yaml(REPO / "configs" / "degradations.yaml")
fcfg = FilterConfig.from_dict(load_yaml(REPO / "configs" / "filter.yaml"))
scenes = [s for s in load_scenes(ROOT, args.scenes) if s.path_len > 10.0]
print(f"{len(scenes)} scenes with real motion")

# ---------------------------------------------------------------- route coverage (RQ2)
cov = coverage_table(scenes)
cov.to_csv(REPO / "results" / "class_coverage.csv", index=False)
show = ["divider_present", "ped_crossing_present", "stop_line_present", "transverse_present",
        "lon_all_ge_5m", "lon_divider_ge_5m", "lon_transverse_ge_5m"]
by_r = cov.groupby("radius")[show]
print("\nShare of frames (mean over scenes):")
print(by_r.mean().round(2).T.to_string())
print("\nShare of frames (worst scene):")
print(by_r.min().round(2).T.to_string())

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
panels = (
    (["divider_present", "ped_crossing_present", "stop_line_present", "transverse_present"],
     ["divider", "ped_crossing", "stop_line", "any transverse"], "class in range"),
    (["lon_all_ge_5m", "lon_divider_ge_5m", "lon_transverse_ge_5m"],
     ["all classes", "dividers only", "transverse only"], "at least 5 m of cross-road line"),
)
for a, (cols, labels, title) in zip(ax, panels):
    for j, r in enumerate((30.0, 50.0)):
        sub = cov[cov.radius == r]
        xs = np.arange(len(cols)) + (j - 0.5) * 0.38
        a.bar(xs, sub[cols].mean().values, 0.38, label=f"{r:.0f} m (mean)")
        a.plot(xs, sub[cols].min().values, "k_", markersize=12,
               label="worst scene" if j == 0 else None)
    a.set_xticks(range(len(cols)))
    a.set_xticklabels(labels, rotation=15)
    a.set_ylim(0, 1.05)
    a.set_ylabel("share of frames along the route")
    a.set_title(title)
ax[0].legend()
fig.tight_layout()
fig.savefig(FIG / "class_coverage.png", dpi=130)
print("saved", FIG / "class_coverage.png")

if args.coverage_only:
    raise SystemExit

# ---------------------------------------------------------------- oracle class ablation
df = run_ablation(scenes, fcfg, cfg["drift"], args.scales, list(range(args.seeds)), cfg["radius"])
key = ["scene", "scale_error", "seed"]
base = df[df.condition == "all"][key + ["pf_lon_med"]].rename(columns={"pf_lon_med": "lon_all"})
df = df.merge(base, on=key)
df["lon_vs_all"] = df.pf_lon_med / df.lon_all
df["lon_ratio"] = df.pf_lon_med / df.odo_lon_med
df["condition"] = pd.Categorical(df["condition"], [a[0] for a in ABLATIONS], ordered=True)
df.to_csv(REPO / "results" / "class_ablation.csv", index=False)

cols = ["pf_lon_med", "pf_lat_med", "pf_yaw_med_deg", "lon_ratio", "lon_vs_all"]
print("\nMedians over moving scenes and seeds")
print("lon_ratio = filter / odometry-only (0 perfect, 1 no better than dead reckoning)")
print("lon_vs_all = filter / all-class oracle on the same scene, drift and seed")
print(df.groupby(["scale_error", "condition"], observed=True)[cols].median().round(2).to_string())

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for a, (col, title, log) in zip(ax, (
    ("lon_ratio", "longitudinal error / odometry-only error", False),
    ("lon_vs_all", "longitudinal error / all-class oracle (log)", True),
    ("pf_lat_med", "median lateral error, m (log)", True),
)):
    piv = df.groupby(["condition", "scale_error"], 
        observed=True)[col].median().unstack("scale_error")

    piv.plot.bar(ax=a, rot=25, logy=log)
    a.set_title(title)
    a.set_xlabel("")
    a.legend(title="odometry scale error")
fig.tight_layout()
fig.savefig(FIG / "class_ablation.png", dpi=130)
print("saved", FIG / "class_ablation.png")
