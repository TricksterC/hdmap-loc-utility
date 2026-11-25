import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from hdloc.experiments.common import evaluate, load_scenes
from hdloc.localization.motion_model import DriftModel
from hdloc.localization.particle_filter import FilterConfig
from hdloc.utils.io import load_yaml

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "nuscenes" / "v1.0-mini"

ap = argparse.ArgumentParser()
ap.add_argument("--kinds", nargs="*", help="subset of sweeps to run")
ap.add_argument("--scenes", nargs="*", type=int, help="scene indices (default: all)")
ap.add_argument("--seeds", type=int, help="use only the first N seeds")
ap.add_argument("--out", default="transfer_sweep", help="output file stem in results/")
ap.add_argument("--scale-error", type=float, help="override the config odometry scale error")
args = ap.parse_args()

cfg = load_yaml(REPO / "configs" / "degradations.yaml")
fcfg = FilterConfig.from_dict(load_yaml(REPO / "configs" / "filter.yaml"))
drift_cfg = dict(cfg["drift"])
if args.scale_error is not None:
    drift_cfg["scale_error"] = args.scale_error
drift = DriftModel(**drift_cfg)
seeds = cfg["seeds"][: args.seeds] if args.seeds else cfg["seeds"]
kinds = args.kinds or list(cfg["sweeps"])

rows = []
for scene in load_scenes(ROOT, args.scenes):
    for kind in kinds:
        for value in cfg["sweeps"][kind]:
            for seed in seeds:
                res = evaluate(scene, kind, value, seed, fcfg, drift, cfg["radius"])
                rows.append(
                    {
                        "scene": scene.name,
                        "map": scene.map_name,
                        "path_len": scene.path_len,
                        "kind": kind,
                        "value": value,
                        "seed": seed,
                        **res,
                    }
                )
    print("finished", scene.name, flush=True)

df = pd.DataFrame(rows)
(REPO / "results").mkdir(exist_ok=True)
df.to_csv(REPO / "results" / f"{args.out}.csv", index=False)

moving = df[df.path_len > 10.0]  # leave (near-)stationary scenes out of the summary
cols = ["pf_lon_med", "pf_lat_med", "odo_lon_med", "odo_lat_med", "transverse_frac"]
print(moving.groupby(["kind", "value"])[cols].median().round(2).to_string())

fig, axes = plt.subplots(3, 5, figsize=(20, 10))
for ax, kind in zip(axes.ravel(), kinds):
    g = moving[moving.kind == kind].groupby("value")
    for col, fmt, label in (
        ("pf_lon_med", "o-", "filter |lon|"),
        ("pf_lat_med", "s-", "filter |lat|"),
        ("odo_lon_med", "k--", "odometry |lon|"),
    ):
        s = g[col].median()
        ax.plot(s.index.values, s.values, fmt, label=label)
    ax.set_title(kind)
    ax.set_ylabel("median error (m)")
axes.ravel()[0].legend()
for ax in axes.ravel()[len(kinds) :]:
    ax.axis("off")
out = REPO / "results" / "figures"
out.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out / f"{args.out}.png", dpi=130)
print("saved", out / f"{args.out}.png")
