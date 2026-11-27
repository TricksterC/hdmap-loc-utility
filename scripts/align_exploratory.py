"""EXPLORATORY follow-up to align_vs_error.py. Written AFTER seeing the S1/S2 results, so
nothing here replaces them; report S1/S2 as pre-registered and this as post hoc.

  A  Rank correlation split by damage type. Coherent damage (shifts, rotation) moves pose by
     metres; non-rigid damage (dropout, false elements, jitter, dropped classes) by centimetres.
  B  Error prediction in metres, leave-one-scene-out. AP and mIoU get a fitted monotone curve
     (score -> error) trained on the other scenes; alignment gets NO fit (prediction = offset).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hdloc.metrics.stats import spearman_ci

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--files", nargs="*", default=["transfer_sweep", "map_ablations"])
args = ap.parse_args()

df = pd.concat([pd.read_csv(REPO / "results" / f"{f}.csv") for f in args.files], ignore_index=True)
df = df.drop_duplicates(["scene", "kind", "value", "seed"]).reset_index(drop=True)
df = df[(df.path_len > 10.0) & ~df.kind.str.startswith("drift_") & (df.kind != "range_m")].copy()
df = df[df.align_lat_med.notna()].copy()
ident = {"lateral_shift": 0, "longitudinal_shift": 0, "rotation_deg": 0, "jitter_sigma": 0,
         "element_dropout": 0, "drop_transverse": 0, "false_dividers": 0, "false_stop_lines": 0,
         "drop_dividers": 0}
is_id = np.array([ident.get(k) == v for k, v in zip(df.kind, df.value)])
base = df[is_id].groupby(["scene", "seed"]).pf_lon_bias.median().rename("lon_base")
df = df.join(base, on=["scene", "seed"])
df["lon_excess"] = (df.pf_lon_bias - df.lon_base).abs()
df = df[df.lon_base.notna()].copy()

COHERENT = ["lateral_shift", "longitudinal_shift", "rotation_deg"]
df["group"] = np.where(df.kind.isin(COHERENT), "coherent", "non-rigid")
COMPONENTS = (
    ("lateral", "align_lat_med", "pf_lat_med"),
    ("longitudinal", "align_lon_med", "lon_excess"),
    ("heading", "align_yaw_med_deg", "pf_yaw_med_deg"),
)
SCORES = (("alignment", None), ("AP", "map_ap"), ("mIoU", "map_miou"))


def rho(x, y, groups):
    r, lo, hi = spearman_ci(x, y, groups=groups)
    return (abs(r), -hi, -lo) if r < 0 else (r, lo, hi)


print("A. |Spearman rho| [95% CI over scenes], by damage group")
print(f"{'group':10s}{'error':13s}{'n':>5s}{'err p10-p90':>14s}"
      f"{'alignment':>22s}{'AP':>22s}{'mIoU':>22s}")
for grp, g in df.groupby("group"):
    for name, acol, ecol in COMPONENTS:
        e = g[ecol].abs()
        cells = []
        for lab, col in SCORES:
            r = rho(g[acol if col is None else col], g[ecol], g.scene)
            cells.append(f"{r[0]:.2f} [{r[1]:.2f},{r[2]:.2f}]")
        rng = f"{e.quantile(.1):.2f}-{e.quantile(.9):.2f}"
        print(f"{grp:10s}{name:13s}{len(g):5d}{rng:>14s}"
              f"{cells[0]:>22s}{cells[1]:>22s}{cells[2]:>22s}")

print("\nA2. Coherent families one at a time: |Spearman rho| [95% CI over scenes]")
print(f"{'family':20s}{'error':13s}{'alignment':>22s}{'AP':>22s}{'mIoU':>22s}")
for fam in COHERENT:
    g = df[df.kind == fam]
    for name, acol, ecol in COMPONENTS:
        cells = []
        for lab, col in SCORES:
            r = rho(g[acol if col is None else col], g[ecol], g.scene)
            cells.append(f"{r[0]:.2f} [{r[1]:.2f},{r[2]:.2f}]")
        print(f"{fam:20s}{name:13s}{cells[0]:>22s}{cells[1]:>22s}{cells[2]:>22s}")


def pav(x, y):
    """Isotonic (non-decreasing) fit of y on x; returns a predict function."""
    o = np.argsort(x)
    xs, ys = np.asarray(x)[o], np.asarray(y, float)[o]
    blocks = [[v, 1.0] for v in ys]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, w2 = out.pop()
            v1, w1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    fit = np.concatenate([[v] * int(w) for v, w in out])
    return lambda q: np.interp(q, xs, fit)


print("\nB. Leave-one-scene-out mean absolute error of predicting |pose error| (metres / deg)")
print("   alignment: prediction = offset itself (no fitting).")
print("   align+c: offset plus one fitted constant (the error floor), fit on the other scenes.")
print("   AP, mIoU: isotonic fit on the other scenes.")
print(f"{'group':10s}{'error':13s}{'predict 0':>11s}{'alignment':>11s}{'align+c':>9s}"
      f"{'AP':>9s}{'mIoU':>9s}")
for grp, g in list(df.groupby("group")) + [("all", df)]:
    for name, acol, ecol in COMPONENTS:
        y = g[ecol].abs().to_numpy()
        pred = {"zero": np.zeros(len(g)), "align": g[acol].abs().to_numpy(),
                "AP": np.zeros(len(g)), "mIoU": np.zeros(len(g)), "align+c": np.zeros(len(g))}
        for s in g.scene.unique():
            te, tr = (g.scene == s).to_numpy(), (g.scene != s).to_numpy()
            off = g[acol].abs().to_numpy()
            pred["align+c"][te] = off[te] + max(float(np.median(y[tr] - off[tr])), 0.0)
            for lab, col in (("AP", "map_ap"), ("mIoU", "map_miou")):
                f = pav(-g.loc[tr, col].to_numpy(), y[tr])
                pred[lab][te] = f(-g.loc[te, col].to_numpy())
        mae = {k: float(np.mean(np.abs(v - y))) for k, v in pred.items()}
        print(f"{grp:10s}{name:13s}{mae['zero']:11.3f}{mae['align']:11.3f}{mae['align+c']:9.3f}"
              f"{mae['AP']:9.3f}{mae['mIoU']:9.3f}")
