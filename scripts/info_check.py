"""POST HOC (written after seeing the S1/S2 results): does the alignment's own constraint
strength flag the runs where its offset reads ~0 but the filter is far off?

Population: moving scenes, no drift sweeps, no range cut, no rotation (its sideways error is a
heading lag, not missing constraint), runs whose sideways alignment offset is below OFFSET_OK.
"bad" = sideways pose error above BAD_M.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

OFFSET_OK = 0.05
BAD_M = 0.5
FLAG_PCT = 5

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--files", nargs="*", default=["heldout_sweep"])
ap.add_argument("--boot", type=int, default=300)
args = ap.parse_args()

df = pd.concat([pd.read_csv(REPO / "results" / f"{f}.csv") for f in args.files], ignore_index=True)
df = df.drop_duplicates(["scene", "kind", "value", "seed"])
df = df[(df.path_len > 10) & df.align_lat_med.notna() & ~df.kind.str.startswith("drift_")
        & (df.kind != "range_m") & (df.kind != "rotation_deg")]
m = df[df.align_lat_med < OFFSET_OK].reset_index(drop=True)
bad = m.pf_lat_med > BAD_M
print(f"runs with offset < {OFFSET_OK} m: {len(m)}; sideways error > {BAD_M} m: {int(bad.sum())}")
print(m.groupby(bad)[["align_info_lat", "align_info_lon"]].median().round(1).to_string())


def auc(x):
    b = x.pf_lat_med > BAD_M
    if b.sum() == 0 or (~b).sum() == 0:
        return np.nan
    u = mannwhitneyu(x.align_info_lat[b], x.align_info_lat[~b]).statistic
    return 1 - u / (b.sum() * (~b).sum())


print(f"AUC (bad run has lower constraint than a good one): {auc(m):.2f}")
rng = np.random.default_rng(0)
by = {s: g for s, g in m.groupby("scene")}
names = list(by)
vals = [auc(pd.concat([by[s] for s in rng.choice(names, len(names))])) for _ in range(args.boot)]
lo, mid, hi = np.nanpercentile(vals, [2.5, 50, 97.5])
print(f"bootstrap over scenes: {lo:.2f} / {mid:.2f} / {hi:.2f}")
flag = m.align_info_lat < np.percentile(m.align_info_lat, FLAG_PCT)
print(f"flagging the lowest {FLAG_PCT}%: flagged {int(flag.sum())}, "
      f"caught {int((flag & bad).sum())} of {int(bad.sum())}")
print("kinds of the bad runs:")
print(m[bad].kind.value_counts().to_string())
