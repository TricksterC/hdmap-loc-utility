"""Rebuild every chapter figure from the CSVs in results/. Run: python3 scripts/make_figures.py

dev = results/transfer_sweep.csv (the 8 moving mini scenes).
held-out = results/heldout_sweep.csv (48 scenes, list fixed in configs/heldout_scenes.yaml).
Figures go to figures/ (committed). This script only draws; it changes no number.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from hdloc.metrics.calibration import loso_mae

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures"
SETS = {"dev": "transfer_sweep", "held-out": "heldout_sweep"}
IDENTITY = {"lateral_shift": 0, "longitudinal_shift": 0, "rotation_deg": 0, "jitter_sigma": 0,
            "element_dropout": 0, "drop_transverse": 0, "false_dividers": 0,
            "false_stop_lines": 0, "drop_dividers": 0, "drop_all": 0, "range_m": 50}
COHERENT = ["lateral_shift", "longitudinal_shift", "rotation_deg"]
GROUP_OF = {
    **{k: "shift or rotation" for k in COHERENT},
    **{k: "elements missing" for k in
       ("element_dropout", "drop_dividers", "drop_transverse", "drop_all", "range_m")},
        **{k: "extra or noisy elements" for k in
       ("jitter_sigma", "false_dividers", "false_stop_lines")},
}
GROUPS = {"shift or rotation": "#D55E00", "elements missing": "#0072B2",
          "extra or noisy elements": "#009E73"}
METHODS = {"predict 0": "#bbbbbb", "AP": "#E69F00", "mIoU": "#56B4E9",
           "alignment": "#009E73", "alignment + c": "#005f45"}

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def load(stem):
    df = pd.read_csv(REPO / "results" / f"{stem}.csv")
    df = df.drop_duplicates(["scene", "kind", "value", "seed"])
    df = df[df.path_len > 10.0].copy()
    is_id = np.array([IDENTITY.get(k) == v for k, v in zip(df.kind, df.value)])
    base = df[is_id].groupby(["scene", "seed"]).pf_lon_med.median().rename("lon_base")
    df = df.join(base, on=["scene", "seed"])
    df["lon_excess"] = df.pf_lon_med - df.lon_base
    return df


def styles(data):
    n = {k: v.scene.nunique() for k, v in data.items()}
    return {
        "dev": dict(color="#7f7f7f", ls="--", marker="s", label=f"dev ({n['dev']} scenes)"),
        "held-out": dict(color="#0072B2", ls="-", marker="o",
                         label=f"held-out ({n['held-out']} scenes)"),
    }


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)
    print("saved", OUT / name)


def fig_coherent(data, sty):
    panels = [
        ("lateral_shift", "pf_lat_med", "map shifted sideways (m)", "sideways error (m)", True),
        ("longitudinal_shift", "lon_excess", "map shifted forward (m)",
         "forward error beyond baseline (m)", True),
        ("rotation_deg", "pf_yaw_med_deg", "map rotated (deg)", "heading error (deg)", True),
        ("rotation_deg", "pf_lat_med", "map rotated (deg)", "sideways error (m)", False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (kind, ycol, xl, yl, diag) in zip(axes, panels):
        for name, df in data.items():
            g = df[df.kind == kind].groupby("value")[ycol]
            med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
            s = sty[name]
            ax.plot(med.index, med.values, color=s["color"], ls=s["ls"], marker=s["marker"],
                    ms=4, label=s["label"])
            ax.fill_between(med.index, lo.values, hi.values, color=s["color"], alpha=0.12, lw=0)
        if diag:
            top = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([0, top], [0, top], color="k", lw=0.7, ls=":")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "fig_transfer_coherent.png")


def fig_random(data, sty):
    fams = [("element_dropout", "fraction of elements dropped"),
            ("false_dividers", "false dividers added"),
            ("false_stop_lines", "false stop lines added"),
            ("jitter_sigma", "jitter sigma (m)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, (kind, xl) in zip(axes, fams):
        for name, df in data.items():
            for ycol, color in (("pf_lat_med", "#0072B2"), ("lon_excess", "#E69F00")):
                g = df[df.kind == kind].groupby("value")[ycol].median()
                ax.plot(g.index, g.values, color=color, ls=sty[name]["ls"],
                        marker=sty[name]["marker"], ms=4)
        ax.set_xlabel(xl)
        ax.set_ylim(0, 2)  # same scale as the shift panels, so "barely matters" is visible
    axes[0].set_ylabel("median error (m)")
    handles = [Line2D([], [], color="#0072B2", label="sideways"),
               Line2D([], [], color="#E69F00", label="forward (beyond baseline)"),
               Line2D([], [], color="k", ls="--", label="dev"),
               Line2D([], [], color="k", ls="-", label="held-out")]
    axes[0].legend(handles=handles, frameon=False, fontsize=8)
    save(fig, "fig_transfer_random.png")


def fig_classes(data, sty):
    conds = [("none", "drop_all", 0), ("no dividers", "drop_dividers", 1),
             ("no crosswalks\nor stop lines", "drop_transverse", 1), ("no elements", "drop_all", 1)]
    titles = {"lat": "median sideways error (m)", "lon": "median forward error (m)",
              "div": "runs that got lost (%)"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    w = 0.38
    for j, (name, df) in enumerate(data.items()):
        vals = {"lat": [], "lon": [], "div": []}
        for _, k, v in conds:
            g = df[(df.kind == k) & (df.value == v)]
            vals["lat"].append(g.pf_lat_med.median())
            vals["lon"].append(g.pf_lon_med.median())
            vals["div"].append(100 * g.pf_diverged.mean())
        x = np.arange(len(conds)) + (j - 0.5) * w
        for ax, key in zip(axes, ("lat", "lon", "div")):
            ax.bar(x, vals[key], w, color=sty[name]["color"], label=sty[name]["label"])
    for ax, key in zip(axes, ("lat", "lon", "div")):
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([c[0] for c in conds], fontsize=8)
        ax.set_title(titles[key], fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "fig_class_ablation.png")


def fig_ap(df):
    d = df[~df.kind.str.startswith("drift_")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, col, lab in zip(axes, ("map_ap", "map_miou"), ("AP", "mIoU")):
        for grp, color in GROUPS.items():
            g = d[d.kind.map(GROUP_OF) == grp]
            ax.scatter(g[col], g.pf_lat_med.clip(lower=0.005), s=8, alpha=0.4, color=color,
                       label=grp, lw=0)
        rho = spearmanr(d[col], d.pf_lat_med)[0]
        ax.set_yscale("log")
        ax.set_xlabel(lab)
        ax.set_title(f"{lab} vs sideways error (Spearman {rho:.2f})", fontsize=10)
    axes[0].set_ylabel("sideways error (m, log scale)")
    axes[0].legend(frameon=False, fontsize=8, markerscale=2)
    save(fig, "fig_ap_vs_error.png")


def fig_align(df):
    d = df[~df.kind.str.startswith("drift_") & (df.kind != "range_m") & df.align_lat_med.notna()]
    comps = [("align_lat_med", "pf_lat_med", "sideways (m)", 0.01),
             ("align_lon_med", "lon_excess", "forward, beyond baseline (m)", 0.01),
             ("align_yaw_med_deg", "pf_yaw_med_deg", "heading (deg)", 0.05)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (a, e, lab, floor) in zip(axes, comps):
        x, y = d[a].abs().clip(lower=floor), d[e].abs().clip(lower=floor)
        for grp, color in GROUPS.items():
            m = d.kind.map(GROUP_OF) == grp
            ax.scatter(x[m], y[m], s=8, alpha=0.4, color=color, label=grp, lw=0)
        lim = [floor, max(x.max(), y.max()) * 1.2]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"alignment offset, {lab}")
        ax.set_ylabel(f"pose error, {lab}")
    axes[0].legend(frameon=False, fontsize=8, markerscale=2)
    save(fig, "fig_alignment_vs_error.png")


def fig_calibration(data):
    comps = [("align_lat_med", "pf_lat_med", "sideways error (m)"),
             ("align_lon_med", "lon_excess", "forward error (m)"),
             ("align_yaw_med_deg", "pf_yaw_med_deg", "heading error (deg)")]
    fig, axes = plt.subplots(len(data), 3, figsize=(13, 3.4 * len(data)), squeeze=False)
    for r, (name, df) in enumerate(data.items()):
        d = df[df.kind.isin(COHERENT) & df.align_lat_med.notna()]
        for c, (a, e, lab) in enumerate(comps):
            res = loso_mae(d, a, e)
            ax = axes[r][c]
            ax.bar(range(len(METHODS)), [res[m] for m in METHODS], color=list(METHODS.values()))
            for i, m in enumerate(METHODS):
                ax.text(i, res[m], f"{res[m]:.2f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(range(len(METHODS)))
            ax.set_xticklabels(list(METHODS), rotation=30, ha="right", fontsize=8)
            ax.set_title(f"{name}: {lab}", fontsize=10)
            ax.set_ylabel("mean abs. prediction error")
    save(fig, "fig_calibration.png")


def main():
    data = {k: load(v) for k, v in SETS.items()}
    sty = styles(data)
    fig_coherent(data, sty)
    fig_random(data, sty)
    fig_classes(data, sty)
    fig_ap(data["held-out"])
    fig_align(data["held-out"])
    fig_calibration(data)


if __name__ == "__main__":
    main()
