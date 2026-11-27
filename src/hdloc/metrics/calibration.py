"""Predicting pose error in metres from a map score, leave one scene out."""

from __future__ import annotations

import numpy as np


def pav(x, y):
    """Non-decreasing least-squares fit of y on x. Returns a function of new x."""
    order = np.argsort(x, kind="stable")
    xs = np.asarray(x, float)[order]
    ys = np.asarray(y, float)[order]
    vals: list[float] = []
    wts: list[float] = []
    for v in ys:
        vals.append(float(v))
        wts.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2 = vals.pop(), wts.pop()
            v1, w1 = vals.pop(), wts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
    fit = np.repeat(vals, np.asarray(wts).astype(int))
    return lambda q: np.interp(q, xs, fit)


def loso_mae(g, acol: str, ecol: str) -> dict[str, float]:
    """Mean absolute error of predicting |pose error| `ecol`, leaving one scene out.

    alignment: the offset itself, nothing fitted. alignment + c: offset plus one constant (the
    error floor) fitted on the other scenes. AP, mIoU: monotone curve fitted on other scenes.
    """
    y = g[ecol].abs().to_numpy(float)
    off = g[acol].abs().to_numpy(float)
    scenes = g["scene"].to_numpy()
    ap, iou = -g["map_ap"].to_numpy(float), -g["map_miou"].to_numpy(float)
    pred = {k: np.zeros(len(g)) for k in ("alignment + c", "AP", "mIoU")}
    for s in np.unique(scenes):
        te = scenes == s
        tr = ~te
        pred["alignment + c"][te] = off[te] + max(float(np.median(y[tr] - off[tr])), 0.0)
        pred["AP"][te] = pav(ap[tr], y[tr])(ap[te])
        pred["mIoU"][te] = pav(iou[tr], y[tr])(iou[te])
    out = {"predict 0": float(y.mean()), "alignment": float(np.abs(off - y).mean())}
    out.update({k: float(np.abs(v - y).mean()) for k, v in pred.items()})
    return out
