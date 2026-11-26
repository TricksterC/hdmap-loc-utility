from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from hdloc.data.schema import CLASSES, TRANSVERSE, VectorMap
from hdloc.maps.observe import resample_polyline
from hdloc.maps.rasterize import Grid, rasterize_class

THRESHOLDS = (0.5, 1.0, 1.5)  # metres, Chamfer distance


def _prep(vmap: VectorMap, cls: str, spacing: float) -> list:
    out = []
    for e in vmap.of_class(cls):
        pts = resample_polyline(e.points, spacing)
        if len(pts):
            out.append((pts, cKDTree(pts), pts.min(axis=0), pts.max(axis=0)))
    return out


def _chamfer(a, b) -> float:
    """Symmetric mean nearest-point distance between two sampled polylines."""
    return 0.5 * (b[1].query(a[0])[0].mean() + a[1].query(b[0])[0].mean())


def _near(a, b, margin: float) -> bool:
    return bool(np.all(a[2] <= b[3] + margin) and np.all(b[2] <= a[3] + margin))


def match_counts(truth: VectorMap, pred: VectorMap, thresholds=THRESHOLDS, spacing=0.25) -> dict:
    """One frame: {class: {threshold: [true positives, n_pred, n_truth]}}.

    Elements are matched one-to-one, smallest Chamfer distance first.
    """
    tmax = max(thresholds)
    out = {}
    for cls in CLASSES:
        g, p = _prep(truth, cls, spacing), _prep(pred, cls, spacing)
        if not g and not p:
            continue
        pairs = []
        for i, pe in enumerate(p):
            for j, ge in enumerate(g):
                if _near(pe, ge, tmax):
                    d = _chamfer(pe, ge)
                    if d < tmax:
                        pairs.append((d, i, j))
        pairs.sort()
        out[cls] = {}
        for t in thresholds:
            used_p, used_g, tp = set(), set(), 0
            for d, i, j in pairs:
                if d >= t:
                    break
                if i in used_p or j in used_g:
                    continue
                used_p.add(i)
                used_g.add(j)
                tp += 1
            out[cls][t] = [tp, len(p), len(g)]
    return out


class MapMetricAccumulator:
    """Pools AP and mIoU counts over all frames of one run, then reports them."""

    def __init__(self, radius: float, thresholds=THRESHOLDS, resolution=0.25, line_width=1.0):
        assert 1.0 in thresholds
        self.thresholds = tuple(thresholds)
        self.grid = Grid.from_bounds(-radius, -radius, radius, radius, resolution)
        self.thick = max(1, int(round(line_width / resolution)))
        self.counts = {c: {t: np.zeros(3) for t in self.thresholds} for c in CLASSES}
        self.iou = {c: np.zeros(2) for c in CLASSES}  # intersection, union (pixels)

    def add(self, truth: VectorMap, pred: VectorMap) -> None:
        for cls, per_t in match_counts(truth, pred, self.thresholds).items():
            for t, v in per_t.items():
                self.counts[cls][t] += v
        for cls in CLASSES:
            if not truth.of_class(cls) and not pred.of_class(cls):
                continue
            a = rasterize_class(truth, cls, self.grid, self.thick) > 0
            b = rasterize_class(pred, cls, self.grid, self.thick) > 0
            self.iou[cls] += (np.logical_and(a, b).sum(), np.logical_or(a, b).sum())

    def result(self) -> dict:
        t0 = self.thresholds[0]
        out = {f"ap_{c}": np.nan for c in CLASSES}
        ap_all, ap_t0 = [], []
        for cls in CLASSES:
            c = self.counts[cls]
            n_gt = c[t0][2]
            if n_gt == 0:
                continue  # no ground truth for this class: AP undefined, skip it
            per_t = []
            for t in self.thresholds:
                tp, n_pred, _ = c[t]
                per_t.append((tp / n_pred if n_pred else 0.0) * tp / n_gt)
            out[f"ap_{cls}"] = float(np.mean(per_t))
            ap_all.append(float(np.mean(per_t)))
            ap_t0.append(per_t[0])
        if ap_all:
            out["map_ap"], out["map_ap_t05"] = float(np.mean(ap_all)), float(np.mean(ap_t0))
        else:  # no ground truth at all: perfect only if nothing was predicted either
            empty = sum(self.counts[c][t0][1] for c in CLASSES) == 0
            out["map_ap"] = out["map_ap_t05"] = 1.0 if empty else 0.0
        ious = [i / u for i, u in self.iou.values() if u > 0]
        out["map_miou"] = float(np.mean(ious)) if ious else 1.0
        tp = sum(self.counts[c][1.0][0] for c in TRANSVERSE)
        gt = sum(self.counts[c][1.0][2] for c in TRANSVERSE)
        out["transverse_recall"] = float(tp / gt) if gt else np.nan
        return out
