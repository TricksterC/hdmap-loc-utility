from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hdloc.data.schema import TRANSVERSE, VectorMap
from hdloc.degradation.pipeline import degrade
from hdloc.localization.baselines import odometry_only
from hdloc.localization.motion_model import DriftModel, inject_drift, relative_increments
from hdloc.localization.observation_model import sample_points
from hdloc.localization.particle_filter import FilterConfig, run_filter
from hdloc.maps.observe import densify_map, observe
from hdloc.maps.prior_map import PriorMap, build_prior, grid_for_poses
from hdloc.metrics.loc_metrics import pose_errors


@dataclass
class SceneData:
    name: str
    map_name: str
    poses: np.ndarray  # (T, 3) ground truth [x, y, yaw]
    vmap: VectorMap  # clean, densified ground-truth map (global frame)
    prior: PriorMap

    @property
    def path_len(self) -> float:
        return float(np.linalg.norm(np.diff(self.poses[:, :2], axis=0), axis=1).sum())


def load_scenes(root: Path | str, indices=None, resolution=0.2, margin=80.0) -> list[SceneData]:
    from nuscenes.map_expansion.map_api import NuScenesMap
    from nuscenes.nuscenes import NuScenes

    from hdloc.data.nuscenes_loader import load_scene_poses, load_trajectory_map

    nusc = NuScenes(version="v1.0-mini", dataroot=str(root), verbose=False)
    maps: dict = {}
    scenes = []
    for i in indices if indices else range(len(nusc.scene)):
        scene = nusc.scene[i]
        map_name, poses = load_scene_poses(nusc, scene)
        if map_name not in maps:
            maps[map_name] = NuScenesMap(dataroot=str(root), map_name=map_name)
        vmap = densify_map(load_trajectory_map(maps[map_name], poses), 1.0)
        prior = build_prior(vmap, grid_for_poses(poses, margin=margin, resolution=resolution))
        scenes.append(SceneData(scene["name"], map_name, poses, vmap, prior))
    return scenes


def evaluate(
    scene: SceneData,
    kind: str,
    value: float,
    seed: int,
    filter_cfg: FilterConfig,
    drift: DriftModel,
    radius: float = 50.0,
) -> dict:
    """One filter run with one degradation. Separate RNGs keep runs paired across values."""
    rng_drift = np.random.default_rng(seed)
    rng_deg = np.random.default_rng(seed + 1000)
    rng_pf = np.random.default_rng(seed + 2000)

    gt = scene.poses
    incs = inject_drift(relative_increments(gt), drift, rng_drift)
    dead = odometry_only(gt[0], incs)

    obs_radius = float(value) if kind == "range_m" else radius
    obs = []
    for pose in gt:
        seen = observe(scene.vmap, pose, obs_radius)
        seen = degrade(seen, kind, value, rng_deg, obs_radius)
        obs.append(sample_points(seen, filter_cfg.obs.spacing))

    est, _ = run_filter(scene.prior, gt[0], incs, obs, filter_cfg, rng_pf)

    out = {"transverse_frac": float(np.mean([any(c in o for c in TRANSVERSE) for o in obs]))}
    errs = {"pf": pose_errors(est, gt), "odo": pose_errors(dead, gt)}
    for tag, e in errs.items():
        out[f"{tag}_lon_med"] = float(np.median(np.abs(e["lon"])))
        out[f"{tag}_lat_med"] = float(np.median(np.abs(e["lat"])))
        out[f"{tag}_lon_p95"] = float(np.percentile(np.abs(e["lon"]), 95))
        out[f"{tag}_lat_p95"] = float(np.percentile(np.abs(e["lat"]), 95))
        out[f"{tag}_yaw_med_deg"] = float(np.degrees(np.median(np.abs(e["yaw"]))))
    out["pf_lon_bias"] = float(np.median(errs["pf"]["lon"]))  # signed
    out["pf_lat_bias"] = float(np.median(errs["pf"]["lat"]))
    return out
