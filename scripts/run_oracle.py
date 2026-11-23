import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes

from hdloc.data.nuscenes_loader import load_scene_poses, load_trajectory_map
from hdloc.localization.baselines import odometry_only
from hdloc.localization.motion_model import DriftModel, inject_drift, relative_increments
from hdloc.localization.observation_model import sample_points
from hdloc.localization.particle_filter import FilterConfig, run_filter
from hdloc.maps.observe import densify_map, observe
from hdloc.maps.prior_map import build_prior, grid_for_poses
from hdloc.metrics.loc_metrics import pose_errors
from hdloc.utils.io import load_yaml
from hdloc.utils.seed import set_seed

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "nuscenes" / "v1.0-mini"
scene_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

rng = set_seed(0)
nusc = NuScenes(version="v1.0-mini", dataroot=str(ROOT), verbose=False)
scene = nusc.scene[scene_idx]
map_name, poses = load_scene_poses(nusc, scene)
nmap = NuScenesMap(dataroot=str(ROOT), map_name=map_name)

vmap = densify_map(load_trajectory_map(nmap, poses), 1.0)
prior = build_prior(vmap, grid_for_poses(poses, margin=80.0, resolution=0.2))

drift = DriftModel(scale_error=0.03, yaw_bias=0.003, trans_noise=0.02, yaw_noise=0.002)
incs = inject_drift(relative_increments(poses), drift, rng)
dead = odometry_only(poses[0], incs)

obs = [sample_points(observe(vmap, p, 50.0)) for p in poses]  # oracle: true map
cfg = FilterConfig.from_dict(load_yaml(REPO / "configs" / "filter.yaml"))
est, spread = run_filter(prior, poses[0], incs, obs, cfg, rng)

print(f"{scene['name']} ({map_name}), {len(poses)} frames")
for name, traj in (("odometry-only", dead), ("oracle filter", est)):
    e = pose_errors(traj, poses)
    print(
        f"{name:14s} median |lon| {np.median(abs(e['lon'])):.2f} m | "
        f"|lat| {np.median(abs(e['lat'])):.2f} m | "
        f"|yaw| {np.degrees(np.median(abs(e['yaw']))):.2f} deg"
    )

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(poses[:, 0], poses[:, 1], "k-", lw=2, label="ground truth")
ax[0].plot(dead[:, 0], dead[:, 1], "r--", label="odometry-only")
ax[0].plot(est[:, 0], est[:, 1], "g-", label="oracle filter")
ax[0].axis("equal")
ax[0].legend()
ax[0].set_title(scene["name"])
t = np.arange(len(poses)) * 0.5
for name, traj in (("odometry-only", dead), ("oracle filter", est)):
    e = pose_errors(traj, poses)
    ax[1].plot(t, abs(e["lon"]), label=f"{name} |lon|")
    ax[1].plot(t, abs(e["lat"]), "--", label=f"{name} |lat|")
ax[1].set_xlabel("time (s)")
ax[1].set_ylabel("error (m)")
ax[1].legend()
out = REPO / "results" / "figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / f"oracle_{scene['name']}.png", dpi=130)
