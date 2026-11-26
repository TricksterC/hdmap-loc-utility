from pathlib import Path

import matplotlib.pyplot as plt
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes

from hdloc.data.nuscenes_loader import load_scene_poses, load_trajectory_map
from hdloc.maps.prior_map import build_prior, grid_for_poses

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "nuscenes" / "v1.0-mini"

nusc = NuScenes(version="v1.0-mini", dataroot=str(ROOT), verbose=False)
scene = nusc.scene[0]

# Load vehicle's entire trajectory
map_name, poses = load_scene_poses(nusc, scene)
nmap = NuScenesMap(dataroot=str(ROOT), map_name=map_name)

# Extract relevant features from HD map
vmap = load_trajectory_map(nmap, poses)
grid = grid_for_poses(poses, margin=80.0, resolution=0.2)
prior = build_prior(vmap, grid, max_dist=10.0)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, cls in zip(axes, ("divider", "stop_line")):
    dt = prior.dts.get(cls)
    if dt is not None:
        im = ax.imshow(dt, origin="lower", extent=grid.extent, cmap="viridis")
        fig.colorbar(im, ax=ax, label="distance to nearest line (m)")
    ax.plot(poses[:, 0], poses[:, 1], "r-", lw=2, label="ego trajectory")
    ax.set_title(f"{scene['name']}: {cls} distance transform")
    ax.legend()
out = REPO / "results" / "figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "distance_transform.png", dpi=130)
print("saved", out / "distance_transform.png", "| grid", grid.shape)
