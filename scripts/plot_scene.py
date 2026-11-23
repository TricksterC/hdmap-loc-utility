from pathlib import Path

import matplotlib.pyplot as plt
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes

from hdloc.data.nuscenes_loader import load_local_map, load_scene_poses, to_ego

ROOT = Path(__file__).resolve().parents[1] / "data" / "nuscenes" / "v1.0-mini"
nusc = NuScenes(version="v1.0-mini", dataroot=str(ROOT), verbose=False)

# Load scene's vehicle trajectory
map_name, poses = load_scene_poses(nusc, nusc.scene[0])

# Load HD Map
nmap = NuScenesMap(dataroot=str(ROOT), map_name=map_name)

# Retrieve vehicle initial position and heading
x, y, yaw = poses[0]

# Extract map features near the vehicle & convert their coordinates 
# from global to vehicle's coordinate frame
m = to_ego(load_local_map(nmap, x, y), x, y, yaw)

# Plot extracted features
colors = {"divider": "tab:blue", "ped_crossing": "tab:orange", "stop_line": "tab:red"}
for e in m.elements:
    plt.plot(e.points[:, 0], e.points[:, 1], color=colors.get(e.cls, "gray"))

# Black triangle at origin, represents the vehicle
plt.plot(0, 0, "k^", markersize=12)
plt.axis("equal")
plt.title(f"{nusc.scene[0]['name']} (ego frame, car at origin facing +x)")
out = Path(__file__).resolve().parents[1] / "results" / "figures"
out.mkdir(parents=True, exist_ok=True)
plt.savefig(out / "scene_check.png", dpi=150)
