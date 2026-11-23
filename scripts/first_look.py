# Load the dataset
from pathlib import Path

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes

# Make script to run from anywhere
REPO = Path(__file__).resolve().parents[1]
DATAROOT = str(REPO / "data" / "nuscenes" / "v1.0-mini")

nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=False)

scene = nusc.scene[0] # First scene in dataset

# Find the location 
log = nusc.get("log", scene["log_token"])
print("Scene:", scene["name"], "| location:", log["location"])

# Get vehicle's position
sample = nusc.get("sample", scene["first_sample_token"])
sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
pose = nusc.get("ego_pose", sd["ego_pose_token"])
x, y, _ = pose["translation"]
print("Ego position (global):", round(x, 2), round(y, 2))
print("Ego rotation (quaternion):", pose["rotation"])

# Print all lane dividers, road dividers, pedestrian crossings,
# and stop lines that intersect a 100 m × 100 m region around the vehicle.
nmap = NuScenesMap(dataroot=DATAROOT, map_name=log["location"])
layers = ["lane_divider", "road_divider", "ped_crossing", "stop_line"]
patch = (x - 50, y - 50, x + 50, y + 50)
found = nmap.get_records_in_patch(patch, layers, mode="intersect")
for layer in layers:
    print(f"{layer:14s}: {len(found[layer])} elements within 50 m")
