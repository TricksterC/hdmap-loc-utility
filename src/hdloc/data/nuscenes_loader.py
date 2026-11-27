from __future__ import annotations

import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes

from hdloc.data.schema import Element, VectorMap
from hdloc.maps.transforms import global_to_ego, quat_to_yaw


def load_scene_poses(nusc: NuScenes, scene: dict) -> tuple[str, np.ndarray]:
    """Ground-truth trajectory of a scene: (map_name, array of [x, y, yaw] per keyframe)."""
    map_name = nusc.get("log", scene["log_token"])["location"]
    poses = []
    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        p = nusc.get("ego_pose", sd["ego_pose_token"])
        poses.append([p["translation"][0], p["translation"][1], quat_to_yaw(p["rotation"])])
        token = sample["next"]
    return map_name, np.array(poses)


def load_local_map(nmap: NuScenesMap, x: float, y: float, radius: float = 50.0) -> VectorMap:
    """Map elements around (x, y), in the global frame."""
    patch = (x - radius, y - radius, x + radius, y + radius)
    layers = ["lane_divider", "road_divider", "ped_crossing", "stop_line"]
    found = nmap.get_records_in_patch(patch, layers, mode="intersect")

    elements = []
    for layer in ("lane_divider", "road_divider"):
        for token in found[layer]:
            rec = nmap.get(layer, token)
            line = nmap.extract_line(rec["line_token"])
            elements.append(Element("divider", np.array(line.coords)))
    for layer, cls in (("ped_crossing", "ped_crossing"), ("stop_line", "stop_line")):
        for token in found[layer]:
            rec = nmap.get(layer, token)
            poly = nmap.extract_polygon(rec["polygon_token"])
            elements.append(Element(cls, np.array(poly.exterior.coords)))
    return VectorMap(elements, frame="global")


def to_ego(vmap: VectorMap, x: float, y: float, yaw: float) -> VectorMap:
    return vmap.transformed(lambda p: global_to_ego(p, x, y, yaw), frame="ego")

def load_trajectory_map(nmap: NuScenesMap, poses: np.ndarray, margin: float = 80.0) -> VectorMap:
    """Map elements covering a whole trajectory plus a margin, in the global frame."""
    xs, ys = poses[:, 0], poses[:, 1]
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2
    return load_local_map(nmap, cx, cy, radius=half + margin)

def scene_table(nusc: NuScenes) -> list[dict]:
    """One row per scene: name, location, number of keyframes, path length in metres."""
    rows = []
    for scene in nusc.scene:
        location, poses = load_scene_poses(nusc, scene)
        steps = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
        rows.append({"name": scene["name"], "location": location,
                     "n_samples": len(poses), "path_len": float(steps.sum())})
    return rows
