from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "data" / "nuscenes" / "v1.0-mini"
pytestmark = pytest.mark.skipif(not ROOT.exists(), reason="nuScenes mini not available")


def test_loads_trajectory_and_map():
    from nuscenes.map_expansion.map_api import NuScenesMap
    from nuscenes.nuscenes import NuScenes

    from hdloc.data.nuscenes_loader import load_local_map, load_scene_poses, to_ego

    nusc = NuScenes(version="v1.0-mini", dataroot=str(ROOT), verbose=False)
    map_name, poses = load_scene_poses(nusc, nusc.scene[0])
    assert poses.shape[1] == 3 and len(poses) > 10

    nmap = NuScenesMap(dataroot=str(ROOT), map_name=map_name)
    x, y, yaw = poses[0]
    ego_map = to_ego(load_local_map(nmap, x, y), x, y, yaw)
    assert ego_map.frame == "ego" and len(ego_map.elements) > 0
