"""Write configs/heldout_scenes.yaml. Run it ONCE, commit the file, then do not touch it.

Needs the trainval metadata (JSON tables only). Held-out scenes are chosen evenly across the four
nuScenes locations, excluding the 10 mini scenes and near-stationary scenes (path <= 10 m).
"""

import argparse
from pathlib import Path

import yaml
from nuscenes.nuscenes import NuScenes

from hdloc.data.nuscenes_loader import scene_table
from hdloc.data.splits import select_heldout

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--trainval-root", default=str(REPO / "data" / "nuscenes" / "trainval"))
ap.add_argument("--mini-root", default=str(REPO / "data" / "nuscenes" / "v1.0-mini"))
ap.add_argument("--per-location", type=int, default=12)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--out", default=str(REPO / "configs" / "heldout_scenes.yaml"))
args = ap.parse_args()

out = Path(args.out)
if out.exists():
    raise SystemExit(f"{out} already exists. The held-out list is fixed once; not overwriting.")

mini = NuScenes(version="v1.0-mini", dataroot=args.mini_root, verbose=False)
dev_names = {s["name"] for s in mini.scene}
nusc = NuScenes(version="v1.0-trainval", dataroot=args.trainval_root, verbose=False)
names, available = select_heldout(scene_table(nusc), dev_names, args.per_location, args.seed)

print("candidates per location:", available)
print(f"selected {len(names)} scenes")
out.parent.mkdir(parents=True, exist_ok=True)
header = (
    "# Held-out scenes. Written once by scripts/make_heldout_list.py; never edited afterwards.\n"
    f"# seed={args.seed}, per_location={args.per_location}, excludes the mini scenes, path > 10 m\n"
)
out.write_text(header + yaml.safe_dump({"scenes": names, "candidates": available}))
print("wrote", out)
