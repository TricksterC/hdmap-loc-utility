# hdmap-loc-utility

How do errors in a predicted vector HD map turn into pose errors? A log-replay study on nuScenes
with a particle-filter localizer. We measure how coherent and random map damage passes into
sideways, forward and heading error, which element classes make each direction observable, and
whether AP and mIoU predict localization error. We also test a rigid-offset "alignment" score in
metres. Written for a chapter in *AI for Autonomous Driving* (Wiley).

## Findings

- A map shifted or rotated as a whole passes almost one-to-one into pose error (1 m sideways
  shift gives 1.00 m; 2 m forward shift gives 1.95 m; 4 deg rotation gives 0.5 m sideways).
  Random damage (dropout, false elements, jitter) barely matters.
- With no map elements at all, the filter gets lost in 60% of held-out runs. With only dividers
  removed, 8% of runs are lost, although the median run looks fine.
- AP and mIoU rank maps well within one damage type, but are not calibrated across types: the same
  AP can mean 0.04 m or more than 1 m of error. Pooled Spearman with sideways error: AP -0.42,
  mIoU -0.53 (held-out).
- The alignment offset is calibrated in metres. On coherent damage it predicts sideways error to
  0.10 m (mIoU 0.22 m, AP 0.28 m), forward to 0.06 m (0.32 m, 0.33 m), heading to 0.20 deg
  (0.50, 0.47). It cannot see non-rigid damage or the sideways lag caused by rotation.
- Reporting the alignment offset together with its constraint strength flags most of the runs
  where the offset reads about zero but the filter is far off (AUC 0.97 on 18 such held-out runs,
  mostly with all dividers removed; post hoc).
- Our pre-registered ranking test (S1) failed on all three components on both dev and held-out
  scenes; S2 passed for sideways and heading. See the chapter for the full account.

## Setup

```
make install            # run inside a Python 3.10+ virtualenv
make check              # ruff (with --fix) + tests
make test               # tests only
```

`requirements-lock.txt` lists the package versions the results were produced with
(numpy < 2, pandas < 2.3)

Data (not in the repo): nuScenes v1.0-mini and the map expansion, plus the trainval metadata
(JSON tables only, no sensor data) for the held-out scenes. Expected layout:

```
data/nuscenes/v1.0-mini/    (inner v1.0-mini/ and maps/expansion/)
data/nuscenes/trainval/     (v1.0-trainval/ and maps -> ../v1.0-mini/maps)
```

## Reproduce

```
python3 scripts/run_oracle.py                # quick check that the data loads
make sweep                                   # dev sweep (8 moving mini scenes)
python3 scripts/run_transfer_sweep.py --root data/nuscenes/trainval --version v1.0-trainval \
    --names-file configs/heldout_scenes.yaml --out heldout_sweep    # held-out sweep
python3 scripts/run_class_ablation.py        # per-class ablation (results/class_ablation.csv)
python3 scripts/ap_vs_error.py [--files heldout_sweep]
python3 scripts/align_vs_error.py [--files heldout_sweep]       # pre-registered S1/S2
python3 scripts/align_exploratory.py [--files heldout_sweep]    # post hoc analysis
python3 scripts/info_check.py                                   # post hoc constraint-strength analysis
python3 scripts/make_figures.py                                 # writes figures/
```

The CSVs the figures are built from are committed in `results/`, so you can run the analysis
scripts and `make_figures.py` without the data. Sweep times depend on your machine; the dev sweep
took a few minutes and the held-out sweep about nine on ours.

The held-out scene list (`configs/heldout_scenes.yaml`) was fixed before the held-out run.
Tag `freeze-v1` is the pipeline at that point; `results-v2` has the corrected longitudinal
error definition (see below). `results-v1` is superseded.

## Notes on the analysis

- The S1/S2 criteria are written in the header of `scripts/align_vs_error.py` and were not changed.
- The first runs defined longitudinal error by subtracting an absolute baseline from a signed
  quantity. We corrected it to the signed forward bias relative to an undamaged map; the original
  outputs are in `results/s1s2_*_original.txt`. The pass/fail outcomes did not change.
- `scripts/align_exploratory.py` and `scripts/info_check.py` were written after seeing the S1/S2
  results and are post hoc.

## Layout

```
src/hdloc/   data, maps, localization (particle filter), degradation, metrics, experiments
scripts/     sweeps, analyses, figures
configs/     filter, degradations, held-out scene list
tests/       unit tests
results/     CSVs and logs (results/figures/ is scratch and not tracked)
figures/     final figures
```

## Limitations

Damage is synthetic; one localizer, one dataset; drift model is ours; nuScenes ground-truth pose
comes from LiDAR map matching. No real online-map predictions are evaluated.

## Citation and license

See `CITATION.cff`. Code is MIT licensed (see `LICENSE`). nuScenes is used under its own terms
(non-commercial use); cite Caesar et al., nuScenes (CVPR 2020).
