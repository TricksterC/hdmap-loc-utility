from hdloc.data.splits import select_heldout


def _rows():
    rows = []
    for loc in ("boston", "singapore"):
        for i in range(30):
            rows.append({"name": f"{loc}-{i:03d}", "location": loc,
                         "path_len": 50.0 if i % 5 else 3.0, "n_samples": 40})
    return rows


def test_excludes_dev_and_stationary_scenes():
    dev = {"boston-001", "boston-002"}
    names, avail = select_heldout(_rows(), dev, per_location=10, seed=0)
    assert not dev & set(names)
    assert all(int(n.split("-")[1]) % 5 for n in names)  # stationary ones have i % 5 == 0
    assert avail["boston"] == 22


def test_even_across_locations_and_deterministic():
    a, _ = select_heldout(_rows(), set(), per_location=8, seed=7)
    b, _ = select_heldout(_rows(), set(), per_location=8, seed=7)
    assert a == b
    assert sum(n.startswith("boston") for n in a) == 8
    assert sum(n.startswith("singapore") for n in a) == 8


def test_takes_all_when_a_location_is_short():
    names, avail = select_heldout(_rows(), set(), per_location=100, seed=0)
    assert len(names) == sum(avail.values()) == 48
