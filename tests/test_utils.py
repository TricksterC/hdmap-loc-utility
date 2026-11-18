from hdloc.utils.seed import set_seed


def test_seed_is_repeatable():
    a = set_seed(0).random(5)
    b = set_seed(0).random(5)
    assert (a == b).all()
