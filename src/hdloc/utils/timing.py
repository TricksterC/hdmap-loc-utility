import time
from contextlib import contextmanager


@contextmanager
def timer(label="block"):
    t0 = time.perf_counter()
    yield
    print(f"[timer] {label}: {time.perf_counter() - t0:.3f}s")
