"""R5 bench: default vs host_visible ndarray readback A/B + fast-path breakdown.

Run on the venv carrying the R5 wheel. Metal only (the R5 target).
Interleaved min-of-k rounds (machine-state variance discipline, see gate-4
lesson in docs/robostudio_features.md).

Usage:
  python bench_r5_host_visible.py            # A/B table
  python bench_r5_host_visible.py --break    # component breakdown on 64KB
"""
import sys
import time

import numpy as np

import quadrants as qd
import quadrants.lang.impl as qd_impl
from quadrants.lang.impl import grouped  # kernel-scope name (as _kernels.py does)

qd.init(arch=qd.metal)


def best_ms(fn, k):
    best = float("inf")
    for _ in range(k):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def ab(size_label, nbytes, k, rounds=4):
    n = nbytes // 4
    host = np.zeros(n, np.float32)
    a_def = qd.ndarray(qd.f32, n)
    a_hv = qd.ndarray(qd.f32, n, host_visible=True)
    a_hv.from_numpy(host)
    qd.sync()
    acc = {("to", "def"): float("inf"), ("to", "hv"): float("inf"),
           ("fr", "def"): float("inf"), ("fr", "hv"): float("inf")}
    for _ in range(rounds):
        for arm, arr in (("def", a_def), ("hv", a_hv)):
            acc[("to", arm)] = min(acc[("to", arm)], best_ms(lambda a=arr: a.to_numpy(), k))
            acc[("fr", arm)] = min(acc[("fr", arm)], best_ms(lambda a=arr, h=host: a.from_numpy(h), k))
    print(f"{size_label:>6}  to_numpy def={acc[('to', 'def')]:8.3f} hv={acc[('to', 'hv')]:8.3f} "
          f"(x{acc[('to', 'hv')] / max(acc[('to', 'def')], 1e-9):.3f})  |  from_numpy "
          f"def={acc[('fr', 'def')]:8.3f} hv={acc[('fr', 'hv')]:8.3f} "
          f"(x{acc[('fr', 'hv')] / max(acc[('fr', 'def')], 1e-9):.3f})")


print(f"commit banner: {sys.argv[0]}")
print("size      " + "to/default / to/hv (ratio) | from/default / from/hv (ratio)   [ms, min-of-k x rounds]")
for label, nbytes, k in (("1KB", 1024, 50), ("64KB", 64 * 1024, 30), ("256KB", 256 * 1024, 20),
                         ("1MB", 1024 * 1024, 15), ("4MB", 4 * 1024 * 1024, 12),
                         ("16MB", 16 * 1024 * 1024, 10), ("64MB", 64 * 1024 * 1024, 8)):
    ab(label, nbytes, k)

if "--break" in sys.argv:
    print("-- fast-path breakdown (64KB hv, min-of-50) --")
    n = 64 * 1024 // 4
    host = np.zeros(n, np.float32)
    a = qd.ndarray(qd.f32, n, host_visible=True)
    a.from_numpy(host)
    qd.sync()
    out = np.empty(n, np.float32)
    # component timings
    t_npempty = best_ms(lambda: np.empty(n, np.float32), 200)
    t_sync = best_ms(lambda: qd.sync(), 200)
    t_copy = best_ms(lambda: qd_impl.get_runtime().prog.ndarray_copy_to_host(a.arr, out.ctypes.data, out.nbytes), 200)
    t_full = best_ms(lambda: a.to_numpy(), 200)
    b_def = qd.ndarray(qd.f32, n)
    b_def.from_numpy(host)
    qd.sync()
    t_def = best_ms(lambda: b_def.to_numpy(), 50)
    print(f"  np.empty(64KB f32)          = {t_npempty:7.4f} ms")
    print(f"  qd.sync() (idle drain)      = {t_sync:7.4f} ms")
    print(f"  ndarray_copy_to_host alone  = {t_copy:7.4f} ms  (sync + map + memcpy + unmap)")
    print(f"  hv to_numpy() full chain    = {t_full:7.4f} ms")
    print(f"  default to_numpy() 64KB     = {t_def:7.4f} ms")
