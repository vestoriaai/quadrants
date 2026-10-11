"""Gate probe (C5): qd.algorithms CPU-backend serial paths — scan/reduce/select vs numpy.

The CPU path dispatches at kernel-compile time via qd.static(is_cpu()); this probe runs the
CPU backend exclusively (the GPU staircases keep their own upstream gates). Cases:
  - exclusive_scan_add: i32 / u32 / f32, n in {1, 5, 257, 5000}  (257 crosses BLOCK_DIM=256)
  - reduce_add: same dtypes/n
  - select: i32 payload, random 0/1 flags, several densities
Run: python probe_algorithms_cpu.py cpu   (exit 0 = pass)
Ledger: mujocoqd/compiler-line C5.
"""
import sys

import numpy as np

import quadrants as qd
from quadrants.algorithms import (
    exclusive_scan_add,
    exclusive_scan_scratch_slots,
    reduce_add,
    reduce_scratch_slots,
    select,
    select_scratch_slots,
)
from quadrants.algorithms._cpu_paths import is_cpu

qd.init(arch=qd.cpu)

fails = []
LOG256 = 3  # handles n <= 256**3


def gate(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL':4}] {name} {detail}")
    if not ok:
        fails.append(name)


np_dt = {qd.i32: np.int32, qd.u32: np.uint32, qd.f32: np.float32}

# -------------------------------------------------- exclusive_scan_add
for dt in (qd.i32, qd.u32, qd.f32):
    for n in (1, 5, 257, 5000):
        rng = np.random.default_rng(n)
        if dt == qd.f32:
            arr = (rng.standard_normal(n) * 10).astype(np.float32)
        else:
            arr = rng.integers(-1000, 1000, n).astype(np_dt[dt])
        arr_t = qd.ndarray(dtype=dt, shape=n)
        arr_t.from_numpy(arr)
        out_t = qd.ndarray(dtype=dt, shape=n)
        out_t.from_numpy(np.zeros(n, dtype=np_dt[dt]))
        scratch = qd.ndarray(dtype=qd.u32, shape=exclusive_scan_scratch_slots(n, LOG256))
        scratch.from_numpy(np.zeros(scratch.shape[0], dtype=np.uint32))

        @qd.kernel
        def k(arr: qd.types.ndarray(ndim=1), out: qd.types.ndarray(ndim=1), scratch: qd.types.ndarray(ndim=1), n: qd.i32):
            exclusive_scan_add(arr, out, scratch, n, dt, LOG256)

        k(arr_t, out_t, scratch, n)
        qd.sync()
        got = out_t.to_numpy()
        want = np.concatenate([[np.zeros(1, dtype=np_dt[dt])[0]], np.cumsum(arr)[:-1]]).astype(np_dt[dt])
        # f32 tolerance for accumulation-order difference (serial vs numpy pairwise)
        tol = 1e-3 * max(1.0, np.abs(want).max()) if dt == qd.f32 else 0
        ok = np.abs(got.astype(np.float64) - want.astype(np.float64)).max() <= tol
        gate(f"scan_add {dt} n={n}", ok, f"maxerr={np.abs(got.astype(np.float64) - want.astype(np.float64)).max():.3g}")

# -------------------------------------------------- reduce_add
for dt in (qd.i32, qd.f32):
    for n in (1, 5, 257, 5000):
        rng = np.random.default_rng(n + 7)
        arr = (rng.standard_normal(n) * 10).astype(np.float32) if dt == qd.f32 else rng.integers(-1000, 1000, n).astype(np_dt[dt])
        arr_t = qd.ndarray(dtype=dt, shape=n)
        arr_t.from_numpy(arr)
        out_t = qd.ndarray(dtype=dt, shape=1)
        out_t.from_numpy(np.zeros(1, dtype=np_dt[dt]))
        scratch = qd.ndarray(dtype=qd.u32, shape=reduce_scratch_slots(n, LOG256))
        scratch.from_numpy(np.zeros(scratch.shape[0], dtype=np.uint32))

        @qd.kernel
        def kr(arr: qd.types.ndarray(ndim=1), out: qd.types.ndarray(ndim=1), scratch: qd.types.ndarray(ndim=1), n: qd.i32):
            reduce_add(arr, out, scratch, n, dt, LOG256)

        kr(arr_t, out_t, scratch, n)
        qd.sync()
        got = float(out_t.to_numpy()[0])
        want = float(arr.astype(np.float64).sum())
        tol = 1e-3 * max(1.0, abs(want)) if dt == qd.f32 else 0
        gate(f"reduce_add {dt} n={n}", abs(got - want) <= tol, f"got={got:.4g} want={want:.4g}")

# -------------------------------------------------- select
for n, density in ((1, 1.0), (100, 0.5), (257, 0.1), (5000, 0.3)):
    rng = np.random.default_rng(n)
    arr = (rng.standard_normal(n) * 10).astype(np.float32)
    flags = (rng.random(n) < density).astype(np.int32)
    arr_t = qd.ndarray(dtype=qd.f32, shape=n)
    arr_t.from_numpy(arr)
    fl_t = qd.ndarray(dtype=qd.i32, shape=n)
    fl_t.from_numpy(flags)
    out_t = qd.ndarray(dtype=qd.f32, shape=n)
    out_t.from_numpy(np.zeros(n, dtype=np.float32))
    no_t = qd.ndarray(dtype=qd.i32, shape=1)
    no_t.from_numpy(np.zeros(1, dtype=np.int32))
    scratch = qd.ndarray(dtype=qd.u32, shape=select_scratch_slots(n))
    scratch.from_numpy(np.zeros(scratch.shape[0], dtype=np.uint32))

    @qd.kernel
    def ks(arr: qd.types.ndarray(ndim=1), flags: qd.types.ndarray(ndim=1), out: qd.types.ndarray(ndim=1),
           num_out: qd.types.ndarray(ndim=1), scratch: qd.types.ndarray(ndim=1), n: qd.i32):
        select(arr, flags, out, num_out, scratch, n, LOG256)

    ks(arr_t, fl_t, out_t, no_t, scratch, n)
    qd.sync()
    got = out_t.to_numpy()[: int(no_t.to_numpy()[0])]
    want = arr[flags == 1]
    gate(f"select n={n} d={density}", got.shape == want.shape and np.allclose(got, want),
         f"count={got.shape[0]} want={want.shape[0]}")

# -------------------------------------------------- is_cpu predicate (post-init)
gate("is_cpu_predicate", is_cpu() is True, f"{is_cpu()}")

print(f"\nALGORITHMS CPU GATE: {'ALL PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
