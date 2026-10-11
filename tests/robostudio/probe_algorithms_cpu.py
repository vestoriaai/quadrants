"""Gate probe (C5): qd.algorithms CPU-backend serial paths — scan/reduce/select vs numpy.

The CPU path dispatches at kernel-compile time via qd.static(is_cpu()); this probe runs the
CPU backend exclusively (the GPU staircases keep their own upstream gates). Cases:
  - exclusive_scan_add: i32 / u32 / f32, n in {1, 5, 257, 5000}  (257 crosses BLOCK_DIM=256)
  - reduce_add: same dtypes/n
  - select: i32 payload, random 0/1 flags, several densities
  - exclusive_scan_min / exclusive_scan_max: i32 / f32, n in {5, 257, 5000}
  - sort: u32 keys + f32 values (stable tie-break), n in {1, 257, 5000}
Run: python probe_algorithms_cpu.py cpu   (exit 0 = pass)
Ledger: mujocoqd/compiler-line C5.
"""
import sys

import numpy as np

import quadrants as qd
from quadrants.algorithms import (
    exclusive_scan_add,
    exclusive_scan_max,
    exclusive_scan_min,
    exclusive_scan_scratch_slots,
    reduce_add,
    reduce_scratch_slots,
    select,
    select_scratch_slots,
    sort,
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

# -------------------------------------------------- exclusive_scan_min / exclusive_scan_max
# Identities mirror _MIN/_MAX_IDENTITY_BY_DTYPE (INT32_MAX / INT32_MIN; +/-inf for f32): lane 0 emits the
# identity, lane i the running min/max over arr[0:i] (exclusive). want = exclusive
# minimum.accumulate / maximum.accumulate vs numpy.
min_identity = {qd.i32: np.iinfo(np.int32).max, qd.f32: np.float32(np.inf)}
max_identity = {qd.i32: np.iinfo(np.int32).min, qd.f32: np.float32(-np.inf)}


def minmax_close(got, want, dt):
    """i32 exact; f32 at 1e-3 x max|want| relative, with the +/-inf identity lanes matched exactly
    (abs(inf - inf) is nan, so the tolerance compare alone would false-FAIL lane 0)."""
    if dt != qd.f32:
        return np.array_equal(got, want)
    nonfin = ~np.isfinite(want)
    if np.any(got[nonfin] != want[nonfin]):
        return False
    fin = ~nonfin
    if not np.any(fin):
        return True
    tol = 1e-3 * max(1.0, float(np.abs(want[fin].astype(np.float64)).max()))
    return bool(np.all(np.abs(got[fin].astype(np.float64) - want[fin].astype(np.float64)) <= tol))


def minmax_maxerr(got, want):
    g, w = got.astype(np.float64), want.astype(np.float64)
    fin = np.isfinite(g) & np.isfinite(w)
    return float(np.abs(g[fin] - w[fin]).max()) if np.any(fin) else 0.0


for dt in (qd.i32, qd.f32):
    for n in (5, 257, 5000):
        rng = np.random.default_rng(n + 41)
        if dt == qd.f32:
            arr = (rng.standard_normal(n) * 10).astype(np.float32)
        else:
            arr = rng.integers(-1000, 1000, n).astype(np_dt[dt])
        arr_t = qd.ndarray(dtype=dt, shape=n)
        arr_t.from_numpy(arr)

        # --- min
        scratch = qd.ndarray(dtype=qd.u32, shape=exclusive_scan_scratch_slots(n, LOG256))
        scratch.from_numpy(np.zeros(scratch.shape[0], dtype=np.uint32))
        out_t = qd.ndarray(dtype=dt, shape=n)
        out_t.from_numpy(np.zeros(n, dtype=np_dt[dt]))

        @qd.kernel
        def kmin(arr: qd.types.ndarray(ndim=1), out: qd.types.ndarray(ndim=1), scratch: qd.types.ndarray(ndim=1), n: qd.i32):
            exclusive_scan_min(arr, out, scratch, n, dt, LOG256)

        kmin(arr_t, out_t, scratch, n)
        qd.sync()
        got = out_t.to_numpy()
        want = np.concatenate([[min_identity[dt]], np.minimum.accumulate(arr)[:-1]]).astype(np_dt[dt])
        gate(f"scan_min {dt} n={n}", minmax_close(got, want, dt), f"maxerr={minmax_maxerr(got, want):.3g}")

        # --- max (fresh out / re-zeroed scratch)
        scratch.from_numpy(np.zeros(scratch.shape[0], dtype=np.uint32))
        out_t = qd.ndarray(dtype=dt, shape=n)
        out_t.from_numpy(np.zeros(n, dtype=np_dt[dt]))

        @qd.kernel
        def kmax(arr: qd.types.ndarray(ndim=1), out: qd.types.ndarray(ndim=1), scratch: qd.types.ndarray(ndim=1), n: qd.i32):
            exclusive_scan_max(arr, out, scratch, n, dt, LOG256)

        kmax(arr_t, out_t, scratch, n)
        qd.sync()
        got = out_t.to_numpy()
        want = np.concatenate([[max_identity[dt]], np.maximum.accumulate(arr)[:-1]]).astype(np_dt[dt])
        gate(f"scan_max {dt} n={n}", minmax_close(got, want, dt), f"maxerr={minmax_maxerr(got, want):.3g}")

# -------------------------------------------------- sort (u32 keys + f32 values, LSD-path signature)
# Keys drawn from a narrow range so duplicate keys are guaranteed: with ties present, matching numpy's
# argsort(kind="stable") permutation *and* the value-follow pins the stable tie-break contract, not just the
# key multiset. tmp_* are same-shape zero working buffers; scratch is the caller-owned zeroed u32 buffer.
for n in (1, 257, 5000):
    rng = np.random.default_rng(n + 91)
    keys = rng.integers(0, max(2, n // 4), n).astype(np.uint32)
    values = (rng.standard_normal(n) * 10).astype(np.float32)
    keys_t = qd.ndarray(dtype=qd.u32, shape=n)
    keys_t.from_numpy(keys)
    tmp_keys_t = qd.ndarray(dtype=qd.u32, shape=n)
    tmp_keys_t.from_numpy(np.zeros(n, dtype=np.uint32))
    values_t = qd.ndarray(dtype=qd.f32, shape=n)
    values_t.from_numpy(values)
    tmp_values_t = qd.ndarray(dtype=qd.f32, shape=n)
    tmp_values_t.from_numpy(np.zeros(n, dtype=np.float32))
    scratch = qd.ndarray(dtype=qd.u32, shape=256)
    scratch.from_numpy(np.zeros(256, dtype=np.uint32))

    @qd.kernel
    def kso(keys: qd.types.ndarray(ndim=1), tmp_keys: qd.types.ndarray(ndim=1), values: qd.types.ndarray(ndim=1),
            tmp_values: qd.types.ndarray(ndim=1), scratch: qd.types.ndarray(ndim=1), n: qd.i32):
        sort(keys, tmp_keys, values, tmp_values, scratch, n, qd.u32, True, 32, 3)

    kso(keys_t, tmp_keys_t, values_t, tmp_values_t, scratch, n)
    qd.sync()
    order = np.argsort(keys, kind="stable")
    want_keys, want_values = keys[order], values[order]
    got_keys, got_values = keys_t.to_numpy(), values_t.to_numpy()
    n_ties = n - len(np.unique(keys))
    ok = np.array_equal(got_keys, want_keys) and np.array_equal(got_values, want_values)
    gate(f"sort u32+f32 n={n}", ok, f"ties={n_ties}")

# -------------------------------------------------- is_cpu predicate (post-init)
gate("is_cpu_predicate", is_cpu() is True, f"{is_cpu()}")

print(f"\nALGORITHMS CPU GATE: {'ALL PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
