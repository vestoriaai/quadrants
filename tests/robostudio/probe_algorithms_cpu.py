"""Gate probe (C5): qd.algorithms CPU-backend serial paths — scan/reduce/select vs numpy.

The CPU path dispatches at kernel-compile time via qd.static(is_cpu()); this probe runs the
CPU backend exclusively (the GPU staircases keep their own upstream gates). Cases:
  - exclusive_scan_add: i32 / u32 / f32, n in {1, 5, 257, 5000}  (257 crosses BLOCK_DIM=256)
  - reduce_add: same dtypes/n
  - select: i32 payload, random 0/1 flags, several densities
  - exclusive_scan_min / exclusive_scan_max: i32 / f32, n in {5, 257, 5000}
  - sort (serial LSD radix): u32 keys + f32 values (stable tie-break), n in {1, 257, 5000}
  - sort extension (mujocoqd/algorithms-2 SIGBUS repair): negative i32 keys, f32 NaN (numpy
    order, bitwise compare), duplicates, boundary n in {1, 2, 3, 17, 255, 256, 257, 511, 512,
    513, 1000, 4096}, end_bit in {8, 16, 24, 32}, has_values=False, 0-d ndarray n form, and
    64-bit keys (CPU-only luxury; Metal has no 64-bit). The n=1000 negative-key shape
    deterministically corrupted under the racy pre-fix form (a "serial" kernel's second-and-
    later top-level loops defaulting to parallel on the CPU thread pool) and is kept as the
    regression pin for the per-loop loop_config root cause.
  - expected-error probes (fresh subprocess each, one-init-per-process): the GPU block-tier
    staircase cannot lower on the CPU backend — block.sync raises ValueError at trace time,
    block/subgroup collectives see subgroup size 0 (AssertionError), and a bare SharedArray
    lowers to an addrspace(3) LLVM global with no CPU backing, failing kernel JIT
    ("not found" RuntimeError at launch).
Run: python probe_algorithms_cpu.py cpu   (exit 0 = pass)
Ledger: mujocoqd/compiler-line C5 + mujocoqd/algorithms-2.
"""
import os
import subprocess
import sys
import tempfile

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
    sort_scratch_slots,
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
# key multiset. tmp_* are same-shape zero working buffers; scratch is the caller-owned zeroed u32 buffer
# sized via sort_scratch_slots (the contract the serial radix also honours with its scratch[0:256] counters).
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
    scratch = qd.ndarray(dtype=qd.u32, shape=sort_scratch_slots(n, LOG256))
    scratch.from_numpy(np.zeros(scratch.shape[0], dtype=np.uint32))

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

# -------------------------------------------------- sort extension (mujocoqd/algorithms-2 SIGBUS repair)
# Serial LSD radix vs numpy stable argsort, bitwise on keys (NaN lanes compare equal bitwise, which
# positional == cannot). Shapes: the pre-fix racy form deterministically corrupted negative keys at
# n=1000 (parallelized counting loops) — that exact shape is the regression pin.
u32_1d = qd.types.ndarray(qd.u32, ndim=1)
i32_1d = qd.types.ndarray(qd.i32, ndim=1)
f32_1d = qd.types.ndarray(qd.f32, ndim=1)
u64_1d = qd.types.ndarray(qd.u64, ndim=1)
i64_1d = qd.types.ndarray(qd.i64, ndim=1)
f64_1d = qd.types.ndarray(qd.f64, ndim=1)
i32_0d = qd.types.ndarray(qd.i32, ndim=0)


@qd.kernel
def ks_u32(keys: u32_1d, tmp_keys: u32_1d, values: f32_1d, tmp_values: f32_1d, scratch: u32_1d, n: i32_0d):
    sort(keys, tmp_keys, values, tmp_values, scratch, n, qd.u32, True, 32, 3)


@qd.kernel
def ks_i32(keys: i32_1d, tmp_keys: i32_1d, values: f32_1d, tmp_values: f32_1d, scratch: u32_1d, n: i32_0d):
    sort(keys, tmp_keys, values, tmp_values, scratch, n, qd.i32, True, 32, 3)


@qd.kernel
def ks_f32(keys: f32_1d, tmp_keys: f32_1d, values: f32_1d, tmp_values: f32_1d, scratch: u32_1d, n: i32_0d):
    sort(keys, tmp_keys, values, tmp_values, scratch, n, qd.f32, True, 32, 3)


@qd.kernel
def ks_u64(keys: u64_1d, tmp_keys: u64_1d, values: f32_1d, tmp_values: f32_1d, scratch: u32_1d, n: i32_0d):
    sort(keys, tmp_keys, values, tmp_values, scratch, n, qd.u64, True, 64, 3)


@qd.kernel
def ks_i64(keys: i64_1d, tmp_keys: i64_1d, values: f32_1d, tmp_values: f32_1d, scratch: u32_1d, n: i32_0d):
    sort(keys, tmp_keys, values, tmp_values, scratch, n, qd.i64, True, 64, 3)


@qd.kernel
def ks_f64(keys: f64_1d, tmp_keys: f64_1d, values: f32_1d, tmp_values: f32_1d, scratch: u32_1d, n: i32_0d):
    sort(keys, tmp_keys, values, tmp_values, scratch, n, qd.f64, True, 64, 3)


@qd.kernel
def ks_u32_eb8(keys: u32_1d, tmp_keys: u32_1d, values: f32_1d, tmp_values: f32_1d, scratch: u32_1d, n: i32_0d):
    sort(keys, tmp_keys, values, tmp_values, scratch, n, qd.u32, True, 8, 3)


@qd.kernel
def ks_u32_eb16(keys: u32_1d, tmp_keys: u32_1d, values: f32_1d, tmp_values: f32_1d, scratch: u32_1d, n: i32_0d):
    sort(keys, tmp_keys, values, tmp_values, scratch, n, qd.u32, True, 16, 3)


@qd.kernel
def ks_u32_eb24(keys: u32_1d, tmp_keys: u32_1d, values: f32_1d, tmp_values: f32_1d, scratch: u32_1d, n: i32_0d):
    sort(keys, tmp_keys, values, tmp_values, scratch, n, qd.u32, True, 24, 3)


@qd.kernel
def ks_u32_novals(keys: u32_1d, tmp_keys: u32_1d, values: u32_1d, tmp_values: u32_1d, scratch: u32_1d, n: i32_0d):
    sort(keys, tmp_keys, values, tmp_values, scratch, n, qd.u32, False, 32, 3)


@qd.kernel
def ks_u32_n0d(keys: u32_1d, tmp_keys: u32_1d, values: f32_1d, tmp_values: f32_1d, scratch: u32_1d, n: i32_0d):
    sort(keys, tmp_keys, values, tmp_values, scratch, n, qd.u32, True, 32, 3)


def sort_bits_close(got, want, dt):
    """Bitwise key equality (NaN-safe: compare the raw bit patterns)."""
    np_int = {qd.u32: np.uint32, qd.i32: np.uint32, qd.f32: np.uint32,
              qd.u64: np.uint64, qd.i64: np.uint64, qd.f64: np.uint64}[dt]
    return np.array_equal(got.view(np_int), want.view(np_int))


case_id = 0
for dt, kk, npt in ((qd.u32, ks_u32, np.uint32), (qd.i32, ks_i32, np.int32), (qd.f32, ks_f32, np.float32),
                    (qd.u64, ks_u64, np.uint64), (qd.i64, ks_i64, np.int64), (qd.f64, ks_f64, np.float64)):
    for n in (1, 2, 3, 17, 255, 256, 257, 511, 512, 513, 1000, 4096):
        case_id += 1
        rng = np.random.default_rng(n * 31 + case_id)
        if dt in (qd.f32, qd.f64):
            keys = (rng.standard_normal(n) * 100).astype(npt)
            if n >= 4096:
                keys[:: n // 16] = npt("nan")
        else:
            lo, hi = (np.iinfo(npt).min, np.iinfo(npt).max) if npt != np.uint64 else (0, 2**63)
            keys = ((rng.standard_normal(n) * 1e9).astype(np.int64).clip(lo, hi - 1)).astype(npt)
        if dt == qd.u32 and n == 1000:
            # the pre-fix corruption pin: keys with high bit patterns (sign-flipped negatives)
            keys = ((rng.integers(0, 250, n) - 125)).astype(npt)
        values = (rng.standard_normal(n) * 10).astype(np.float32)
        keys_t = qd.ndarray(dtype=dt, shape=n); keys_t.from_numpy(keys)
        tmp_keys_t = qd.ndarray(dtype=dt, shape=n); tmp_keys_t.from_numpy(np.zeros(n, npt))
        values_t = qd.ndarray(dtype=qd.f32, shape=n); values_t.from_numpy(values)
        tmp_values_t = qd.ndarray(dtype=qd.f32, shape=n); tmp_values_t.from_numpy(np.zeros(n, np.float32))
        scratch = qd.ndarray(dtype=qd.u32, shape=sort_scratch_slots(n, LOG256))
        scratch.from_numpy(np.zeros(scratch.shape[0], dtype=np.uint32))
        n0 = qd.ndarray(dtype=qd.i32, shape=()); n0.fill(n)
        kk(keys_t, tmp_keys_t, values_t, tmp_values_t, scratch, n0)
        qd.sync()
        order = np.argsort(keys, kind="stable")
        ok = sort_bits_close(keys_t.to_numpy(), keys[order], dt) and np.array_equal(values_t.to_numpy(), values[order])
        gate(f"sort_radix {np.dtype(npt).name} n={n}", ok)

# end_bit variants: reference = stable argsort of the masked key bits (radix over low bits only)
for eb, kk in ((8, ks_u32_eb8), (16, ks_u32_eb16), (24, ks_u32_eb24)):
    n = 1000
    rng = np.random.default_rng(n + eb)
    keys = rng.integers(0, 2**32 - 1, n).astype(np.uint32)
    values = (rng.standard_normal(n) * 10).astype(np.float32)
    keys_t = qd.ndarray(dtype=qd.u32, shape=n); keys_t.from_numpy(keys)
    tmp_keys_t = qd.ndarray(dtype=qd.u32, shape=n); tmp_keys_t.from_numpy(np.zeros(n, np.uint32))
    values_t = qd.ndarray(dtype=qd.f32, shape=n); values_t.from_numpy(values)
    tmp_values_t = qd.ndarray(dtype=qd.f32, shape=n); tmp_values_t.from_numpy(np.zeros(n, np.float32))
    scratch = qd.ndarray(dtype=qd.u32, shape=sort_scratch_slots(n, LOG256))
    scratch.from_numpy(np.zeros(scratch.shape[0], dtype=np.uint32))
    n0 = qd.ndarray(dtype=qd.i32, shape=()); n0.fill(n)
    kk(keys_t, tmp_keys_t, values_t, tmp_values_t, scratch, n0)
    qd.sync()
    mask = np.uint32((1 << eb) - 1)
    order = np.argsort(keys & mask, kind="stable")
    ok = np.array_equal(keys_t.to_numpy(), keys[order]) and np.array_equal(values_t.to_numpy(), values[order])
    gate(f"sort_radix u32 end_bit={eb}", ok)

# has_values=False (keys-only; placeholders in the value slots)
n = 513
rng = np.random.default_rng(n)
keys = rng.integers(0, 2**32 - 1, n).astype(np.uint32)
keys_t = qd.ndarray(dtype=qd.u32, shape=n); keys_t.from_numpy(keys)
tmp_keys_t = qd.ndarray(dtype=qd.u32, shape=n); tmp_keys_t.from_numpy(np.zeros(n, np.uint32))
scratch = qd.ndarray(dtype=qd.u32, shape=sort_scratch_slots(n, LOG256))
scratch.from_numpy(np.zeros(scratch.shape[0], dtype=np.uint32))
n0 = qd.ndarray(dtype=qd.i32, shape=()); n0.fill(n)
ks_u32_novals(keys_t, tmp_keys_t, keys_t, tmp_keys_t, scratch, n0)
qd.sync()
order = np.argsort(keys, kind="stable")
gate("sort_radix u32 has_values=False", np.array_equal(keys_t.to_numpy(), keys[order]))

# 0-d ndarray n (the GPU-contract count form) on the CPU path
n = 300
rng = np.random.default_rng(n)
keys = rng.integers(0, 1000, n).astype(np.uint32)
values = (rng.standard_normal(n) * 10).astype(np.float32)
keys_t = qd.ndarray(dtype=qd.u32, shape=n); keys_t.from_numpy(keys)
tmp_keys_t = qd.ndarray(dtype=qd.u32, shape=n); tmp_keys_t.from_numpy(np.zeros(n, np.uint32))
values_t = qd.ndarray(dtype=qd.f32, shape=n); values_t.from_numpy(values)
tmp_values_t = qd.ndarray(dtype=qd.f32, shape=n); tmp_values_t.from_numpy(np.zeros(n, np.float32))
scratch = qd.ndarray(dtype=qd.u32, shape=sort_scratch_slots(n, LOG256))
scratch.from_numpy(np.zeros(scratch.shape[0], dtype=np.uint32))
n0 = qd.ndarray(dtype=qd.i32, shape=()); n0.fill(n)
ks_u32_n0d(keys_t, tmp_keys_t, values_t, tmp_values_t, scratch, n0)
qd.sync()
order = np.argsort(keys, kind="stable")
gate("sort_radix u32 0-d n", np.array_equal(keys_t.to_numpy(), keys[order])
     and np.array_equal(values_t.to_numpy(), values[order]))

# -------------------------------------------------- expected-error probes: GPU block-tier staircase on CPU
# The GPU radix staircase is block-tier (SharedArray + block barriers + subgroup collectives); each
# tier of the CPU backend's inability to lower it is pinned here in a fresh subprocess (the CPU
# backend is one-init-per-process, and every layer fails loudly rather than silently).
_PROBES = {
    "staircase_block_sync_valueerror": (
        """
import quadrants as qd
from quadrants.lang.kernel_impl import kernel
from quadrants.lang.misc import loop_config
from quadrants.lang.simt import block as _block
qd.init(arch=qd.cpu)

@kernel
def k(n: qd.i32):
    loop_config(block_dim=256)
    for i in range(n):
        _block.sync()

try:
    k(4)
    print("MARKER_NO_ERROR")
except BaseException as e:  # the frontend wraps the ValueError in QuadrantsCompilationError
    print("MARKER_BARRIER_UNSUPPORTED" if "block.sync is not supported" in str(e) else "MARKER_OTHER")
""",
        "MARKER_BARRIER_UNSUPPORTED",
    ),
    "staircase_subgroup_assert": (
        """
import quadrants as qd
from quadrants.lang.kernel_impl import kernel
from quadrants.lang.misc import loop_config
from quadrants.lang.simt import block as _block
from quadrants.types.primitive_types import u32
qd.init(arch=qd.cpu)
u32_1d = qd.types.ndarray(qd.u32, ndim=1)

@kernel
def k(out: u32_1d):
    loop_config(block_dim=256)
    for i in range(256):
        v = u32(1)
        agg = _block.reduce_add(v, 256, u32)
        if i == 0:
            out[i] = agg

t = qd.ndarray(qd.u32, shape=8)
try:
    k(t)
    print("MARKER_NO_ERROR")
except BaseException as e:  # AssertionError surfaces wrapped as a compilation error
    msg = str(e)
    ok = ("subgroup size 0" in msg) or ("subgroup size must be 32 or 64" in msg)
    print("MARKER_SUBGROUP_UNSUPPORTED" if ok else "MARKER_OTHER")
""",
        "MARKER_SUBGROUP_UNSUPPORTED",
    ),
    "staircase_shared_array_jit_failure": (
        """
import quadrants as qd
from quadrants.lang.kernel_impl import kernel
from quadrants.lang.misc import loop_config
from quadrants.lang.ops import bit_cast
from quadrants.lang.simt import block as _block
from quadrants.types.primitive_types import i32, u32
qd.init(arch=qd.cpu)
u32_1d = qd.types.ndarray(qd.u32, ndim=1)

@kernel
def k(out: u32_1d):
    loop_config(block_dim=256)
    for i in range(256):
        tid = i % 256
        sh = _block.SharedArray((256,), i32)
        sh[tid] = i32(255 - tid)
        out[i] = bit_cast(sh[255 - tid], u32)

t = qd.ndarray(qd.u32, shape=256)
try:
    k(t)
    print("MARKER_NO_ERROR")
except RuntimeError as e:
    print("MARKER_JIT_ERROR" if "not found" in str(e) else "MARKER_OTHER_ERROR")
""",
        "MARKER_JIT_ERROR",
    ),
}

for probe_name, (src, want_marker) in _PROBES.items():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=180,
                           env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")})
        got = want_marker in r.stdout
        if got:
            detail = "ok"
        else:
            out = r.stdout.strip().splitlines()
            err = r.stderr.strip().splitlines()
            detail = f"stdout tail: {out[-1]}" if out else (f"stderr tail: {err[-1]}" if err else "no output")
        gate(f"expected_error {probe_name}", got, detail)
    finally:
        os.unlink(path)

# -------------------------------------------------- is_cpu predicate (post-init)
gate("is_cpu_predicate", is_cpu() is True, f"{is_cpu()}")

print(f"\nALGORITHMS CPU GATE: {'ALL PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
