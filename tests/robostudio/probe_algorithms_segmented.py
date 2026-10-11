"""Gate probe (C4): qd.algorithms.sort_segmented — segmented stable key-value sort.

Reference = per-segment numpy stable argsort, compared bitwise on keys (NaN-safe) and exactly on
values. Segments are the contiguous ranges [ids[j], ids[j+1]) over num_segments+1 start offsets
with the end sentinel ids[num_segments] == n. Cases (each on the target backend):
  - variable-length segments incl. EMPTY segments (duplicate offsets), single segment, and
    1-element segments
  - key dtypes u32 / i32 / f32 (f32 negative keys and NaN ordering bitwise); u64/i64/f64 on cpu
    (Metal has no 64-bit)
  - has_values=False (placeholder value slots)
  - boundary n in {1, 2, 257, 5000}; duplicates via narrow key ranges
  - SAP-broadphase-shaped throughput pin (fixed-size per-world segments), WARN-only: absolute ms
    pins are machine-variance-prone (see ledger "Readback gate-4 perf pins"), so the verdict
    keys on correctness and the number is informational.
Run: python probe_algorithms_segmented.py [metal|cpu]   (exit 0 = pass)
Ledger: mujocoqd/algorithms-2 C4.
"""
import sys
import time

import numpy as np

import quadrants as qd
from quadrants.algorithms import sort_segmented, sort_segmented_scratch_slots

ARCH = sys.argv[1] if len(sys.argv) > 1 else "metal"
qd.init(arch=qd.metal if ARCH == "metal" else qd.cpu)

fails = []
warns = []


def gate(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL':4}] {name} {detail}")
    if not ok:
        fails.append(name)


D = 3  # log256_max_n
u32_1d = qd.types.ndarray(qd.u32, ndim=1)
i32_1d = qd.types.ndarray(qd.i32, ndim=1)
f32_1d = qd.types.ndarray(qd.f32, ndim=1)
u64_1d = qd.types.ndarray(qd.u64, ndim=1)
i64_1d = qd.types.ndarray(qd.i64, ndim=1)
f64_1d = qd.types.ndarray(qd.f64, ndim=1)
i32_0d = qd.types.ndarray(qd.i32, ndim=0)

ND = {qd.u32: (u32_1d, np.uint32), qd.i32: (i32_1d, np.int32), qd.f32: (f32_1d, np.float32),
      qd.u64: (u64_1d, np.uint64), qd.i64: (i64_1d, np.int64), qd.f64: (f64_1d, np.float64)}

# explicit kernel definitions (one per (dtype, has_values) used below; kernels cannot be
# built via exec — source introspection)
if True:
    @qd.kernel
    def k_u32(keys: u32_1d, tmp_keys: u32_1d, values: f32_1d, tmp_values: f32_1d,
              seg: i32_1d, tmp_seg: i32_1d, scratch: u32_1d, n: i32_0d, nseg: i32_0d):
        sort_segmented(keys, tmp_keys, values, tmp_values, seg, tmp_seg, scratch, n, nseg,
                       qd.u32, True, 32, D)

    @qd.kernel
    def k_i32(keys: i32_1d, tmp_keys: i32_1d, values: f32_1d, tmp_values: f32_1d,
              seg: i32_1d, tmp_seg: i32_1d, scratch: u32_1d, n: i32_0d, nseg: i32_0d):
        sort_segmented(keys, tmp_keys, values, tmp_values, seg, tmp_seg, scratch, n, nseg,
                       qd.i32, True, 32, D)

    @qd.kernel
    def k_f32(keys: f32_1d, tmp_keys: f32_1d, values: f32_1d, tmp_values: f32_1d,
              seg: i32_1d, tmp_seg: i32_1d, scratch: u32_1d, n: i32_0d, nseg: i32_0d):
        sort_segmented(keys, tmp_keys, values, tmp_values, seg, tmp_seg, scratch, n, nseg,
                       qd.f32, True, 32, D)

    @qd.kernel
    def k_u32_nv(keys: u32_1d, tmp_keys: u32_1d, values: u32_1d, tmp_values: u32_1d,
                 seg: i32_1d, tmp_seg: i32_1d, scratch: u32_1d, n: i32_0d, nseg: i32_0d):
        sort_segmented(keys, tmp_keys, values, tmp_values, seg, tmp_seg, scratch, n, nseg,
                       qd.u32, False, 32, D)

    @qd.kernel
    def k_u64(keys: u64_1d, tmp_keys: u64_1d, values: f32_1d, tmp_values: f32_1d,
              seg: i32_1d, tmp_seg: i32_1d, scratch: u32_1d, n: i32_0d, nseg: i32_0d):
        sort_segmented(keys, tmp_keys, values, tmp_values, seg, tmp_seg, scratch, n, nseg,
                       qd.u64, True, 64, D)

    @qd.kernel
    def k_i64(keys: i64_1d, tmp_keys: i64_1d, values: f32_1d, tmp_values: f32_1d,
              seg: i32_1d, tmp_seg: i32_1d, scratch: u32_1d, n: i32_0d, nseg: i32_0d):
        sort_segmented(keys, tmp_keys, values, tmp_values, seg, tmp_seg, scratch, n, nseg,
                       qd.i64, True, 64, D)

    @qd.kernel
    def k_f64(keys: f64_1d, tmp_keys: f64_1d, values: f32_1d, tmp_values: f32_1d,
              seg: i32_1d, tmp_seg: i32_1d, scratch: u32_1d, n: i32_0d, nseg: i32_0d):
        sort_segmented(keys, tmp_keys, values, tmp_values, seg, tmp_seg, scratch, n, nseg,
                       qd.f64, True, 64, D)

KMAP = {(qd.u32, True): k_u32, (qd.i32, True): k_i32, (qd.f32, True): k_f32,
        (qd.u32, False): k_u32_nv, (qd.u64, True): k_u64, (qd.i64, True): k_i64, (qd.f64, True): k_f64}

UINT = {qd.u32: np.uint32, qd.i32: np.uint32, qd.f32: np.uint32,
        qd.u64: np.uint64, qd.i64: np.uint64, qd.f64: np.uint64}


def make_ids(n, nseg, rng, empty):
    """num_segments+1 start offsets with sentinel; `empty` injects duplicate offsets."""
    if nseg <= 1:
        return np.array([0, n], dtype=np.int32)
    starts = sorted(rng.choice(np.arange(1, n), size=nseg - 1, replace=False).tolist())
    ids = [0] + starts + [n]
    if empty:
        ids[-2] = ids[-3]  # duplicate offsets -> an empty segment before the sentinel
    return np.array(ids, dtype=np.int32)


def ref_sort(keys, vals, ids):
    out_k = keys.copy()
    out_v = vals.copy()
    for j in range(len(ids) - 1):
        lo, hi = int(ids[j]), int(ids[j + 1])
        o = np.argsort(keys[lo:hi], kind="stable")
        out_k[lo:hi] = keys[lo:hi][o]
        out_v[lo:hi] = vals[lo:hi][o]
    return out_k, out_v


def trial(name, dt, hv, n, ids, rng, values=None):
    npt = ND[dt][1]
    if dt in (qd.f32, qd.f64):
        keys = (rng.standard_normal(n) * 100).astype(npt)
        if n >= 4000:
            keys[:: n // 32] = npt("nan")
    elif dt == qd.u32 and values is None:
        keys = (rng.integers(0, 250, n) - 125).astype(npt)  # high bit patterns (sign-flipped negatives)
    else:
        keys = (rng.standard_normal(n) * 1e6).astype(np.int64).astype(npt)
    if values is None:
        vals = (rng.standard_normal(n) * 10).astype(np.float32)
    else:
        vals = values
    wk, wv = ref_sort(keys, vals, ids)
    kt = qd.ndarray(dtype=dt, shape=n); kt.from_numpy(keys)
    ktt = qd.ndarray(dtype=dt, shape=n); ktt.from_numpy(np.zeros(n, npt))
    if hv:
        vt = qd.ndarray(dtype=qd.f32, shape=n); vt.from_numpy(vals)
        vtt = qd.ndarray(dtype=qd.f32, shape=n); vtt.from_numpy(np.zeros(n, np.float32))
    else:
        vt = qd.ndarray(dtype=dt, shape=n); vt.from_numpy(keys)
        vtt = qd.ndarray(dtype=dt, shape=n); vtt.from_numpy(np.zeros(n, npt))
    st = qd.ndarray(dtype=qd.i32, shape=len(ids)); st.from_numpy(ids)
    stt = qd.ndarray(dtype=qd.i32, shape=n); stt.from_numpy(np.zeros(n, np.int32))
    sc = qd.ndarray(dtype=qd.u32, shape=sort_segmented_scratch_slots(n, D))
    sc.from_numpy(np.zeros(sc.shape[0], dtype=np.uint32))
    n0 = qd.ndarray(dtype=qd.i32, shape=()); n0.fill(n)
    ns0 = qd.ndarray(dtype=qd.i32, shape=()); ns0.fill(len(ids) - 1)
    KMAP[(dt, hv)](kt, ktt, vt, vtt, st, stt, sc, n0, ns0)
    qd.sync()
    gk, gv = kt.to_numpy(), vt.to_numpy()
    ok = np.array_equal(gk.view(UINT[dt]), wk.view(UINT[dt]))
    if hv:
        ok &= np.array_equal(gv, wv)
    gate(name, ok, f"n={n} nseg={len(ids)-1}")
    return ok


rng = np.random.default_rng(101)
trial("seg u32 var n=5000", qd.u32, True, 5000, make_ids(5000, 32, rng, False), rng)
trial("seg u32 empty-seg n=1000", qd.u32, True, 1000, make_ids(1000, 40, rng, True), rng)
trial("seg u32 single-seg n=257", qd.u32, True, 257, np.array([0, 257], dtype=np.int32), rng)
trial("seg u32 1-elem-segs n=64", qd.u32, True, 64, np.arange(65, dtype=np.int32), rng)
trial("seg u32 n=2", qd.u32, True, 2, np.array([0, 1, 2], dtype=np.int32), rng)
trial("seg u32 n=1", qd.u32, True, 1, np.array([0, 1], dtype=np.int32), rng)
trial("seg i32 neg n=2000", qd.i32, True, 2000, make_ids(2000, 16, rng, False), rng)
trial("seg f32 neg+nan n=4096", qd.f32, True, 4096, make_ids(4096, 16, rng, False), rng)
trial("seg u32 no-vals n=1000", qd.u32, False, 1000, make_ids(1000, 8, rng, False), rng)
if ARCH == "cpu":
    trial("seg u64 n=1000", qd.u64, True, 1000, make_ids(1000, 8, rng, False), rng)
    trial("seg i64 n=1000", qd.i64, True, 1000, make_ids(1000, 8, rng, False), rng)
    trial("seg f64 n=1000", qd.f64, True, 1000, make_ids(1000, 8, rng, False), rng)

# ----------------------------------------------- SAP-broadphase-shaped throughput pin (WARN-only)
# mujoco SAP shape: per-world fixed-size geom projections (128 worlds x 256 geoms), values =
# geom indices. Absolute-ms pins are machine-variance-prone (gate-4 lesson), so this is a
# completion + informational timing check, not a verdict gate.
NW, NG = 128, 256
n = NW * NG
ids = (np.arange(NW + 1, dtype=np.int32) * NG)
rng = np.random.default_rng(7)
keys = rng.integers(0, 2**31, n).astype(np.uint32)
vals = np.arange(n, dtype=np.int32).astype(np.float32)
kt = qd.ndarray(dtype=qd.u32, shape=n); kt.from_numpy(keys)
ktt = qd.ndarray(dtype=qd.u32, shape=n); ktt.from_numpy(np.zeros(n, np.uint32))
vt = qd.ndarray(dtype=qd.f32, shape=n); vt.from_numpy(vals)
vtt = qd.ndarray(dtype=qd.f32, shape=n); vtt.from_numpy(np.zeros(n, np.float32))
st = qd.ndarray(dtype=qd.i32, shape=len(ids)); st.from_numpy(ids)
stt = qd.ndarray(dtype=qd.i32, shape=n); stt.from_numpy(np.zeros(n, np.int32))
sc = qd.ndarray(dtype=qd.u32, shape=sort_segmented_scratch_slots(n, D))
sc.from_numpy(np.zeros(sc.shape[0], dtype=np.uint32))
n0 = qd.ndarray(dtype=qd.i32, shape=()); n0.fill(n)
ns0 = qd.ndarray(dtype=qd.i32, shape=()); ns0.fill(NW)
k_u32(kt, ktt, vt, vtt, st, stt, sc, n0, ns0)
qd.sync()
want_k, want_v = ref_sort(keys, vals, ids)
ok = np.array_equal(kt.to_numpy(), want_k) and np.array_equal(vt.to_numpy(), want_v)
gate("seg sap-shape correctness n=32768", ok)
R = 10
t0 = time.perf_counter()
for _ in range(R):
    k_u32(kt, ktt, vt, vtt, st, stt, sc, n0, ns0)
qd.sync()
ms = (time.perf_counter() - t0) / R * 1e3
thr = n / (ms / 1e3) / 1e6
print(f"[WARN] seg sap-shape throughput: {ms:.3f} ms/sort ({thr:.1f} Mkeys/s) — informational, not a gate")
warns.append("sap-throughput")

print(f"\nSEGMENTED SORT GATE [{ARCH}]: {'ALL PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
