"""Gate probe (metal-i64 line): i64 type-path characterization + perf, Metal vs CPU.

Background: mujoco_qd collision_flex._compute_filter_key (L1196) emits i64 sort keys
(qd.i64 ndarray, group_key << 32 | spatial_key packing) fed to qd.algorithms.sort.
The skill's standing note said "Metal has no i64"; this probe pins the actual
per-form capability matrix and the cost of the forms that DO work.

Correctness matrix (asserted; exit 1 on any non-xfailed failure):
  t01  i64 ndarray load/store roundtrip (values beyond i32, incl INT64_MAX/MIN)
  t02  i64 arithmetic (* + -)
  t03  (g << 32) | s packing (the _compute_filter_key key form)
  t04  bitwise & | ^
  t05  i64 comparisons driving control flow
  t06  casts i32->i64 / f32->i64 / bool->i64 + mixed i64+i32 arithmetic
  t07  qd.math.clamp on i64
  t08  INT64_MAX literal store (the sentinel form)
  t09  bit_cast i64<->u64 (incl the radix twiddle form)
  t10  qd.algorithms.sort with i64 keys (plain radix sort; the filter_key path)
       - n=10000, keys+values exact-stable vs numpy reference
  t11  qd.algorithms.sort_segmented with i64 keys
       - CPU: host path, must pass.
       - Metal: must pass since the _seg_scatter key_width branch landed
         (mujocoqd/metal-i64); before that fix the GPU path died loudly in
         spirv_codegen bit_cast (bit_cast(keys_in[i], u32) was width-hard-
         wired). SEGMENTED_I64_XFAIL=True re-arms the loud-rejection-if-
         broken acceptance for backends without int64 caps; keep False here.

Perf (informational, WARN-only; run on an idle GPU, see _gpu_busy.sh):
  p1  i64 vs i32 ndarray copy, N=8M          -> GB/s + slowdown
  p2  i64 (g<<32|s) pack vs i32 (g<<16|s)    -> GB/s + slowdown
  p3  plain sort i64 vs i32 keys, n=8M       -> ms + slowdown
  slowdown > 4x prints a HOT-PATH ALERT line.

Run: python probe_metal_i64.py [metal|cpu]   (exit 0 = pass)
"""
import sys
import time

import numpy as np

import quadrants as qd
from quadrants.algorithms import sort, sort_scratch_slots, sort_segmented, sort_segmented_scratch_slots

ARCH = sys.argv[1] if len(sys.argv) > 1 else "metal"
METAL = ARCH == "metal"
qd.init(arch=qd.metal if METAL else qd.cpu)

# Hard gate since the _seg_scatter key_width branch (mujocoqd/metal-i64);
# True re-arms loud-rejection-accepted only for backends without int64 caps.
SEGMENTED_I64_XFAIL = False
PERF_ALERT_SLOWDOWN = 4.0

fails = []
xfails = []


def gate(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag:4}] {name} {detail}")
    if not ok:
        fails.append(name)


def run_kernel(name, k, *args):
    """Compile+launch+sync; returns (ok, error-tail)."""
    try:
        k(*args)
        qd.sync()
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e).replace("\n", " ")[-200:]


i64_1d = qd.types.ndarray(qd.i64, ndim=1)
u64_1d = qd.types.ndarray(qd.u64, ndim=1)
i32_1d = qd.types.ndarray(qd.i32, ndim=1)
u32_1d = qd.types.ndarray(qd.u32, ndim=1)
f32_1d = qd.types.ndarray(qd.f32, ndim=1)
i32_0d = qd.types.ndarray(qd.i32, ndim=0)

N = 8
KEYS = np.array([123456789012345, -987654321098, 4294967296 + 7, 9223372036854775807,
                 -9223372036854775808, 2147483648, -2147483649, 5], dtype=np.int64)


def _assert_eq(name, actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    ok = actual.shape == expected.shape and np.array_equal(actual, expected)
    if ok:
        gate(name, True)
    else:
        idx = np.flatnonzero(actual != expected)[:3] if actual.shape == expected.shape else "shape"
        gate(name, False, f"exp={np.asarray(expected)[idx]} got={actual[idx]}")


# ---- t01: load/store roundtrip ----
def t01():
    a = qd.ndarray(dtype=qd.i64, shape=N)
    b = qd.ndarray(dtype=qd.i64, shape=N)
    a.from_numpy(KEYS)

    @qd.kernel
    def k(src: i64_1d, dst: i64_1d):
        for i in range(N):
            dst[i] = src[i]

    ok, err = run_kernel("t01 launch", k, a, b)
    if not ok:
        gate("t01 i64 load/store roundtrip", False, err)
        return
    _assert_eq("t01 i64 load/store roundtrip", b.to_numpy(), KEYS)


t01()


# ---- t02: arithmetic ----
_x = np.array([2**40, 2**33, 10**12, -2**35, 3, 1 << 40, 999999999999, -1], dtype=np.int64)
_y = np.array([3, 5, -7, 11, 1 << 41, 2, 123456789, 2**40], dtype=np.int64)
_z = np.array([17, -3, 2**34, 5, 9, -11, 2**38, 2**41], dtype=np.int64)


def t02():
    a = qd.ndarray(dtype=qd.i64, shape=N)
    b = qd.ndarray(dtype=qd.i64, shape=N)
    c = qd.ndarray(dtype=qd.i64, shape=N)
    out = qd.ndarray(dtype=qd.i64, shape=N)
    a.from_numpy(_x)
    b.from_numpy(_y)
    c.from_numpy(_z)

    @qd.kernel
    def k(a: i64_1d, b: i64_1d, c: i64_1d, out: i64_1d):
        for i in range(N):
            out[i] = a[i] * b[i] + c[i] - b[i]

    ok, err = run_kernel("t02 launch", k, a, b, c, out)
    if not ok:
        gate("t02 i64 arith (* + -)", False, err)
        return
    _assert_eq("t02 i64 arith (* + -)", out.to_numpy(), _x * _y + _z - _y)


t02()


# ---- t03: (g << 32) | s ----
_g = np.array([0, 1, 2, 1000, 65535, 123456789, 7, 2147483], dtype=np.int64)
_s = np.array([0, 1, 2147483647, 123456789, 2**31, 7, 1 << 30, 55], dtype=np.int64)


def t03():
    g = qd.ndarray(dtype=qd.i64, shape=N)
    s = qd.ndarray(dtype=qd.i64, shape=N)
    out = qd.ndarray(dtype=qd.i64, shape=N)
    g.from_numpy(_g)
    s.from_numpy(_s)

    @qd.kernel
    def k(g: i64_1d, s: i64_1d, out: i64_1d):
        for i in range(N):
            out[i] = (g[i] << qd.i64(32)) | s[i]

    ok, err = run_kernel("t03 launch", k, g, s, out)
    if not ok:
        gate("t03 (g << 32) | s pack", False, err)
        return
    _assert_eq("t03 (g << 32) | s pack", out.to_numpy(), (_g << 32) | _s)


t03()


# ---- t04: bitwise ----
_av = np.array([0xF0F0F0F0F0F0, 123456789012345, -1, 2**40 + 15, 255, -256, 1 << 41, 7], dtype=np.int64)
_bv = np.array([0x0F0F0F0F0F0F, 987654321098, 0xFF, 15, 0xAA, 0xAA, 3, -9], dtype=np.int64)


def t04():
    a = qd.ndarray(dtype=qd.i64, shape=N)
    b = qd.ndarray(dtype=qd.i64, shape=N)
    out = qd.ndarray(dtype=qd.i64, shape=N)
    a.from_numpy(_av)
    b.from_numpy(_bv)

    @qd.kernel
    def k(a: i64_1d, b: i64_1d, out: i64_1d):
        for i in range(N):
            out[i] = (a[i] & b[i]) ^ (a[i] | b[i])

    ok, err = run_kernel("t04 launch", k, a, b, out)
    if not ok:
        gate("t04 i64 bitwise", False, err)
        return
    _assert_eq("t04 i64 bitwise", out.to_numpy(), (_av & _bv) ^ (_av | _bv))


t04()


# ---- t05: comparisons ----
_cv = np.array([-2**40, -5, 0, 5, 2**33, 2**40, -2**33, 9223372036854775806], dtype=np.int64)
_cp = np.array([2**33], dtype=np.int64)
_cexp = np.zeros(N, dtype=np.int64)
for _i in range(N):
    _cexp[_i] = 1 if _cv[_i] < _cp[0] else (2 if _cv[_i] == _cp[0] else (3 if _cv[_i] > 0 else 0))


def t05():
    a = qd.ndarray(dtype=qd.i64, shape=N)
    p = qd.ndarray(dtype=qd.i64, shape=1)
    out = qd.ndarray(dtype=qd.i32, shape=N)
    a.from_numpy(_cv)
    p.from_numpy(_cp)

    @qd.kernel
    def k(a: i64_1d, p: i64_1d, out: i32_1d):
        for i in range(N):
            r = 0
            if a[i] < p[0]:
                r = 1
            elif a[i] == p[0]:
                r = 2
            elif a[i] > 0:
                r = 3
            out[i] = r

    ok, err = run_kernel("t05 launch", k, a, p, out)
    if not ok:
        gate("t05 i64 compare control flow", False, err)
        return
    _assert_eq("t05 i64 compare control flow", out.to_numpy(), _cexp)


t05()


# ---- t06: casts + mix ----
_iv = np.array([7, -7, 2147483647, -2147483648, 123456789, 0, -1, 2**31 - 1], dtype=np.int32)
_fv = np.array([1.5, -2.25, 1000000.5, 0.0, 3.999, -3.999, 21474836.47, 0.5], dtype=np.float32)
_dv = np.array([2**40, -(2**41), 5, 2**35, -6, 2**62, 77, -2**45], dtype=np.int64)
_t6exp = np.array(
    [int(_iv[i]) + int(_fv[i] * 1000000.0) + (1 if _iv[i] >= 0 else 0) + int(_dv[i]) + 3
     for i in range(N)],
    dtype=np.int64)


def t06():
    ii = qd.ndarray(dtype=qd.i32, shape=N)
    ff = qd.ndarray(dtype=qd.f32, shape=N)
    big = qd.ndarray(dtype=qd.i64, shape=N)
    out = qd.ndarray(dtype=qd.i64, shape=N)
    ii.from_numpy(_iv)
    ff.from_numpy(_fv)
    big.from_numpy(_dv)

    @qd.kernel
    def k(ii: i32_1d, ff: f32_1d, big: i64_1d, out: i64_1d):
        for i in range(N):
            a = qd.i64(ii[i])
            b = qd.i64(ff[i] * 1000000.0)
            flag = 1 if ii[i] >= 0 else 0
            c = qd.i64(flag)
            out[i] = a + b + c + big[i] + 3

    ok, err = run_kernel("t06 launch", k, ii, ff, big, out)
    if not ok:
        gate("t06 casts (i32->i64, f32->i64, bool->i64) + mix", False, err)
        return
    _assert_eq("t06 casts (i32->i64, f32->i64, bool->i64) + mix", out.to_numpy(), _t6exp)


t06()


# ---- t07: clamp ----
_hv = np.array([-(2**40), -5, 0, 2147483647, 2**40, 999, -100000000, 2147483648], dtype=np.int64)


def t07():
    a = qd.ndarray(dtype=qd.i64, shape=N)
    out = qd.ndarray(dtype=qd.i64, shape=N)
    a.from_numpy(_hv)

    @qd.kernel
    def k(a: i64_1d, out: i64_1d):
        for i in range(N):
            out[i] = qd.math.clamp(a[i] + qd.i64(100000000), qd.i64(0), qd.i64(2147483647))

    ok, err = run_kernel("t07 launch", k, a, out)
    if not ok:
        gate("t07 clamp(i64, 0, 2147483647)", False, err)
        return
    _assert_eq("t07 clamp(i64, 0, 2147483647)", out.to_numpy(),
               np.clip(_hv + 100000000, 0, 2147483647).astype(np.int64))


t07()


# ---- t08: INT64_MAX literal ----
def t08():
    out = qd.ndarray(dtype=qd.i64, shape=N)

    @qd.kernel
    def k(out: i64_1d):
        for i in range(N):
            if i == 3:
                out[i] = qd.i64(9223372036854775807)
            else:
                out[i] = qd.i64(2147483647)

    ok, err = run_kernel("t08 launch", k, out)
    if not ok:
        gate("t08 INT64_MAX literal store", False, err)
        return
    exp = np.full(N, 2147483647, dtype=np.int64)
    exp[3] = 9223372036854775807
    _assert_eq("t08 INT64_MAX literal store", out.to_numpy(), exp)


t08()


# ---- t09: bit_cast forms ----
def t09():
    from quadrants.lang.ops import bit_cast
    a = qd.ndarray(dtype=qd.i64, shape=N)
    out = qd.ndarray(dtype=qd.i64, shape=N)
    a.from_numpy(KEYS)

    @qd.kernel
    def k(src: i64_1d, dst: i64_1d):
        for i in range(N):
            w = bit_cast(src[i], qd.u64)
            dst[i] = bit_cast(w ^ qd.u64(0x8000000000000000), qd.i64)

    ok, err = run_kernel("t09 launch", k, a, out)
    if not ok:
        gate("t09 bit_cast i64<->u64 (twiddle form)", False, err)
        return
    _assert_eq("t09 bit_cast i64<->u64 (twiddle form)", out.to_numpy(),
               np.bitwise_xor(KEYS.view(np.uint64), np.uint64(0x8000000000000000)).view(np.int64))


t09()


# ---- t10: plain sort with i64 keys ----
def t10():
    D = 3
    n = 10000
    rng = np.random.default_rng(11)
    keys_np = rng.integers(-2**62, 2**62, n).astype(np.int64)
    vals_np = np.arange(n, dtype=np.int32)
    keys = qd.ndarray(qd.i64, n)
    tmp_keys = qd.ndarray(qd.i64, n)
    values = qd.ndarray(qd.i32, n)
    tmp_values = qd.ndarray(qd.i32, n)
    scratch = qd.ndarray(qd.u32, sort_scratch_slots(n, D))
    keys.from_numpy(keys_np)
    values.from_numpy(vals_np)
    tmp_keys.from_numpy(np.zeros(n, np.int64))
    tmp_values.from_numpy(np.zeros(n, np.int32))
    scratch.from_numpy(np.zeros(scratch.shape[0], np.uint32))
    n0 = qd.ndarray(qd.i32, ())
    n0.fill(n)

    @qd.kernel
    def k(keys: i64_1d, tmp_keys: i64_1d, values: i32_1d, tmp_values: i32_1d,
          scratch: u32_1d, n: i32_0d):
        sort(keys, tmp_keys, values, tmp_values, scratch, n, qd.i64, True, 64, D)

    ok, err = run_kernel("t10 launch", k, keys, tmp_keys, values, tmp_values, scratch, n0)
    if not ok:
        gate("t10 plain sort i64 keys n=10000", False, err)
        return
    order = np.argsort(keys_np, kind="stable")
    _assert_eq("t10 plain sort i64 keys n=10000 (keys)", keys.to_numpy(), keys_np[order])
    _assert_eq("t10 plain sort i64 keys n=10000 (values)", values.to_numpy(), vals_np[order])


t10()


# ---- t11: segmented sort with i64 keys ----
def t11():
    D = 3
    n, nseg = 1000, 8
    rng = np.random.default_rng(7)
    keys_np = rng.integers(-2**40, 2**40, n).astype(np.int64)
    vals_np = np.arange(n, dtype=np.int32)
    ids = np.sort(rng.choice(np.arange(1, n), nseg - 1, replace=False)).astype(np.int32)
    seg_np = np.concatenate([[0], ids, [n]]).astype(np.int32)

    keys = qd.ndarray(qd.i64, n)
    tmp_keys = qd.ndarray(qd.i64, n)
    values = qd.ndarray(qd.i32, n)
    tmp_values = qd.ndarray(qd.i32, n)
    seg = qd.ndarray(qd.i32, nseg + 1)
    tmp_seg = qd.ndarray(qd.i32, n)  # workspace of length n, not nseg+1
    scratch = qd.ndarray(qd.u32, sort_segmented_scratch_slots(n, D))
    keys.from_numpy(keys_np)
    values.from_numpy(vals_np)
    seg.from_numpy(seg_np)
    tmp_keys.from_numpy(np.zeros(n, np.int64))
    tmp_values.from_numpy(np.zeros(n, np.int32))
    tmp_seg.from_numpy(np.zeros(n, np.int32))
    scratch.from_numpy(np.zeros(scratch.shape[0], np.uint32))
    n0 = qd.ndarray(qd.i32, ())
    n0.fill(n)
    ns0 = qd.ndarray(qd.i32, ())
    ns0.fill(nseg)

    @qd.kernel
    def k(keys: i64_1d, tmp_keys: i64_1d, values: i32_1d, tmp_values: i32_1d,
          seg: i32_1d, tmp_seg: i32_1d, scratch: u32_1d, n: i32_0d, nseg: i32_0d):
        sort_segmented(keys, tmp_keys, values, tmp_values, seg, tmp_seg, scratch, n, nseg,
                       qd.i64, True, 64, D)

    try:
        k(keys, tmp_keys, values, tmp_values, seg, tmp_seg, scratch, n0, ns0)
        qd.sync()
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " ")
        if METAL and SEGMENTED_I64_XFAIL and "bit_cast is only supported" in msg:
            xfails.append("t11 sort_segmented i64 (metal, loud rejection as documented)")
            print(f"[XFAIL] t11 sort_segmented i64 keys n=1000 x8 seg (metal)  known gap: "
                  f"_segmented.py:162 bit_cast(keys_in[i], u32) width-hard-wired")
            return
        gate("t11 sort_segmented i64 keys", False, msg[-200:])
        return
    # compiled and ran - check correctness
    got_k, got_v = keys.to_numpy(), values.to_numpy()
    exp_k, exp_v = keys_np.copy(), vals_np.copy()
    for s in range(nseg):
        lo, hi = int(seg_np[s]), int(seg_np[s + 1])
        if hi - lo > 1:
            order = np.argsort(keys_np[lo:hi], kind="stable")
            exp_k[lo:hi] = keys_np[lo:hi][order]
            exp_v[lo:hi] = vals_np[lo:hi][order]
    if METAL and SEGMENTED_I64_XFAIL:
        fails.append("t11")
        gate("t11 sort_segmented i64 keys", False,
             "UNEXPECTED PASS on metal under SEGMENTED_I64_XFAIL=True - flip the flag")
        return
    _assert_eq("t11 sort_segmented i64 keys", got_k, exp_k)
    _assert_eq("t11 sort_segmented i64 keys (values)", got_v, exp_v)


t11()


# ---- t12 control: segmented sort with i32 keys on the same harness ----
def t12():
    D = 3
    n, nseg = 1000, 8
    rng = np.random.default_rng(13)
    keys_np = rng.integers(-2**31, 2**31, n).astype(np.int32)
    vals_np = np.arange(n, dtype=np.int32)
    ids = np.sort(rng.choice(np.arange(1, n), nseg - 1, replace=False)).astype(np.int32)
    seg_np = np.concatenate([[0], ids, [n]]).astype(np.int32)

    keys = qd.ndarray(qd.i32, n)
    tmp_keys = qd.ndarray(qd.i32, n)
    values = qd.ndarray(qd.i32, n)
    tmp_values = qd.ndarray(qd.i32, n)
    seg = qd.ndarray(qd.i32, nseg + 1)
    tmp_seg = qd.ndarray(qd.i32, n)  # workspace of length n, not nseg+1
    scratch = qd.ndarray(qd.u32, sort_segmented_scratch_slots(n, D))
    keys.from_numpy(keys_np)
    values.from_numpy(vals_np)
    seg.from_numpy(seg_np)
    tmp_keys.from_numpy(np.zeros(n, np.int32))
    tmp_values.from_numpy(np.zeros(n, np.int32))
    tmp_seg.from_numpy(np.zeros(n, np.int32))
    scratch.from_numpy(np.zeros(scratch.shape[0], np.uint32))
    n0 = qd.ndarray(qd.i32, ())
    n0.fill(n)
    ns0 = qd.ndarray(qd.i32, ())
    ns0.fill(nseg)

    @qd.kernel
    def k(keys: i32_1d, tmp_keys: i32_1d, values: i32_1d, tmp_values: i32_1d,
          seg: i32_1d, tmp_seg: i32_1d, scratch: u32_1d, n: i32_0d, nseg: i32_0d):
        sort_segmented(keys, tmp_keys, values, tmp_values, seg, tmp_seg, scratch, n, nseg,
                       qd.i32, True, 32, D)

    ok, err = run_kernel("t12 launch", k, keys, tmp_keys, values, tmp_values, seg, tmp_seg,
                         scratch, n0, ns0)
    if not ok:
        gate("t12 sort_segmented i32 control", False, err)
        return
    got_k, got_v = keys.to_numpy(), values.to_numpy()
    exp_k, exp_v = keys_np.copy(), vals_np.copy()
    for s in range(nseg):
        lo, hi = int(seg_np[s]), int(seg_np[s + 1])
        if hi - lo > 1:
            order = np.argsort(keys_np[lo:hi], kind="stable")
            exp_k[lo:hi] = keys_np[lo:hi][order]
            exp_v[lo:hi] = vals_np[lo:hi][order]
    _assert_eq("t12 sort_segmented i32 control", got_k, exp_k)
    _assert_eq("t12 sort_segmented i32 control (values)", got_v, exp_v)


t12()


# ---- perf (informational, WARN-only) ----
def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _paired(name, mk, w64, w32, pairs, el_gb=2 * 8_388_608):
    """mk(dt, npdt) -> launch closure; interleave i64/i32 reps, report medians."""
    f64, f32 = mk(qd.i64, np.int64), mk(qd.i32, np.int32)
    t64s, t32s = [], []
    for _ in range(pairs):
        t0 = time.perf_counter()
        f64()
        qd.sync()
        t64s.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        f32()
        qd.sync()
        t32s.append(time.perf_counter() - t0)
    m64, m32 = _median(t64s), _median(t32s)
    sl = _median([a / b for a, b in zip(t64s, t32s)])
    if w64:
        g64, g32 = el_gb * w64 / m64 / 1e9, el_gb * w32 / m32 / 1e9
        print(f"{name} i64: {g64:7.1f} GB/s ({m64*1e3:7.2f} ms)   "
              f"i32: {g32:7.1f} GB/s ({m32*1e3:7.2f} ms)   slowdown x{sl:.2f} (median of {pairs} pairs)")
    else:
        print(f"{name} i64: {m64*1e3:7.2f} ms   i32: {m32*1e3:7.2f} ms   "
              f"slowdown x{sl:.2f} (median of {pairs} pairs)")
    return sl


def perf():
    NP = 8_388_608  # 8M
    el = 2 * NP  # one read + one write per element
    print(f"\n--- perf (interleaved i64/i32 pairs, median; GB/s informational, shape-bound) ---")

    def mk_copy(dt, npdt):
        src = qd.ndarray(dt, NP)
        dst = qd.ndarray(dt, NP)
        src.from_numpy(np.arange(NP).astype(npdt))

        @qd.kernel
        def k(src: qd.types.ndarray(dt, ndim=1), dst: qd.types.ndarray(dt, ndim=1)):
            for i in range(NP):
                dst[i] = src[i]

        qd.sync()
        return lambda: k(src, dst)

    sl1 = _paired("p1 copy ", mk_copy, 8, 4, 8, el)

    def mk_pack(dt, npdt):
        g = qd.ndarray(dt, NP)
        s = qd.ndarray(dt, NP)
        out = qd.ndarray(dt, NP)
        g.from_numpy((np.arange(NP) % 65536).astype(npdt))
        s.from_numpy(np.arange(NP).astype(npdt))

        @qd.kernel
        def k(g: qd.types.ndarray(dt, ndim=1), s: qd.types.ndarray(dt, ndim=1),
              out: qd.types.ndarray(dt, ndim=1)):
            for i in range(NP):
                if qd.static(dt == qd.i64):
                    out[i] = (g[i] << 32) | s[i]
                else:
                    out[i] = (g[i] << 16) | s[i]

        qd.sync()
        return lambda: k(g, s, out)

    sl2 = _paired("p2 pack ", mk_pack, 8, 4, 8, el)

    def mk_sort(dt, npdt):
        D = 3
        keys = qd.ndarray(dt, NP)
        tmp_keys = qd.ndarray(dt, NP)
        values = qd.ndarray(qd.i32, NP)
        tmp_values = qd.ndarray(qd.i32, NP)
        scratch = qd.ndarray(qd.u32, sort_scratch_slots(NP, D))
        keys.from_numpy((np.arange(NP, dtype=np.int64) * 2654435761 % (2**62)).astype(npdt))
        values.from_numpy(np.arange(NP, dtype=np.int32))
        tmp_keys.from_numpy(np.zeros(NP, npdt))
        tmp_values.from_numpy(np.zeros(NP, np.int32))
        scratch.from_numpy(np.zeros(scratch.shape[0], np.uint32))
        n0 = qd.ndarray(qd.i32, ())
        n0.fill(NP)
        end_bit = 64 if dt == qd.i64 else 32

        @qd.kernel
        def k(keys: qd.types.ndarray(dt, ndim=1), tmp_keys: qd.types.ndarray(dt, ndim=1),
              values: i32_1d, tmp_values: i32_1d, scratch: u32_1d, n: i32_0d):
            sort(keys, tmp_keys, values, tmp_values, scratch, n, dt, True, end_bit, D)

        qd.sync()
        return lambda: k(keys, tmp_keys, values, tmp_values, scratch, n0)

    sl3 = _paired("p3 sort ", mk_sort, 0, 0, 5, el)

    for nm, sl in (("p1 copy", sl1), ("p2 pack", sl2), ("p3 sort", sl3)):
        if sl > PERF_ALERT_SLOWDOWN:
            print(f"[ALERT] {nm} i64 slowdown x{sl:.2f} > {PERF_ALERT_SLOWDOWN}x - hot-path review")


if METAL:
    perf()

for _x in xfails:
    print(f"[XFAIL] {_x}")
xf_note = f" (xfail accepted: {len(xfails)})" if xfails else ""
verdict = "ALL PASS" if not fails else f"FAIL: {', '.join(fails)}"
print(f"\nI64 PROBE [{ARCH}]: {verdict}{xf_note}")
raise SystemExit(0 if not fails else 1)
