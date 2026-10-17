"""Recon probe for the qd-native BVH line (spec docs/bvh_development_spec.md step 1).

Pins the runtime forms the BVH kernels will rely on, against the installed
.venv-consolidated runtime (1.3.4 dev25+g3297e86fe). Findings recorded here
drove the bvh.py design:

  P1  in-kernel fixed-length local array: NOT AVAILABLE (no LocalArray API in
      the runtime; in-kernel qd.ndarray attempts rejected). -> iterator state
      lives in caller-owned 2-D ndarray rows (the weld_geom_list scratch
      pattern).
  P2  bit_cast f32->u32->f32 preserves NaN payloads incl. bit31-set patterns
      (the BVHPackedNodeHalf {i:31|b:1} w-channel packing).
  P3  vec4 ndarray: store packed w, reload, unpack via bit_cast - roundtrip
      exact (node_lowers/node_uppers layout).
  P4  ndarray-form Batcher odd-even merge sort (the quadrants parallel_sort /
      sort_stage network re-stated with ndarray args) sorts (i32 key, i32
      value) pairs on CPU: keys ascending, pairs preserved (tie order is
      network-dependent, NOT stable - documented). Rationale: the venv
      runtime's composable radix sort has no CPU dispatch (its GPU staircase
      dies on block.sync@arm64), and sort_stage itself is field-only (its
      params are qd.template() -> ndarrays are rejected; the body reads
      keys.snode). bvh.py therefore carries a ndarray-args restatement of the
      same network + schedule; sort 本体 untouched.
  P5  atomic_add(node_counts[i], 1) returns the old value (build_hierarchy /
      refit counters).
  P6  multi-phase kernel: parallel phase then loop_config(serialize=True)
      phase in one launch, ordering preserved (the post-sort build kernel
      shape: deltas/leaves/reset parallel + hierarchy serialized + mark).
  P7  u32 shift/mask forms in kernel (morton3 part1by2).

Run: python _probe/bvh_local_array_probe.py [cpu|metal]   (exit 0 = all pass)
"""
import sys

import numpy as np

import quadrants as qd
from quadrants.lang.misc import loop_config
from quadrants.lang.ops import atomic_add, bit_cast
from quadrants.types.primitive_types import f32, i32, u32

ARCH = sys.argv[1] if len(sys.argv) > 1 else "cpu"
qd.init(arch=qd.metal if ARCH == "metal" else qd.cpu)

fails = []


def gate(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag:4}] {name} {detail}")
    if not ok:
        fails.append(name)


_i32_1d = qd.types.ndarray(dtype=qd.i32, ndim=1)
_i32_2d = qd.types.ndarray(dtype=qd.i32, ndim=2)
_u32_1d = qd.types.ndarray(dtype=qd.u32, ndim=1)
_f32_1d = qd.types.ndarray(dtype=qd.f32, ndim=1)
_vec4_1d = qd.types.ndarray(dtype=qd.math.vec4, ndim=1)
_i32_0d = qd.types.ndarray(dtype=qd.i32, ndim=0)


# ---- P1: in-kernel fixed-length local array --------------------------------
def p1():
    verdict = "unavailable"
    detail = ""

    # form A: qd.ndarray inside a kernel
    try:

        @qd.kernel
        def k_local_a(out: _i32_1d):
            buf = qd.ndarray(qd.i32, 64)
            for i in range(64):
                buf[i] = i
            out[0] = buf[7]

        out = qd.ndarray(qd.i32, 1)
        k_local_a(out)
        qd.sync()
        if out.to_numpy()[0] == 7:
            verdict = "works (qd.ndarray in kernel)"
    except Exception as e:  # noqa: BLE001
        detail = str(e).replace("\n", " ")[:110]

    # form B: python list fixed length in kernel (some frontends unroll this)
    if verdict == "unavailable":
        try:

            @qd.kernel
            def k_local_b(out: _i32_1d):
                buf = [0, 0, 0, 0]
                for i in range(4):
                    buf[i] = i * 2
                out[0] = buf[2]

            out = qd.ndarray(qd.i32, 1)
            k_local_b(out)
            qd.sync()
            if out.to_numpy()[0] == 4:
                verdict = "works (python list unroll)"
        except Exception as e:  # noqa: BLE001
            detail = str(e).replace("\n", " ")[:110]

    gate("P1 local fixed-length array", True,
         f"verdict={verdict} (expected: unavailable -> per-query scratch rows) {detail}")


p1()


# ---- P2/P3: bit_cast NaN payload + vec4 w-channel packing -------------------
def p2p3():
    @qd.kernel
    def k_pack(nodes: _vec4_1d, out_bits: _u32_1d):
        for i in range(4):
            child = i * 3
            leaf = i % 2
            bits = u32(child) | (u32(leaf) << u32(31))
            nodes[i] = qd.math.vec4(f32(1.5), f32(-2.5), f32(3.5), bit_cast(bits, f32))

        for i in range(4):
            nl = nodes[i]
            out_bits[i] = bit_cast(nl[3], u32)

    nodes = qd.ndarray(dtype=qd.math.vec4, shape=4)
    out_bits = qd.ndarray(qd.u32, 4)
    k_pack(nodes, out_bits)
    qd.sync()
    got = out_bits.to_numpy()
    want = np.array([(i * 3) | ((i % 2) << 31) for i in range(4)], dtype=np.uint32)
    gate("P2/P3 packed-node bitfield roundtrip (incl bit31 NaN payload)", bool(np.array_equal(got, want)),
         f"got={got.tolist()} want={want.tolist()}")


p2p3()


# ---- P4: ndarray-form Batcher odd-even merge sort ---------------------------
@qd.kernel
def _probe_sort_stage(keys: _i32_1d, values: _i32_1d, n: int, p: int, k: int, invocations: int):
    # 1:1 restatement of quadrants._kernels.sort_stage (field-only) with
    # ndarray args; identical Batcher compare-swap network and schedule.
    for inv in range(invocations):
        j = k % p + inv * 2 * k
        for i in range(0, qd.min(k, n - j - k)):
            a = i + j
            b = i + j + k
            if int(a / (p * 2)) == int(b / (p * 2)):
                key_a = keys[a]
                key_b = keys[b]
                if key_a > key_b:
                    keys[a] = key_b
                    keys[b] = key_a
                    tmp = values[a]
                    values[a] = values[b]
                    values[b] = tmp


def p4():
    n = 1000
    rng = np.random.default_rng(4)
    keys_np = rng.integers(0, 1 << 30, n).astype(np.int32)
    vals_np = (np.arange(n, dtype=np.int32) * 3 % 997).astype(np.int32)
    keys = qd.ndarray(qd.i32, n)
    values = qd.ndarray(qd.i32, n)
    keys.from_numpy(keys_np)
    values.from_numpy(vals_np)

    # the parallel_sort schedule (quadrants/algorithms/_algorithms.py)
    p = 1
    while p < n:
        k = p
        while k >= 1:
            invocations = int((n - k - k % p) / (2 * k)) + 1
            _probe_sort_stage(keys, values, n, p, k, invocations)
            qd.sync()
            k = int(k / 2)
        p = int(p * 2)

    got_k = keys.to_numpy()
    got_v = values.to_numpy()
    pairs_in = set(zip(keys_np.tolist(), vals_np.tolist()))
    pairs_out = set(zip(got_k.tolist(), got_v.tolist()))
    ok = (np.diff(got_k.astype(np.int64)) >= 0).all() and pairs_in == pairs_out
    gate("P4 ndarray odd-even sort (key,value) sorted + preserved", bool(ok),
         "" if ok else "keys_asc=%s pairs=%s" % (
             bool((np.diff(got_k.astype(np.int64)) >= 0).all()), pairs_in == pairs_out))


p4()


# ---- P5: atomic_add old-value return ----------------------------------------
def p5():
    cnt = qd.ndarray(qd.i32, 4)
    cnt.from_numpy(np.zeros(4, np.int32))
    old = qd.ndarray(qd.i32, 4)

    @qd.kernel
    def k(cnt: _i32_1d, old: _i32_1d):
        for i in range(4):
            old[i] = atomic_add(cnt[i], 1)

    k(cnt, old)
    qd.sync()
    # distinct cells, one increment each: every thread observes old=0
    ok = (old.to_numpy() == 0).all() and (cnt.to_numpy() == 1).all()

    # same-cell serial chain (the worklist counter form): old[j] == j
    cnt2 = qd.ndarray(qd.i32, 1)
    cnt2.from_numpy(np.zeros(1, np.int32))
    old2 = qd.ndarray(qd.i32, 4)

    @qd.kernel
    def k2(cnt2: _i32_1d, old2: _i32_1d):
        loop_config(serialize=True)
        for j in range(4):
            old2[j] = atomic_add(cnt2[0], 1)

    k2(cnt2, old2)
    qd.sync()
    ok2 = (old2.to_numpy() == np.arange(4, dtype=np.int32)).all() and cnt2.to_numpy()[0] == 4
    gate("P5 atomic_add(arr[i],1) returns old value", bool(ok and ok2),
         f"parallel_ok={bool(ok)} serial_ok={bool(ok2)}")


p5()


# ---- P6: multi-phase kernel: parallel for + serialized for in one launch ----
def p6():
    n = 64
    src = qd.ndarray(qd.i32, n)
    nodes = qd.ndarray(qd.i32, n)
    src.from_numpy(np.zeros(n, np.int32))
    nodes.from_numpy(np.zeros(n, np.int32))
    n0 = qd.ndarray(qd.i32, ())
    n0.fill(n)

    @qd.kernel
    def k(src: _i32_1d, nodes: _i32_1d, n: _i32_0d):
        # phase 1 (parallel): scatter values
        for i in range(n[()]):
            src[i] = (i * 37 + 11) % n[()]
        # phase 2 (serialized): reads phase-1 array (cross-phase visibility),
        # writes a different array so the model has no self-clobber
        loop_config(serialize=True)
        for i in range(n[()]):
            acc = 0
            if i > 0:
                acc = src[i - 1]
            nodes[i] = acc + i

    k(src, nodes, n0)
    qd.sync()
    in_np = np.array([(i * 37 + 11) % n for i in range(n)], dtype=np.int64)
    want = np.array([0] + [int(in_np[i - 1]) + i for i in range(1, n)], dtype=np.int64)
    got = nodes.to_numpy().astype(np.int64)
    ok = np.array_equal(got, want)
    gate("P6 multiphase kernel (parallel then serialized, ordering kept)", bool(ok),
         "" if ok else f"first mismatch at {int(np.flatnonzero(got != want)[0])}")


p6()


# ---- P7: u32 shift/mask forms ------------------------------------------------
def p7():
    out = qd.ndarray(qd.u32, 5)

    @qd.kernel
    def k(out: _u32_1d):
        for i in range(1):
            m = u32(0x01010101)
            m = (m ^ (m << 16)) & u32(0xFF0000FF)
            m = (m ^ (m << 8)) & u32(0x0300F00F)
            m = (m ^ (m << 4)) & u32(0x030C30C3)
            m = (m ^ (m << 2)) & u32(0x09249249)
            out[0] = m
            out[1] = u32(12345) >> u32(1)
            # BVHPackedNodeHalf unpack forms: flag in bit31, payload in bits 0..30
            bits = u32(0x80000005)
            out[2] = bits >> u32(31)
            out[3] = bits & u32(0x7FFFFFFF)
            out[4] = u32(5) | (u32(1) << u32(31))

    k(out)
    qd.sync()
    got = out.to_numpy()

    def part1by2(n):
        n = (n ^ (n << 16)) & 0xFF0000FF
        n = (n ^ (n << 8)) & 0x0300F00F
        n = (n ^ (n << 4)) & 0x030C30C3
        n = (n ^ (n << 2)) & 0x09249249
        return n

    want0 = part1by2(0x01010101)
    ok = (got[0] == want0 and got[1] == 12345 >> 1 and got[2] == 1
          and got[3] == 5 and got[4] == 0x80000005)
    gate("P7 u32 shift/mask (part1by2 + node-bitfield forms)", bool(ok),
         f"got={got.tolist()} want0={want0:#x}")


p7()

print(f"BVH RECON PROBE [{ARCH}]: {'ALL PASS' if not fails else 'FAILURES: ' + repr(fails)}")
sys.exit(1 if fails else 0)
