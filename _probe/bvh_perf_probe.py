"""Three-way BVH perf probe: warp wp.Bvh (cpu) vs qd Bvh (cpu) vs qd Bvh (metal).

User acceptance add-on for mujocoqd/bvh: consistency is gate 18's job, this is
speed only. Identical inputs across arms (seeded numpy): 1k/10k/50k items,
uniform + clustered distributions.

Reporting framing (coordinator ruling): build = ONE-TIME cost (launch + tree
construction, host-orchestrated); queries = STEADY-STATE compute - rows carry
per-query microseconds at two batch sizes (512 / 5000) so throughput scaling is
visible. The qd metal per-launch command-buffer floor is measured directly (a
trivial kernel, median of 20) and metal query rows print per-query numbers both
with and minus that floor: the floor is the qd metal backend's universal launch
overhead (any kernel pays it), not BVH compute.

Modes (one process per mode; qd is one-arch-per-process):
  cpu    warp-cpu AND qd-cpu in one process, rep order interleaved (A/B
         alternating); median of 5.
  metal  qd metal only; runs tests/robostudio/_gpu_busy.sh first and
         BUSY-SKIPs honestly when the GPU is contended (exit 0).
  second arg: "full" (default) also benches builds; "queries" builds each
  scenario once (untimed) and benches queries only.

Timing protocol: qd build = Bvh(lower, upper) + build() - the full host path
with every launch and sync. warp build = wp.Bvh(lo, up) constructor, same
full-path semantics. Queries = one batch launch + sync, arrays pre-staged
outside the timed region; JIT compiles happen in warmup. Cross-arm insurance:
warp and qd hit totals must agree EXACTLY (same f32 predicate; a mismatch also
detects hit-cap truncation, since warp counts uncapped while the qd batch
stops at the cap). The f64 numpy reference check runs only at the 512 batch
(tolerance max(10, 0.1%) for knife-edge slab ties; one such tie observed at
50k ray: 51964 vs 51965 out of 25M tests).

Run: python _probe/bvh_perf_probe.py [cpu|metal] [full|queries]
The cpu mode needs a warp-bearing venv on PYTHONPATH (QD_GATE_WARP_SP convention).
"""
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ARCH = sys.argv[1] if len(sys.argv) > 1 else "cpu"
MODE = sys.argv[2] if len(sys.argv) > 2 else "full"
ROOT = Path(__file__).resolve().parents[1]

import quadrants as qd

qd.init(arch=qd.metal if ARCH == "metal" else qd.cpu)

_spec = importlib.util.spec_from_file_location("quadrants.bvh", ROOT / "python" / "quadrants" / "bvh.py")
bvh = importlib.util.module_from_spec(_spec)
sys.modules["quadrants.bvh"] = bvh
_spec.loader.exec_module(bvh)

HAS_WARP = False
if ARCH == "cpu":
    try:
        import warp as wp
        HAS_WARP = True
    except ImportError:
        print("PERF PROBE: warp not importable - qd-cpu rows only, no wp-cpu pairing")

TIERS = [1000, 10000, 50000]
DISTS = ["uniform", "cluster"]
QNS = [512, 5000]
REPS = 5
# per-query hit-cap by item tier for the 5k batch (no numpy precount there;
# generous enough that truncation would show up as a warp-vs-qd total mismatch,
# because warp counts uncapped while the qd batch stops at the cap)
CAP_5K = {1000: 1024, 10000: 4096, 50000: 16384}


def gen_items(n, dist):
    rng = np.random.default_rng(1000 + n)
    if dist == "uniform":
        lo = rng.uniform(0.0, 8.0, (n, 3))
        sz = rng.uniform(0.05, 0.6, (n, 3))
    else:
        centers = rng.uniform(1.0, 7.0, (16, 3))
        cid = rng.integers(0, 16, n)
        lo = centers[cid] + rng.uniform(-0.3, 0.3, (n, 3))
        sz = rng.uniform(0.02, 0.25, (n, 3))
    lo = lo.astype(np.float32)
    return lo, (lo + sz).astype(np.float32)


def gen_queries(n, dist, nq):
    rng = np.random.default_rng(2000 + n)
    qlo = rng.uniform(0.0, 7.2, (nq, 3)).astype(np.float32)
    qhi = (qlo + rng.uniform(0.3, 1.0, (nq, 3))).astype(np.float32)
    st = rng.uniform(-2.0, 2.0, (nq, 3)).astype(np.float32)
    dr = rng.uniform(0.3, 1.2, (nq, 3)).astype(np.float32)
    md = rng.uniform(4.0, 9.0, nq).astype(np.float32)
    ce = rng.uniform(0.0, 8.0, (nq, 3)).astype(np.float32)
    ra = rng.uniform(0.2, 1.2, nq).astype(np.float32)
    return qlo, qhi, st, dr, md, ce, ra


# ---------------------------------------------------------- numpy precounts ---
def np_counts_aabb(lo, up, qlo, qhi):
    nq = qlo.shape[0]
    out = np.zeros(nq, np.int64)
    step = min(32, nq)
    for q0 in range(0, nq, step):
        q1 = min(q0 + step, nq)
        hit = ((lo[None, :, :] <= qhi[q0:q1, None, :]).all(axis=2)
               & (up[None, :, :] >= qlo[q0:q1, None, :]).all(axis=2))
        out[q0:q1] = hit.sum(axis=1)
    return out


def np_counts_ray(lo, up, st, dr, md):
    nq = st.shape[0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rcp = 1.0 / dr.astype(np.float64)
        out = np.zeros(nq, np.int64)
        step = min(32, nq)
        for q0 in range(0, nq, step):
            q1 = min(q0 + step, nq)
            pos = st[q0:q1, None, :].astype(np.float64)
            r = rcp[q0:q1, None, :]
            l1 = (lo[None, :, :].astype(np.float64) - pos) * r
            l2 = (up[None, :, :].astype(np.float64) - pos) * r
            lmin = np.minimum(l1, l2).max(axis=2)
            lmax = np.maximum(l1, l2).min(axis=2)
            hit = (lmax >= 0.0) & (lmax >= lmin) & (lmin < md[q0:q1, None])
            out[q0:q1] = hit.sum(axis=1)
    return out


def np_counts_sphere(lo, up, ce, ra):
    nq = ce.shape[0]
    out = np.zeros(nq, np.int64)
    step = min(32, nq)
    for q0 in range(0, nq, step):
        q1 = min(q0 + step, nq)
        c = ce[q0:q1, None, :]
        d = np.maximum(np.maximum(lo[None, :, :] - c, c - up[None, :, :]), 0.0).astype(np.float64)
        d2 = (d * d).sum(axis=2)
        out[q0:q1] = (d2 <= (ra[q0:q1, None] ** 2)).sum(axis=1)
    return out


def make_v3(np3):
    a = qd.ndarray(dtype=qd.math.vec3, shape=int(np3.shape[0]))
    a.from_numpy(np.ascontiguousarray(np3, dtype=np.float32))
    return a


def qd_slots(nq):
    state = qd.ndarray(qd.i32, (nq, 4))
    state.from_numpy(np.zeros((nq, 4), np.int32))
    stack = qd.ndarray(qd.i32, (nq, bvh._STACK_DEPTH))
    stack.from_numpy(np.zeros((nq, bvh._STACK_DEPTH), np.int32))
    return state, stack


def median_s(ts):
    return sorted(ts)[len(ts) // 2]


def row(kind, arm, secs, nq=None, floor_ms=None):
    ms = median_s(secs) * 1000.0
    if nq is None:
        print(f"  PERF {kind:7s} {arm:9s} {ms:10.2f} ms            (median of {len(secs)})")
        return
    ns_q = ms * 1e6 / nq
    if floor_ms is not None:
        ns_q_net = (ms - floor_ms) * 1e6 / nq
        print(f"  PERF {kind:7s} {arm:9s} {ms:10.2f} ms  {ns_q:8.0f} ns/q  "
              f"(floor share {floor_ms * 1e6 / nq:6.0f} ns/q -> net {ns_q_net:8.0f} ns/q)  "
              f"(median of {len(secs)})")
    else:
        print(f"  PERF {kind:7s} {arm:9s} {ms:10.2f} ms  {ns_q:8.0f} ns/q  (median of {len(secs)})")


def bench_qd_build(lo_a, up_a, reps):
    b = bvh.Bvh(lo_a, up_a)  # warmup: compile the build kernels once
    b.build()
    qd.sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        t = bvh.Bvh(lo_a, up_a)
        t.build()
        qd.sync()
        ts.append(time.perf_counter() - t0)
    return ts


def bench_qd_query(tree, kind, qargs, cap, reps, nq):
    state, stack = qd_slots(nq)
    counts = qd.ndarray(qd.i32, nq)
    idx = qd.ndarray(qd.i32, (nq, cap))
    if kind == "aabb":
        def run():
            bvh.bvh_query_aabb_batch(tree.root, tree.node_lowers, tree.node_uppers,
                                     tree.primitive_indices, tree.lower, tree.upper,
                                     qargs[0], qargs[1], stack, state, cap, nq, counts, idx)
    elif kind == "ray":
        def run():
            bvh.bvh_query_ray_batch(tree.root, tree.node_lowers, tree.node_uppers,
                                    tree.primitive_indices, tree.lower, tree.upper,
                                    qargs[0], qargs[1], qargs[2], stack, state, cap, nq, counts, idx)
    else:
        def run():
            bvh.bvh_query_sphere_batch(tree.root, tree.node_lowers, tree.node_uppers,
                                       tree.primitive_indices, tree.lower, tree.upper,
                                       qargs[0], qargs[1], stack, state, cap, nq, counts, idx)
    run()
    qd.sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run()
        qd.sync()
        ts.append(time.perf_counter() - t0)
    return ts, int(counts.to_numpy().sum())


@qd.kernel
def k_launch_floor(x: qd.types.ndarray(dtype=qd.i32, ndim=1)):
    for i in range(1):
        x[0] = 1


def measure_launch_floor():
    """Per-launch cost of the qd backend itself (any kernel pays it)."""
    x = qd.ndarray(qd.i32, 1)
    x.from_numpy(np.zeros(1, np.int32))
    k_launch_floor(x)
    qd.sync()
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        k_launch_floor(x)
        qd.sync()
        ts.append(time.perf_counter() - t0)
    return median_s(ts) * 1000.0


FLOOR_MS = None
if ARCH == "metal":
    busy = subprocess.run(["zsh", str(ROOT / "tests" / "robostudio" / "_gpu_busy.sh")],
                          capture_output=True).returncode != 0
    if busy:
        print("PERF PROBE [metal]: BUSY-SKIP (gpu contended; no metal perf numbers)")
        sys.exit(0)
    FLOOR_MS = measure_launch_floor()
    print(f"  PERF launch-floor qd-metal  {FLOOR_MS * 1000.0:10.1f} us/launch "
          f"(backend-wide, any kernel; quoted separately, not BVH compute)")

# ---------------------------------------------------------------- scenarios ---
for n in TIERS:
    for dist in DISTS:
        lo_np, up_np = gen_items(n, dist)
        lo_a = make_v3(lo_np)
        up_a = make_v3(up_np)

        tree = bvh.Bvh(lo_a, up_a)
        tree.build()
        qd.sync()

        if MODE == "full":
            row("build", f"qd-{ARCH}", bench_qd_build(lo_a, up_a, REPS))

        for nq in QNS:
            qlo, qhi, st, dr, md, ce, ra = gen_queries(n, dist, nq)
            cap = CAP_5K[n] if nq == 5000 else 0
            if nq == 512:
                want = {
                    "aabb": int(np_counts_aabb(lo_np, up_np, qlo, qhi).sum()),
                    "ray": int(np_counts_ray(lo_np, up_np, st, dr, md).sum()),
                    "sphere": int(np_counts_sphere(lo_np, up_np, ce, ra).sum()),
                }
                cap = int(max(np_counts_aabb(lo_np, up_np, qlo, qhi).max(),
                              np_counts_ray(lo_np, up_np, st, dr, md).max(),
                              np_counts_sphere(lo_np, up_np, ce, ra).max())) + 1
            md_a = qd.ndarray(qd.f32, nq)
            md_a.from_numpy(md)
            ra_a = qd.ndarray(qd.f32, nq)
            ra_a.from_numpy(ra)
            qargs = {"aabb": (make_v3(qlo), make_v3(qhi)),
                     "ray": (make_v3(st), make_v3(dr), md_a),
                     "sphere": (make_v3(ce), ra_a)}

            if not HAS_WARP:
                for kind in ("aabb", "ray", "sphere"):
                    ts, tot = bench_qd_query(tree, kind, qargs[kind], cap, REPS, nq)
                    if nq == 512:
                        tol = max(10, int(0.001 * want[kind]))
                        assert abs(tot - want[kind]) <= tol, \
                            f"qd {kind} hits {tot} vs numpy {want[kind]} ({n}/{dist})"
                    row(kind, f"qd-{ARCH}", ts, nq, FLOOR_MS)
                continue

            # ---------------- warp arm, interleaved with the qd arm -------------
            wpk = wp.kernel

            @wpk
            def wp_q_aabb(bvh_id: wp.uint64, qlo_a: wp.array[wp.vec3], qhi_a: wp.array[wp.vec3],
                          out: wp.array[wp.int32], counts: wp.array[wp.int32], cap: int):
                q = wp.tid()
                query = wp.bvh_query_aabb(bvh_id, qlo_a[q], qhi_a[q])
                item = int(-1)
                kk = int(0)
                total = int(0)
                while wp.bvh_query_next(query, item):
                    if kk < cap:
                        out[q * cap + kk] = item
                    kk = kk + int(1)
                    total = total + int(1)
                counts[q] = total

            @wpk
            def wp_q_ray(bvh_id: wp.uint64, st_a: wp.array[wp.vec3], dr_a: wp.array[wp.vec3],
                         md_w: wp.array[wp.float32], out: wp.array[wp.int32],
                         counts: wp.array[wp.int32], cap: int):
                q = wp.tid()
                query = wp.bvh_query_ray(bvh_id, st_a[q], dr_a[q])
                item = int(-1)
                kk = int(0)
                total = int(0)
                while wp.bvh_query_next(query, item, md_w[q]):
                    if kk < cap:
                        out[q * cap + kk] = item
                    kk = kk + int(1)
                    total = total + int(1)
                counts[q] = total

            @wpk
            def wp_q_sphere(bvh_id: wp.uint64, ce_a: wp.array[wp.vec3], ra_w: wp.array[wp.float32],
                            out: wp.array[wp.int32], counts: wp.array[wp.int32], cap: int):
                q = wp.tid()
                query = wp.bvh_query_sphere(bvh_id, ce_a[q], ra_w[q])
                item = int(-1)
                kk = int(0)
                total = int(0)
                while wp.bvh_query_next(query, item):
                    if kk < cap:
                        out[q * cap + kk] = item
                    kk = kk + int(1)
                    total = total + int(1)
                counts[q] = total

            wp.init()
            wcnt = wp.array(np.zeros(nq, dtype=np.int32))
            wout = wp.array(np.zeros(nq * cap, dtype=np.int32))
            qlo_w = wp.array(qlo, dtype=wp.vec3)
            qhi_w = wp.array(qhi, dtype=wp.vec3)
            st_w = wp.array(st, dtype=wp.vec3)
            dr_w = wp.array(dr, dtype=wp.vec3)
            md_w = wp.array(md, dtype=wp.float32)
            ce_w = wp.array(ce, dtype=wp.vec3)
            ra_w = wp.array(ra, dtype=wp.float32)
            wb = wp.Bvh(wp.array(lo_np, dtype=wp.vec3), wp.array(up_np, dtype=wp.vec3))
            wargs = {
                "aabb": [wb.id, qlo_w, qhi_w, wout, wcnt, cap],
                "ray": [wb.id, st_w, dr_w, md_w, wout, wcnt, cap],
                "sphere": [wb.id, ce_w, ra_w, wout, wcnt, cap],
            }
            wkern = {"aabb": wp_q_aabb, "ray": wp_q_ray, "sphere": wp_q_sphere}
            for kind in ("aabb", "ray", "sphere"):
                wp.launch(wkern[kind], dim=nq, inputs=wargs[kind])  # warmup
            wp.synchronize()

            if MODE == "full":
                wb_ts = []
                qd_bts = []
                for r in range(REPS):
                    if r % 2 == 0:
                        t0 = time.perf_counter()
                        # throwaway handle: MUST NOT clobber wb - wargs holds
                        # wb.id; a freed warp BVH id segfaults the next launch
                        wt = wp.Bvh(wp.array(lo_np, dtype=wp.vec3), wp.array(up_np, dtype=wp.vec3))
                        wb_ts.append(time.perf_counter() - t0)
                        t0 = time.perf_counter()
                        tb = bvh.Bvh(lo_a, up_a)
                        tb.build()
                        qd.sync()
                        qd_bts.append(time.perf_counter() - t0)
                    else:
                        t0 = time.perf_counter()
                        tb = bvh.Bvh(lo_a, up_a)
                        tb.build()
                        qd.sync()
                        qd_bts.append(time.perf_counter() - t0)
                        t0 = time.perf_counter()
                        wt = wp.Bvh(wp.array(lo_np, dtype=wp.vec3), wp.array(up_np, dtype=wp.vec3))
                        wb_ts.append(time.perf_counter() - t0)
                row("build", "wp-cpu", wb_ts)
                row("build", "qd-cpu", qd_bts)

            for kind in ("aabb", "ray", "sphere"):
                wts = []
                qtts = []
                qtot = -1
                for r in range(REPS):
                    if r % 2 == 0:
                        t0 = time.perf_counter()
                        wp.launch(wkern[kind], dim=nq, inputs=wargs[kind])
                        wp.synchronize()
                        wts.append(time.perf_counter() - t0)
                        ts_q, qtot = bench_qd_query(tree, kind, qargs[kind], cap, 1, nq)
                        qtts.append(ts_q[0])
                    else:
                        ts_q, qtot = bench_qd_query(tree, kind, qargs[kind], cap, 1, nq)
                        qtts.append(ts_q[0])
                        t0 = time.perf_counter()
                        wp.launch(wkern[kind], dim=nq, inputs=wargs[kind])
                        wp.synchronize()
                        wts.append(time.perf_counter() - t0)
                wtot = int(wcnt.numpy().sum())
                # exact f32-twin agreement; mismatch also flags hit-cap truncation
                assert wtot == qtot, \
                    f"warp {kind} total {wtot} != qd {qtot} ({n}/{dist}/nq={nq})"
                if nq == 512:
                    tol = max(10, int(0.001 * want[kind]))
                    assert abs(wtot - want[kind]) <= tol, \
                        f"warp {kind} hits {wtot} vs numpy {want[kind]} ({n}/{dist})"
                row(kind, "wp-cpu", wts, nq)
                row(kind, f"qd-{ARCH}", qtts, nq, FLOOR_MS if ARCH == "metal" else None)

print(f"PERF PROBE [{ARCH}/{MODE}]: DONE")
sys.exit(0)
