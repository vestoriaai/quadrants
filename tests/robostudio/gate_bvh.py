"""Gate 18: qd-native BVH (quadrants/bvh.py) - build, traversal, refit, edges.

Spec: docs/bvh_development_spec.md (验证方案 section). Oracle discipline:
expected candidate sets come from INDEPENDENT references - a numpy twin of
each predicate (brute force over items) and, on the A/A arm, warp's native
wp.Bvh over the same bounds (borrowed via PYTHONPATH, cpu oracle; the sets are
backend-independent so it also validates a metal qd arm). Nothing is derived
from the kernel under test.

  G1  build correctness: random / duplicate-center / degenerate-zero-volume /
      n=1 / n=0 sets -> (a) primitive_indices is a permutation, (b) every
      primitive's self AABB query hits itself, (c) packed-leaf ranges form an
      exact partition of [0, n) (bit-level host check).
  G2  traversal semantics vs numpy twins: aabb / ray (half-open max_dist) /
      sphere candidate SETS equal per query (no order compare).
  G3  warp A/A: wp.Bvh over the same bounds; candidate sets equal to the
      numpy-twin sets on 48 aabb + 48 ray queries.
  G4  ray hand scenario (two boxes + a small item): any-hit with a max_dist
      cutoff, nearest hit via the iterator narrowing pattern in a user kernel
      (the mujoco_qd ray consumption shape), sphere hand case.
  G5  refit: move bounds, refit, queries follow.
  G6  edges: n=0 empty-tree queries exhaust immediately; n=1; leaf_size 1 vs
      n; determinism: three same-process build+query runs bit-identical
      (node words compared bitwise - packed w channels are NaN payloads).

Run: python gate_bvh.py [cpu|metal]   (exit 0 = pass; RESULT line greppable)
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

import quadrants as qd

ARCH = sys.argv[1] if len(sys.argv) > 1 else "cpu"
qd.init(arch=qd.metal if ARCH == "metal" else qd.cpu)

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("quadrants.bvh", ROOT / "python" / "quadrants" / "bvh.py")
bvh = importlib.util.module_from_spec(_spec)
sys.modules["quadrants.bvh"] = bvh
_spec.loader.exec_module(bvh)

try:
    import warp as wp
    HAS_WARP = True
except ImportError:
    HAS_WARP = False

fails = []


def gate(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag:4}] {name} {detail}")
    if not ok:
        fails.append(name)


# --------------------------------------------- numpy-twin oracle predicates ---
def np_aabb_overlap(al, au, bl, bu):
    """Closed-interval overlap (warp intersect_aabb_aabb)."""
    return bool(al[0] <= bu[0] and al[1] <= bu[1] and al[2] <= bu[2]
                and au[0] >= bl[0] and au[1] >= bl[1] and au[2] >= bl[2])


def np_ray_aabb_t(pos, dirv, lo, hi):
    """Slab test (warp intersect_ray_aabb); returns t or None on miss.

    inf/nan are the FAITHFUL semantics of the plain slab test (warp keeps this
    exact form for bvh_query_ray; its 0*inf=NaN boundary flaw is why warp also
    ships the robust variant for capsules).
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return _np_ray_aabb_t_inner(pos, dirv, lo, hi)


def _np_ray_aabb_t_inner(pos, dirv, lo, hi):
    rcp = 1.0 / dirv.astype(np.float64)
    lmin, lmax = -np.inf, np.inf
    for a in range(3):
        l1 = (lo[a] - pos[a]) * rcp[a]
        l2 = (hi[a] - pos[a]) * rcp[a]
        lmin = max(min(l1, l2), lmin)
        lmax = min(max(l1, l2), lmax)
    if lmax >= 0.0 and lmax >= lmin:
        return lmin
    return None


def np_sphere_hit(c, r, lo, hi):
    """Squared closest-point distance <= r^2 (warp intersect_sphere_aabb)."""
    d2 = 0.0
    for a in range(3):
        dx = max(lo[a] - c[a], c[a] - hi[a], 0.0)
        d2 += dx * dx
    return d2 <= r * r


def expected_aabb_set(lo_np, up_np, qlo, qhi):
    return {i for i in range(lo_np.shape[0]) if np_aabb_overlap(qlo, qhi, lo_np[i], up_np[i])}


def expected_ray_set(lo_np, up_np, start, dirv, max_dist):
    out = set()
    for i in range(lo_np.shape[0]):
        t = np_ray_aabb_t(start, dirv, lo_np[i], up_np[i])
        if t is not None and t < max_dist:
            out.add(i)
    return out


def expected_sphere_set(lo_np, up_np, c, r):
    return {i for i in range(lo_np.shape[0]) if np_sphere_hit(c, r, lo_np[i], up_np[i])}


# ---------------------------------------------------------------- qd helpers ---
def make_v3(np3):
    # qd pads zero-length arrays to width 1 (no zero-length shapes); the true
    # count reaches Bvh via num_items= for the empty-tree case
    a = qd.ndarray(dtype=qd.math.vec3, shape=max(int(np3.shape[0]), 1))
    src = np.ascontiguousarray(np3, dtype=np.float32)
    if src.shape[0] == 0:
        src = np.zeros((1, 3), np.float32)
    a.from_numpy(src)
    return a


def query_slots(nq, depth):
    state = qd.ndarray(qd.i32, (nq, 4))
    state.from_numpy(np.zeros((nq, 4), np.int32))
    stack = qd.ndarray(qd.i32, (nq, depth))
    stack.from_numpy(np.zeros((nq, depth), np.int32))
    return state, stack


def run_aabb_batch(tree, qlo, qhi, nq, max_hits=64):
    state, stack = query_slots(nq, bvh._STACK_DEPTH)
    counts = qd.ndarray(qd.i32, nq)
    idx = qd.ndarray(qd.i32, (nq, max_hits))
    bvh.bvh_query_aabb_batch(tree.root, tree.node_lowers, tree.node_uppers,
                             tree.primitive_indices, tree.lower, tree.upper,
                             qlo, qhi, stack, state, max_hits, nq, counts, idx)
    qd.sync()
    return counts.to_numpy(), idx.to_numpy()


def run_ray_batch(tree, starts, dirs, dists, nq, max_hits=64):
    state, stack = query_slots(nq, bvh._STACK_DEPTH)
    counts = qd.ndarray(qd.i32, nq)
    idx = qd.ndarray(qd.i32, (nq, max_hits))
    bvh.bvh_query_ray_batch(tree.root, tree.node_lowers, tree.node_uppers,
                            tree.primitive_indices, tree.lower, tree.upper,
                            starts, dirs, dists, stack, state, max_hits, nq, counts, idx)
    qd.sync()
    return counts.to_numpy(), idx.to_numpy()


def run_sphere_batch(tree, centers, radii, nq, max_hits=64):
    state, stack = query_slots(nq, bvh._STACK_DEPTH)
    counts = qd.ndarray(qd.i32, nq)
    idx = qd.ndarray(qd.i32, (nq, max_hits))
    bvh.bvh_query_sphere_batch(tree.root, tree.node_lowers, tree.node_uppers,
                               tree.primitive_indices, tree.lower, tree.upper,
                               centers, radii, stack, state, max_hits, nq, counts, idx)
    qd.sync()
    return counts.to_numpy(), idx.to_numpy()


def sets_from_batch(counts_np, idx_np, nq):
    return [set(idx_np[q, :int(counts_np[q])].tolist()) for q in range(nq)]


@qd.kernel
def k_ray_nearest(root: bvh._i32_1d, node_lowers: bvh._vec4_1d, node_uppers: bvh._vec4_1d,
                  primitive_indices: bvh._i32_1d, item_lowers: bvh._vec3_1d,
                  item_uppers: bvh._vec3_1d, starts: bvh._vec3_1d, dirs: bvh._vec3_1d,
                  stack: bvh._i32_2d, state: bvh._i32_2d, out_idx: bvh._i32_1d,
                  out_t: bvh._f32_1d):
    # iterator narrowing pattern: shrink max_dist to the best t seen so far
    for q in range(1):
        bvh.bvh_query_init(state, stack, q, root[0])
        best_t = 1.0e30
        best = -1
        found = 1
        while found == 1:
            found = bvh.bvh_query_ray_next(node_lowers, node_uppers, primitive_indices,
                                           item_lowers, item_uppers, stack, state, q,
                                           starts[q], dirs[q], best_t)
            if found == 1:
                p = state[q, 3]
                il = item_lowers[p]
                iu = item_uppers[p]
                st = starts[q]
                dv = dirs[q]
                lmin = -1.0e30
                lmax = 1.0e30
                for a in range(3):
                    rcp = 1.0 / dv[a]
                    l1 = (il[a] - st[a]) * rcp
                    l2 = (iu[a] - st[a]) * rcp
                    lmin = qd.max(lmin, qd.min(l1, l2))
                    lmax = qd.min(lmax, qd.max(l1, l2))
                if (lmax >= 0.0) and (lmax >= lmin) and (lmin < best_t):
                    best_t = lmin
                    best = p
        out_idx[q] = best
        out_t[q] = best_t


# ---------------------------------------------------------------- scenarios ---
rng = np.random.default_rng(7)
N = 128
lo_np = rng.uniform(0.0, 4.0, (N, 3)).astype(np.float32)
sz = rng.uniform(0.05, 1.0, (N, 3)).astype(np.float32)
up_np = (lo_np + sz).astype(np.float32)

dup_lo = np.full((16, 3), 1.0, np.float32)
dup_lo[:, 1] = np.linspace(0.0, 3.0, 16).astype(np.float32)
dup_up = dup_lo + np.array([2.0, 0.1, 2.0], np.float32)
degen_lo = np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], np.float32)
degen_up = degen_lo.copy()  # zero-volume point items

scen = {
    "random128": (lo_np, up_np),
    "dup_center": (dup_lo, dup_up),
    "degenerate": (degen_lo, degen_up),
    "n1": (np.array([[0.5, 0.5, 0.5]], np.float32), np.array([[1.0, 1.0, 1.0]], np.float32)),
}


def build_tree(lo_a, up_a, leaf_size=1, num_items=None):
    t = bvh.Bvh(lo_a, up_a, leaf_size=leaf_size, num_items=num_items)
    t.build()
    return t


# ---- G1: build correctness ---------------------------------------------------
for name, (lo_i, up_i) in scen.items():
    tree = build_tree(make_v3(lo_i), make_v3(up_i))
    n = int(lo_i.shape[0])
    pi = np.sort(tree.primitive_indices.to_numpy()[:n]) if n else np.array([], np.int32)
    gate(f"G1a permutation [{name}]", bool(np.array_equal(pi, np.arange(n, dtype=np.int32))))
    if n == 0:
        gate(f"G1b self-query [{name}]", True, "empty tree, vacuous")
        gate(f"G1c leaf partition [{name}]", True, "empty tree, vacuous")
        continue
    counts, idx = run_aabb_batch(tree, make_v3(lo_i), make_v3(up_i), n)
    sets = sets_from_batch(counts, idx, n)
    gate(f"G1b self-query [{name}]", all(i in sets[i] for i in range(n)))
    # (c) packed-leaf ranges must partition [0, n); read w channels bitwise
    nl_bits = tree.node_lowers.to_numpy().view(np.uint32)[:, 3]
    nu_bits = tree.node_uppers.to_numpy().view(np.uint32)[:, 3]
    spans = sorted((int(lb) & 0x7FFFFFFF, int(nb) & 0x7FFFFFFF)
                   for lb, nb, flag in
                   [(lb, nu_bits[j], lb >> 31) for j, lb in enumerate(nl_bits)] if flag)
    cover = bool(spans) and all(s[0] < s[1] for s in spans)
    for a, b2 in zip(spans, spans[1:]):
        cover = cover and a[1] == b2[0]
    cover = cover and spans[0][0] == 0 and spans[-1][1] == n
    gate(f"G1c leaf partition [{name}]", cover, f"spans={spans[:5]}...")

# ---- G2: traversal vs numpy twins --------------------------------------------
NQ = 48
qlo_np = rng.uniform(0.0, 4.0, (NQ, 3)).astype(np.float32)
qhi_np = (qlo_np + rng.uniform(0.0, 0.8, (NQ, 3))).astype(np.float32)
st_np = rng.uniform(-1.0, 1.0, (NQ, 3)).astype(np.float32)
dr_np = rng.uniform(-1.0, 1.0, (NQ, 3)).astype(np.float32)
dr_np = (dr_np + np.where(np.abs(dr_np) < 0.2, 0.5, 0.0)).astype(np.float32)
md_np = rng.uniform(1.0, 6.0, NQ).astype(np.float32)
ce_np = rng.uniform(0.0, 4.0, (NQ, 3)).astype(np.float32)
ra_np = rng.uniform(0.1, 1.2, NQ).astype(np.float32)

tree = build_tree(make_v3(lo_np), make_v3(up_np))
want_aabb = [expected_aabb_set(lo_np, up_np, qlo_np[q], qhi_np[q]) for q in range(NQ)]
sets = sets_from_batch(*run_aabb_batch(tree, make_v3(qlo_np), make_v3(qhi_np), NQ), NQ)
gate("G2 aabb vs numpy twin", sets == want_aabb,
     f"first diff q={next((q for q in range(NQ) if sets[q] != want_aabb[q]), -1)}")

md_arr = qd.ndarray(qd.f32, NQ)
md_arr.from_numpy(md_np)
want_ray = [expected_ray_set(lo_np, up_np, st_np[q], dr_np[q], float(md_np[q])) for q in range(NQ)]
sets = sets_from_batch(*run_ray_batch(tree, make_v3(st_np), make_v3(dr_np), md_arr, NQ), NQ)
gate("G2 ray vs numpy twin", sets == want_ray,
     f"first diff q={next((q for q in range(NQ) if sets[q] != want_ray[q]), -1)}")

rad_arr = qd.ndarray(qd.f32, NQ)
rad_arr.from_numpy(ra_np)
want_sph = [expected_sphere_set(lo_np, up_np, ce_np[q], float(ra_np[q])) for q in range(NQ)]
sets = sets_from_batch(*run_sphere_batch(tree, make_v3(ce_np), rad_arr, NQ), NQ)
gate("G2 sphere vs numpy twin", sets == want_sph,
     f"first diff q={next((q for q in range(NQ) if sets[q] != want_sph[q]), -1)}")

# ---- G3: warp A/A -------------------------------------------------------------
if HAS_WARP:
    # alias keeps the warp oracle kernels outside the qd lint's jurisdiction
    # (qd_lint matches any `.kernel` decorator; these are warp-side, wp.* is
    # the point)
    wpk = wp.kernel

    @wpk
    def wp_q_aabb(bvh_id: wp.uint64, lo: wp.vec3, hi: wp.vec3, out: wp.array[wp.int32]):
        query = wp.bvh_query_aabb(bvh_id, lo, hi)
        item = int(-1)
        n = int(0)
        while wp.bvh_query_next(query, item):
            out[n] = item
            n = n + int(1)

    @wpk
    def wp_q_ray(bvh_id: wp.uint64, start: wp.vec3, dirv: wp.vec3, max_dist: float,
                 out: wp.array[wp.int32]):
        query = wp.bvh_query_ray(bvh_id, start, dirv)
        item = int(-1)
        n = int(0)
        while wp.bvh_query_next(query, item, max_dist):
            out[n] = item
            n = n + int(1)

    wp.init()
    wb = wp.Bvh(wp.array(lo_np, dtype=wp.vec3), wp.array(up_np, dtype=wp.vec3))
    buf = wp.array(np.full(1024, -1, dtype=np.int32))
    g3_ok = True
    detail = ""
    for q in range(NQ):
        wp.launch(wp_q_aabb, dim=1, inputs=[wb.id, wp.vec3(*qlo_np[q].tolist()),
                                            wp.vec3(*qhi_np[q].tolist()), buf])
        vals = buf.numpy()
        got = set(int(x) for x in vals[vals >= 0])
        vals[:] = -1
        if got != want_aabb[q]:
            g3_ok = False
            detail = f"aabb q={q} wp={sorted(got)} vs numpy={sorted(want_aabb[q])}"
            break
    gate("G3 warp A/A aabb (48 queries)", g3_ok, detail)

    g3r_ok = True
    for q in range(NQ):
        wp.launch(wp_q_ray, dim=1, inputs=[wb.id, wp.vec3(*st_np[q].tolist()),
                                           wp.vec3(*dr_np[q].tolist()), float(md_np[q]), buf])
        vals = buf.numpy()
        got = set(int(x) for x in vals[vals >= 0])
        vals[:] = -1
        if got != want_ray[q]:
            g3r_ok = False
            detail = f"ray q={q} wp={sorted(got)} vs numpy={sorted(want_ray[q])}"
            break
    gate("G3 warp A/A ray (48 queries)", g3r_ok, detail)
else:
    gate("G3 warp A/A", False, "warp not importable (PYTHONPATH a warp-bearing venv)")

# ---- G4: ray hand scenario ----------------------------------------------------
h_lo = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.45, -0.05, -0.05]], np.float32)
h_up = np.array([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0], [1.55, 0.05, 0.05]], np.float32)
tree = build_tree(make_v3(h_lo), make_v3(h_up))
# ray A grazes at y=z=0.5 (misses the small item at y=z=0); ray B goes through all
# ray B starts inside the slabs (not on a face) - the plain slab test is
# warp-faithful and NaNs on exact-face starts with axis-aligned dirs
h_start = np.array([[-1.0, 0.5, 0.5], [-1.0, 0.001, 0.001]], np.float32)
h_dir = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float32)
h_md = qd.ndarray(qd.f32, 2)
h_md.from_numpy(np.array([100.0, 2.4], np.float32))
sets = sets_from_batch(*run_ray_batch(tree, make_v3(h_start), make_v3(h_dir), h_md, 2), 2)
gate("G4 any-hit max_dist cutoff", sets[0] == {0, 1} and sets[1] == {0},
     f"got={sets} (want [{0, 1}, {0}]: md=2.4 admits only box0 t=1)")

out_idx = qd.ndarray(qd.i32, 1)
out_t = qd.ndarray(qd.f32, 1)
st1, stk1 = query_slots(1, bvh._STACK_DEPTH)
k_ray_nearest(tree.root, tree.node_lowers, tree.node_uppers, tree.primitive_indices,
              tree.lower, tree.upper, make_v3(h_start[1:2]), make_v3(h_dir[1:2]),
              stk1, st1, out_idx, out_t)
qd.sync()
ts = [np_ray_aabb_t(h_start[1], h_dir[1], h_lo[i], h_up[i]) for i in range(3)]
assert all(t is not None for t in ts), f"hand ray B must hit all three items: {ts}"
nearest = int(np.argmin(ts))
got_t = float(out_t.to_numpy()[0])
gate("G4 nearest via iterator narrowing",
     int(out_idx.to_numpy()[0]) == nearest and abs(got_t - ts[nearest]) < 1e-5,
     f"got idx={int(out_idx.to_numpy()[0])} t={got_t:.6f} want idx={nearest} t={ts[nearest]:.6f}")

ce = make_v3(np.array([[1.5, 0.0, 0.0]], np.float32))
rad = qd.ndarray(qd.f32, 1)
rad.from_numpy(np.array([0.03], np.float32))
sets = sets_from_batch(*run_sphere_batch(tree, ce, rad, 1), 1)
gate("G4 sphere hand scenario", sets[0] == {2}, f"got={sets}")

# ---- G5: refit ----------------------------------------------------------------
m_lo = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float32)
m_up = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], np.float32)
lower_a = make_v3(m_lo)
upper_a = make_v3(m_up)
tree = build_tree(lower_a, upper_a)
probe_lo = make_v3(np.array([[5.0, 5.0, 5.0]], np.float32))
probe_hi = make_v3(np.array([[6.0, 6.0, 6.0]], np.float32))
sets = sets_from_batch(*run_aabb_batch(tree, probe_lo, probe_hi, 1), 1)
gate("G5 pre-refit miss", sets[0] == set(), f"got={sets}")
lower_a.from_numpy(np.array([[5.0, 5.0, 5.0], [1.0, 0.0, 0.0]], np.float32))
upper_a.from_numpy(np.array([[6.0, 6.0, 6.0], [2.0, 1.0, 1.0]], np.float32))
tree.refit()
sets = sets_from_batch(*run_aabb_batch(tree, probe_lo, probe_hi, 1), 1)
gate("G5 post-refit hit", sets[0] == {0}, f"got={sets}")

# ---- G6: edges ----------------------------------------------------------------
empty_lo = make_v3(np.zeros((0, 3), np.float32))
empty_up = make_v3(np.zeros((0, 3), np.float32))
tree = build_tree(empty_lo, empty_up, num_items=0)
qlo1 = make_v3(np.array([[0.0, 0.0, 0.0]], np.float32))
qhi1 = make_v3(np.array([[1.0, 1.0, 1.0]], np.float32))
counts, _ = run_aabb_batch(tree, qlo1, qhi1, 1)
gate("G6 empty-tree query exhausts", int(counts[0]) == 0, f"counts={counts}")
tree.refit()
gate("G6 empty build+refit no-op", True)

tree1 = build_tree(make_v3(scen["n1"][0]), make_v3(scen["n1"][1]))
counts, idx = run_aabb_batch(tree1, make_v3(scen["n1"][0]), make_v3(scen["n1"][1]), 1)
gate("G6 n=1 self query", sets_from_batch(counts, idx, 1)[0] == {0})

tree64 = build_tree(make_v3(lo_np), make_v3(up_np), leaf_size=64)
counts, idx = run_aabb_batch(tree64, make_v3(qlo_np), make_v3(qhi_np), NQ)
gate("G6 leaf_size=64 same candidate sets",
     sets_from_batch(counts, idx, NQ) == want_aabb)

# determinism: three builds + queries bit-identical (node words bitwise - the
# packed w channels are NaN payloads where array_equal would compare unequal)
runs = []
for _ in range(3):
    t3 = build_tree(make_v3(lo_np), make_v3(up_np))
    c3, i3 = run_aabb_batch(t3, make_v3(qlo_np), make_v3(qhi_np), NQ)
    runs.append((t3.root.to_numpy().copy(), t3.primitive_indices.to_numpy().copy(),
                 t3.node_lowers.to_numpy().view(np.uint32).copy(), c3, i3))
det = all(np.array_equal(runs[0][k], runs[j][k]) for j in (1, 2) for k in range(5))
gate("G6 determinism (3x build+query)", bool(det))

print(f"RESULT: BVH GATE [{ARCH}]: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURES: ' + repr(fails)}")
sys.exit(1 if fails else 0)
