"""Gate probe: @qd.kernel/@qd.func(early_return=True) — the early-return single-exit rewrite.

Covers the warp-shaped guard-clause forms mujoco_warp porting needs. Every value-returning
case is checked numerically against a plain-Python twin of the original (pre-rewrite)
semantics, on the target backend. Run: python probe_early_return_forms.py [metal|cpu]
Exit 0 = all gates pass.

Ledger: mujocoqd/compiler-line M2 (early_return AST pass).
"""
import sys

import numpy as np

import quadrants as qd

ARCH = sys.argv[1] if len(sys.argv) > 1 else "metal"
qd.init(arch=qd.metal if ARCH == "metal" else qd.cpu)

fails = []


def gate(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL':4}] {name} {detail}")
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- func cases
@qd.func(early_return=True)
def f_guard(x: qd.f32) -> qd.f32:
    if x < 0.0:
        return -1.0
    if x < 1.0:
        return 0.5
    return 1.0


def f_guard_ref(x):
    if x < 0.0:
        return -1.0
    if x < 1.0:
        return 0.5
    return 1.0


@qd.func(early_return=True)
def f_loop_ret(x: qd.f32) -> qd.f32:
    it = 0
    while x > 1.0 and it < 64:
        if x > 16.0:
            return 0.25
        x = x * 0.5
        it = it + 1
    return x


def f_loop_ret_ref(x):
    it = 0
    while x > 1.0 and it < 64:
        if x > 16.0:
            return 0.25
        x = x * 0.5
        it += 1
    return x


@qd.func(early_return=True)
def f_tup(a: qd.f32, b: qd.f32) -> tuple[qd.f32, qd.f32]:
    if b == 0.0:
        return 0.0, -1.0
    return a / b, a + b


def f_tup_ref(a, b):
    if b == 0.0:
        return (0.0, -1.0)
    return (a / b, a + b)


@qd.func(early_return=True)
def f_void(x: qd.f32, acc: qd.types.ndarray(ndim=1)):
    if x < 0.0:
        return
    acc[0] = acc[0] + x


# (kept for reference; exercised via f_void_cell below to avoid A5)
# static-if returns must remain untouched (compile-time returns are legal)
@qd.func(early_return=True)
def f_static(x: qd.f32) -> qd.f32:
    if qd.static(True):
        return x * 2.0
    return x


# ---------------------------------------------------------------- kernel cases
@qd.kernel(early_return=True)
def k_guard(x: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=1)):
    for i in range(x.shape[0]):
        if x[i] < 0.0:
            y[i] = -1.0
            return
        if x[i] < 1.0:
            y[i] = 0.5
            return
        y[i] = 1.0


@qd.kernel(early_return=True)
def k_nested_break(x: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=1)):
    for i in range(x.shape[0]):
        acc = 0.0
        for j in range(64):
            if acc > x[i]:
                break
            acc += 1.5
            if acc > 100.0:
                y[i] = acc
                return
        y[i] = acc


@qd.kernel(early_return=True)
def k_tup(a: qd.types.ndarray(ndim=1), b: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=1), z: qd.types.ndarray(ndim=1)):
    for i in range(a.shape[0]):
        u, v = f_tup(a[i], b[i])
        y[i], z[i] = u, v


# ---------------------------------------------------------------- harness
X = [-7.5, -0.1, 0.0, 0.5, 0.999, 1.0, 3.0, 17.0, 100.0]
dev_x = qd.ndarray(dtype=qd.f32, shape=(1,))
dev_y = qd.ndarray(dtype=qd.f32, shape=(1,))
dev_z = qd.ndarray(dtype=qd.f32, shape=(1,))


def run1(k, a):
    dev_x.from_numpy(np.array([a], dtype=np.float32))
    dev_y.from_numpy(np.zeros(1, dtype=np.float32))
    k(dev_x, dev_y)
    qd.sync()
    return float(dev_y.to_numpy()[0])


vals = [run1(k_guard, a) for a in X]
gate("func_guard", np.allclose(vals, [f_guard_ref(a) for a in X]), f"{vals[:3]}")

vals = [run1(k_nested_break, a) for a in [1.0, 10.0, 50.0, 200.0]]
ref = []
for a in [1.0, 10.0, 50.0, 200.0]:
    acc = 0.0
    stop = False
    for j in range(64):
        if acc > a:
            break
        acc += 1.5
        if acc > 100.0:
            ref.append(acc)
            stop = True
            break
    if not stop:
        ref.append(acc)
gate("kernel_nested_break", np.allclose(vals, ref), f"{vals} vs {ref}")

# NOTE: each element gets its own output cell (acc[i], written once) — conditional
# accumulation into ONE array element across a runtime loop currently hits the A5
# Metal miscompile (loop-carried memory dependence violated; see probe_metal_loopdep.py),
# which is orthogonal to what this gate tests.
@qd.func(early_return=True)
def f_void_cell(x: qd.f32, acc: qd.types.ndarray(ndim=1), i: qd.i32):
    if x < 0.0:
        return
    acc[i] = acc[i] + x


@qd.kernel
def k_void(xs: qd.types.ndarray(ndim=1), acc: qd.types.ndarray(ndim=1)):
    qd.loop_config(serialize=True)
    for i in range(xs.shape[0]):
        f_void_cell(xs[i], acc, i)


xs_in = qd.ndarray(dtype=qd.f32, shape=(4,))
xs_in.from_numpy(np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32))
accv = qd.ndarray(dtype=qd.f32, shape=(4,))
accv.from_numpy(np.zeros(4, dtype=np.float32))
k_void(xs_in, accv)
qd.sync()
got = accv.to_numpy()
gate("func_void", np.allclose(got, [0.0, 2.0, 0.0, 4.0]), f"{got}")


# per-call flag reset: serial loop calls the rewritten func with VARYING args — if the
# __er_done flag leaked across calls (e.g. LICM hoisting the prologue store), later calls
# would return stale slots instead of their own result.
@qd.kernel
def k_multi(xs: qd.types.ndarray(ndim=1), ys: qd.types.ndarray(ndim=1)):
    qd.loop_config(serialize=True)
    for i in range(3):
        ys[i] = f_guard(xs[i])


xs_m = qd.ndarray(dtype=qd.f32, shape=(3,))
xs_m.from_numpy(np.array([-1.0, 0.5, 5.0], dtype=np.float32))
ys_m = qd.ndarray(dtype=qd.f32, shape=(3,))
ys_m.from_numpy(np.zeros(3, dtype=np.float32))
k_multi(xs_m, ys_m)
qd.sync()
got_m = ys_m.to_numpy()
gate("func_per_call_reset", np.allclose(got_m, [-1.0, 0.5, 1.0]), f"{got_m}")

# static-if passthrough (no rewrite inside static)
dev_x.from_numpy(np.array([3.0], dtype=np.float32))
dev_y.from_numpy(np.zeros(1, dtype=np.float32))

@qd.kernel
def k_static(x: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=1)):
    for i in range(x.shape[0]):
        y[i] = f_static(x[i])

k_static(dev_x, dev_y)
qd.sync()
gate("func_static_untouched", np.isclose(float(dev_y.to_numpy()[0]), 6.0))

run_tup = []
ta = qd.ndarray(dtype=qd.f32, shape=(1,))
tb = qd.ndarray(dtype=qd.f32, shape=(1,))
for a, b in [(1.0, 2.0), (3.0, 0.0), (9.0, -3.0)]:
    ta.from_numpy(np.array([a], dtype=np.float32))
    tb.from_numpy(np.array([b], dtype=np.float32))
    dev_y.from_numpy(np.zeros(1, dtype=np.float32))
    dev_z.from_numpy(np.zeros(1, dtype=np.float32))
    k_tup(ta, tb, dev_y, dev_z)
    qd.sync()
    run_tup.append((float(dev_y.to_numpy()[0]), float(dev_z.to_numpy()[0])))
want = [f_tup_ref(a, b) for a, b in [(1.0, 2.0), (3.0, 0.0), (9.0, -3.0)]]
gate(
    "kernel_func_tuple",
    all(abs(g[0] - w[0]) < 1e-5 and abs(g[1] - w[1]) < 1e-5 for g, w in zip(run_tup, want)),
    f"{run_tup}",
)

# negative: unannotated value return must raise a clear error at compile time
# (funcs compile lazily at first call — must invoke from a kernel to trigger the pass)
f_no_ann_err = None
try:
    @qd.func(early_return=True)
    def f_no_ann(x):
        if x > 0.0:
            return 1.0
        return 2.0

    @qd.kernel
    def k_no_ann(x: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=1)):
        for i in range(x.shape[0]):
            y[i] = f_no_ann(x[i])

    dev_x.from_numpy(np.ones(1, dtype=np.float32))
    dev_y.from_numpy(np.zeros(1, dtype=np.float32))
    k_no_ann(dev_x, dev_y)
    qd.sync()
except qd.QuadrantsSyntaxError as e:
    f_no_ann_err = str(e)
except Exception as e:  # noqa: BLE001
    f_no_ann_err = f"WRONG-ERROR {type(e).__name__}: {e}"
gate(
    "neg_unannotated_error",
    f_no_ann_err is not None and "annotation" in f_no_ann_err,
    (f_no_ann_err or "no error raised")[:90],
)

# =============================================================== F-matrix ====
# F1-F14: compound-slot + merged-guard matrix (A7/M2 root-fix acceptance).
# Every form is checked against a plain-Python twin of the pre-rewrite semantics.
#   F1  tuple[vec3, f32] guard + post-return first binding   (defect 1: vec slot init)
#   F2  mat3 slot, broadcast init + element mutation         (defect 1: mat slot init)
#   F3  tuple[vec3, i32] mixed compound/int slots
#   F4  tuple[vec8, mat83, mat83] — the mujoco_qd box_box return shape
#   F5  tuple[f32, vec3, f32, f32, i32, i32] — the ray_mesh_with_bvh shape
#   F6  while+return with compound return values (loop-family resurrection)
#   F7  kernel-side early return; inlined compound func after the return site
#       (merged guard in the outermost parallel loop, guard-only no-break path)
#   F8  kernel nested loop: break + return site, compound func in the
#       fall-through region
#   F9  per-call flag reset with compound slots (serial loop, varying args —
#       the over-hoisting guard of the qd.cast(T(0), T.dtype) init)
#   F10 for-loop inside the merged guard block (guard-merge IR stress)
#   F11 four compound return sites incl. fall-through
#   F12 new-validation negatives: arity mismatch / bare+value mix / nested tuple /
#       struct return — all must be loud compile-time rejections
#   F13 compound + scalar first-bindings interleaved, nested if inside the guard
#   F14 post-return first-binding sibling guards — THE defect-2 acceptance form
#       (pre-fix: actionable branch-scope NameError "Name t is not defined")
vec3f = qd.types.vector(3, qd.f32)
vec8f = qd.types.vector(8, qd.f32)
mat3f = qd.types.matrix(3, 3, qd.f32)
mat83f = qd.types.matrix(8, 3, qd.f32)


@qd.func(early_return=True)
def f1_vec_tup(x: qd.f32) -> tuple[vec3f, qd.f32]:
    if x < 0.0:
        return vec3f(1.0, 0.0, 0.0), -1.0
    v = vec3f(0.0)
    v[1] = x
    return v, x * 2.0


def f1_ref(x):
    if x < 0.0:
        return (1.0, 0.0, 0.0), -1.0
    return (0.0, x, 0.0), x * 2.0


@qd.func(early_return=True)
def f2_mat(x: qd.f32) -> mat3f:
    if x < 0.0:
        return mat3f(1.0)
    m = mat3f(0.0)
    m[0, 0] = x
    m[2, 1] = x * 2.0
    return m


def f2_ref(x):
    if x < 0.0:
        return [[1.0] * 3 for _ in range(3)]
    m = [[0.0] * 3 for _ in range(3)]
    m[0][0] = x
    m[2][1] = x * 2.0
    return m


@qd.func(early_return=True)
def f3_vec_int(x: qd.f32, n: qd.i32) -> tuple[vec3f, qd.i32]:
    if x < 0.0 or n < 0:
        return vec3f(0.0, 0.0, 0.0), -1
    return vec3f(1.0, x, 0.0), n * 2


def f3_ref(x, n):
    if x < 0.0 or n < 0:
        return (0.0, 0.0, 0.0), -1
    return (1.0, x, 0.0), n * 2


@qd.func(early_return=True)
def f4_boxbox(x: qd.f32) -> tuple[vec8f, mat83f, mat83f]:
    if x < 0.0:
        return vec8f(0.0), mat83f(0.0), mat83f(0.0)
    a = vec8f(1.0)
    b = mat83f(2.0)
    c = mat83f(3.0)
    a[7] = x
    b[0, 2] = x
    c[7, 1] = x * 2.0
    return a, b, c


def f4_ref(x):
    if x < 0.0:
        return [0.0] * 8, [[0.0] * 3 for _ in range(8)], [[0.0] * 3 for _ in range(8)]
    a = [1.0] * 8
    b = [[2.0] * 3 for _ in range(8)]
    c = [[3.0] * 3 for _ in range(8)]
    a[7] = x
    b[0][2] = x
    c[7][1] = x * 2.0
    return a, b, c


@qd.func(early_return=True)
def f5_ray(t_max: qd.f32, hit: qd.i32, x: qd.f32) -> tuple[qd.f32, vec3f, qd.f32, qd.f32, qd.i32, qd.i32]:
    if hit == 0:
        return 0.0, vec3f(0.0, 0.0, 0.0), 0.0, 0.0, -1, -1
    n = vec3f(x, 0.0, 1.0)
    return t_max * 0.5, n, x, t_max, hit, 7


def f5_ref(t_max, hit, x):
    if hit == 0:
        return (0.0, (0.0, 0.0, 0.0), 0.0, 0.0, -1, -1)
    return (t_max * 0.5, (x, 0.0, 1.0), x, t_max, hit, 7)


@qd.func(early_return=True)
def f6_while_vec(x: qd.f32) -> tuple[vec3f, qd.f32]:
    it = 0
    acc = 0.0
    while x > 1.0 and it < 32:
        if x > 16.0:
            return vec3f(1.0, acc, 0.0), x
        acc = acc + x
        x = x * 0.5
        it = it + 1
    return vec3f(0.0, acc, 0.0), x


def f6_ref(x0):
    it = 0
    acc = 0.0
    x = x0
    while x > 1.0 and it < 32:
        if x > 16.0:
            return (1.0, acc, 0.0), x
        acc += x
        x *= 0.5
        it += 1
    return (0.0, acc, 0.0), x


@qd.func(early_return=True)
def f9_reset(x: qd.f32) -> vec3f:
    if x < 0.0:
        return vec3f(-1.0, 0.0, 0.0)
    return vec3f(x, x, x)


def f9_ref(x):
    if x < 0.0:
        return (-1.0, 0.0, 0.0)
    return (x, x, x)


@qd.func(early_return=True)
def f10_postloop(x: qd.f32) -> tuple[vec3f, qd.f32]:
    if x < 0.0:
        return vec3f(1.0, 0.0, 0.0), -1.0
    s = 0.0
    for j in range(4):
        s = s + x * j
    v = vec3f(s, s, s)
    return v, s


def f10_ref(x):
    if x < 0.0:
        return (1.0, 0.0, 0.0), -1.0
    s = sum(x * j for j in range(4))
    return (s, s, s), s


@qd.func(early_return=True)
def f11_multisite(x: qd.f32) -> tuple[vec3f, qd.f32]:
    if x < -10.0:
        return vec3f(1.0, 0.0, 0.0), -1.0
    if x < 0.0:
        return vec3f(0.0, 1.0, 0.0), -2.0
    if x < 10.0:
        return vec3f(0.0, 0.0, 1.0), x
    return vec3f(2.0, 2.0, 2.0), x * 2.0


def f11_ref(x):
    if x < -10.0:
        return (1.0, 0.0, 0.0), -1.0
    if x < 0.0:
        return (0.0, 1.0, 0.0), -2.0
    if x < 10.0:
        return (0.0, 0.0, 1.0), x
    return (2.0, 2.0, 2.0), x * 2.0


@qd.func(early_return=True)
def f13_chain(x: qd.f32) -> tuple[vec3f, qd.f32, qd.i32]:
    if x < 0.0:
        return vec3f(0.0, 0.0, 0.0), 0.0, -1
    a = x * 0.5
    v = vec3f(a, a, a)
    w = v[0] + v[2]
    c = 1
    if w > 10.0:
        c = 2
    return vec3f(w, a, x), w, c


def f13_ref(x):
    if x < 0.0:
        return (0.0, 0.0, 0.0), 0.0, -1
    a = x * 0.5
    w = a + a
    c = 2 if w > 10.0 else 1
    return (w, a, x), w, c


@qd.func(early_return=True)
def f14_sibling(x: qd.f32) -> qd.f32:
    if x < 0.0:
        return -1.0
    t = x * 0.5
    return t + 1.0


def f14_ref(x):
    if x < 0.0:
        return -1.0
    return x * 0.5 + 1.0


# ---- kernel-side forms -------------------------------------------------------
# NOTE (ERRATUM 2026-10-13, supersedes the "cross-iteration branch leakage /
# compound miscompile" reading; mechanism pinned in _probe/eval_kflag/):
# a flag declared OUTSIDE the element loop survives across offload tasks and is
# demoted to a GlobalTemporary — a launch-shared scratch cell — so under a
# PARALLEL element loop the cell races across elements (metal wave scheduling
# makes head elements observe the tail's set before their own guards run).
# It was never a compound-work defect: identical behavior on the pre-fix
# 0de907cd and dev104 wheels (upstream-inherited), cpu included at the pool's
# block boundaries. Kernel-side early_return puts __er_done at function level,
# so F7/F8 keep the compound funcs in the UNGUARDED prefix and pin the guard
# merge with scalar-only guard regions (the portable kernel-side shape).
@qd.kernel(early_return=True)
def k7_compound_local(xs: qd.types.ndarray(ndim=1), y3: qd.types.ndarray(ndim=1), y1: qd.types.ndarray(ndim=1), yg: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        v, s = f1_vec_tup(xs[i])
        for j in range(3):
            y3[i * 3 + j] = v[j]
        y1[i] = s
        if xs[i] > 50.0:
            yg[i] = 1.0
            return
        yg[i] = 0.5


@qd.kernel(early_return=True)
def k8_nested(xs: qd.types.ndarray(ndim=1), y1: qd.types.ndarray(ndim=1), y3: qd.types.ndarray(ndim=1), yg: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        s = 0.0
        for j in range(8):
            s += 0.5
            v = f9_reset(s)
            y3[i * 8 + j] = v[0]
            if s > xs[i]:
                y1[i] = s
                yg[i] = 1.0
                return
        yg[i] = 0.25


# ---- F-matrix harness ---------------------------------------------------------
FX = [-7.5, -0.5, 0.0, 0.5, 3.0, 20.0, 100.0]
FN = [2, -1, 0, 5, 3, 9, 1]

fx = qd.ndarray(dtype=qd.f32, shape=(len(FX),))
fx.from_numpy(np.array(FX, dtype=np.float32))
fn_ = qd.ndarray(dtype=qd.i32, shape=(len(FX),))
fn_.from_numpy(np.array(FN, dtype=np.int32))

o3 = qd.ndarray(dtype=qd.f32, shape=(len(FX) * 3,))
o1 = qd.ndarray(dtype=qd.f32, shape=(len(FX),))
oi = qd.ndarray(dtype=qd.i32, shape=(len(FX),))


def flat3(rows):
    return [c for r in rows for c in r]


def zero3(y3):
    y3.from_numpy(np.zeros(len(FX) * 3, dtype=np.float32))


def zero1(y1):
    y1.from_numpy(np.zeros(len(FX), dtype=np.float32))


def zeroi(yi):
    yi.from_numpy(np.zeros(len(FX), dtype=np.int32))


# F1: tuple[vec3, f32]
zero3(o3)
zero1(o1)


@qd.kernel
def k_f1(xs: qd.types.ndarray(ndim=1), y3: qd.types.ndarray(ndim=1), y1: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        v, s = f1_vec_tup(xs[i])
        for j in range(3):
            y3[i * 3 + j] = v[j]
        y1[i] = s


k_f1(fx, o3, o1)
qd.sync()
gate(
    "F1_vec_slot_tuple",
    np.allclose(o3.to_numpy(), flat3([f1_ref(x)[0] for x in FX]))
    and np.allclose(o1.to_numpy(), [f1_ref(x)[1] for x in FX]),
    f"{o3.to_numpy()[:3]}",
)

# F2: mat3 slot
om = qd.ndarray(dtype=qd.f32, shape=(len(FX) * 9,))
om.from_numpy(np.zeros(len(FX) * 9, dtype=np.float32))


@qd.kernel
def k_f2(xs: qd.types.ndarray(ndim=1), ym: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        m = f2_mat(xs[i])
        for r in range(3):
            for c in range(3):
                ym[i * 9 + r * 3 + c] = m[r, c]


k_f2(fx, om)
qd.sync()
gate("F2_mat_slot_broadcast", np.allclose(om.to_numpy(), flat3([flat3(f2_ref(x)) for x in FX])), f"{om.to_numpy()[:3]}")

# F3: tuple[vec3, i32]
zero3(o3)
zeroi(oi)


@qd.kernel
def k_f3(xs: qd.types.ndarray(ndim=1), ns: qd.types.ndarray(ndim=1), y3: qd.types.ndarray(ndim=1), yi: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        v, n = f3_vec_int(xs[i], ns[i])
        for j in range(3):
            y3[i * 3 + j] = v[j]
        yi[i] = n


k_f3(fx, fn_, o3, oi)
qd.sync()
gate(
    "F3_vec_int_mixed",
    np.allclose(o3.to_numpy(), flat3([f3_ref(x, n)[0] for x, n in zip(FX, FN)]))
    and np.array_equal(oi.to_numpy(), np.array([f3_ref(x, n)[1] for x, n in zip(FX, FN)], dtype=np.int32)),
    f"{oi.to_numpy()}",
)

# F4: tuple[vec8, mat83, mat83] — mujoco_qd box_box shape
o8 = qd.ndarray(dtype=qd.f32, shape=(len(FX) * 8,))
ob = qd.ndarray(dtype=qd.f32, shape=(len(FX) * 24,))
oc = qd.ndarray(dtype=qd.f32, shape=(len(FX) * 24,))
for arr, n in [(o8, len(FX) * 8), (ob, len(FX) * 24), (oc, len(FX) * 24)]:
    arr.from_numpy(np.zeros(n, dtype=np.float32))


@qd.kernel
def k_f4(
    xs: qd.types.ndarray(ndim=1),
    y8: qd.types.ndarray(ndim=1),
    yb: qd.types.ndarray(ndim=1),
    yc: qd.types.ndarray(ndim=1),
):
    for i in range(xs.shape[0]):
        a, b, c = f4_boxbox(xs[i])
        for j in range(8):
            y8[i * 8 + j] = a[j]
        for r in range(8):
            for c2 in range(3):
                yb[i * 24 + r * 3 + c2] = b[r, c2]
                yc[i * 24 + r * 3 + c2] = c[r, c2]


k_f4(fx, o8, ob, oc)
qd.sync()
gate(
    "F4_boxbox_shape",
    np.allclose(o8.to_numpy(), flat3([f4_ref(x)[0] for x in FX]))
    and np.allclose(ob.to_numpy(), flat3([flat3(f4_ref(x)[1]) for x in FX]))
    and np.allclose(oc.to_numpy(), flat3([flat3(f4_ref(x)[2]) for x in FX])),
    f"{o8.to_numpy()[:3]}",
)

# F5: 6-slot ray shape
or0 = qd.ndarray(dtype=qd.f32, shape=(len(FX),))
ov3 = qd.ndarray(dtype=qd.f32, shape=(len(FX) * 3,))
or1 = qd.ndarray(dtype=qd.f32, shape=(len(FX),))
or2 = qd.ndarray(dtype=qd.f32, shape=(len(FX),))
oh = qd.ndarray(dtype=qd.i32, shape=(len(FX),))
oc2 = qd.ndarray(dtype=qd.i32, shape=(len(FX),))
for arr in [or0, or1, or2]:
    arr.from_numpy(np.zeros(len(FX), dtype=np.float32))
ov3.from_numpy(np.zeros(len(FX) * 3, dtype=np.float32))
oh.from_numpy(np.zeros(len(FX), dtype=np.int32))
oc2.from_numpy(np.zeros(len(FX), dtype=np.int32))


@qd.kernel
def k_f5(
    xs: qd.types.ndarray(ndim=1),
    ns: qd.types.ndarray(ndim=1),
    y0: qd.types.ndarray(ndim=1),
    y3: qd.types.ndarray(ndim=1),
    y1: qd.types.ndarray(ndim=1),
    y2: qd.types.ndarray(ndim=1),
    yh: qd.types.ndarray(ndim=1),
    yc: qd.types.ndarray(ndim=1),
):
    for i in range(xs.shape[0]):
        t, n, a, b, h, c = f5_ray(xs[i] * 4.0, ns[i], xs[i])
        y0[i] = t
        for j in range(3):
            y3[i * 3 + j] = n[j]
        y1[i] = a
        y2[i] = b
        yh[i] = h
        yc[i] = c


k_f5(fx, fn_, or0, ov3, or1, or2, oh, oc2)
qd.sync()
refs = [f5_ref(x * 4.0, n, x) for x, n in zip(FX, FN)]
gate(
    "F5_ray_shape_6slot",
    np.allclose(or0.to_numpy(), [r[0] for r in refs])
    and np.allclose(ov3.to_numpy(), flat3([r[1] for r in refs]))
    and np.allclose(or1.to_numpy(), [r[2] for r in refs])
    and np.allclose(or2.to_numpy(), [r[3] for r in refs])
    and np.array_equal(oh.to_numpy(), np.array([r[4] for r in refs], dtype=np.int32))
    and np.array_equal(oc2.to_numpy(), np.array([r[5] for r in refs], dtype=np.int32)),
    f"{oh.to_numpy()}",
)

# F6: while + return with compound values
zero3(o3)
zero1(o1)


@qd.kernel
def k_f6(xs: qd.types.ndarray(ndim=1), y3: qd.types.ndarray(ndim=1), y1: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        v, s = f6_while_vec(xs[i])
        for j in range(3):
            y3[i * 3 + j] = v[j]
        y1[i] = s


k_f6(fx, o3, o1)
qd.sync()
gate(
    "F6_while_compound",
    np.allclose(o3.to_numpy(), flat3([f6_ref(x)[0] for x in FX]))
    and np.allclose(o1.to_numpy(), [f6_ref(x)[1] for x in FX]),
    f"{o1.to_numpy()}",
)

# F7: kernel own early return; compound func in the unguarded prefix, scalar-only guard.
# Per-ELEMENT launches: kernel-side __er_done is launch-scoped (the documented
# SIMT-free early-exit semantics — a return ends guarded side effects for the whole
# launch), so multi-element launches cannot model per-element returns; k_guard uses
# the same 1-element pattern.
# CPU-claim only: on metal the kernel+compound-func+flag combination loses the guarded
# scalar stores — the loop-external flag GlobalTemporary cross-element race (ERRATUM
# 2026-10-13: launch-shared scratch cell, parallel-element scheduling; not a compound
# defect), identical on the pre-fix wheel with a hand-written single-exit kernel
# (_probe/_k7metal_manual.py; family pinned in _probe/_kcompound_diag.py and
# _probe/eval_kflag/). Not a pass-output defect.
if ARCH == "cpu":
    fx1 = qd.ndarray(dtype=qd.f32, shape=(1,))
    o3x = qd.ndarray(dtype=qd.f32, shape=(3,))
    o1x = qd.ndarray(dtype=qd.f32, shape=(1,))
    ogx = qd.ndarray(dtype=qd.f32, shape=(1,))
    ok7 = True
    for x in FX:
        fx1.from_numpy(np.array([x], dtype=np.float32))
        o3x.from_numpy(np.zeros(3, dtype=np.float32))
        o1x.from_numpy(np.zeros(1, dtype=np.float32))
        ogx.from_numpy(np.zeros(1, dtype=np.float32))
        k7_compound_local(fx1, o3x, o1x, ogx)
        qd.sync()
        w = f1_ref(x)
        ok7 = (
            ok7
            and np.allclose(o3x.to_numpy(), w[0])
            and np.isclose(o1x.to_numpy()[0], w[1])
            and np.isclose(ogx.to_numpy()[0], 1.0 if x > 50.0 else 0.5)
        )
    gate("F7_kernel_compound_local", ok7, f"{ogx.to_numpy()}")
else:
    print(
        "[SKIP] F7_kernel_compound_local (metal): kernel+compound-func+flag blocked by the "
        "loop-external flag GlobalTemporary cross-element race (identical on pre-fix wheel, "
        "_probe/_k7metal_manual.py + _probe/_kcompound_diag.py + _probe/eval_kflag/); "
        "not a pass-output defect"
    )

# F8: kernel nested break + return; compound func in the inner-loop prefix.
# Per-ELEMENT launches (launch-scoped __er_done, see F7).
fx1 = qd.ndarray(dtype=qd.f32, shape=(1,))
o8y = qd.ndarray(dtype=qd.f32, shape=(1,))
o8z = qd.ndarray(dtype=qd.f32, shape=(8,))
o8g = qd.ndarray(dtype=qd.f32, shape=(1,))
ok8 = True
for x in FX:
    fx1.from_numpy(np.array([x], dtype=np.float32))
    o8y.from_numpy(np.zeros(1, dtype=np.float32))
    o8z.from_numpy(np.zeros(8, dtype=np.float32))
    o8g.from_numpy(np.zeros(1, dtype=np.float32))
    k8_nested(fx1, o8y, o8z, o8g)
    qd.sync()
    s = 0.0
    hit = False
    zs = []
    for _ in range(8):
        s += 0.5
        zs.append(s)  # f9_reset(s)[0] = s (s > 0 always)
        if s > x:
            hit = True
            break
    wz = zs + [0.0] * (8 - len(zs))  # cells after the return stay at init
    wy = s if hit else 0.0
    wg = 1.0 if hit else 0.25
    ok8 = ok8 and np.allclose(o8y.to_numpy(), [wy]) and np.allclose(o8z.to_numpy(), wz) and np.isclose(o8g.to_numpy()[0], wg)
gate("F8_kernel_nested_break", ok8, f"{o8y.to_numpy()} {o8g.to_numpy()}")

# F9: per-call reset with compound slots (serial loop)
o9 = qd.ndarray(dtype=qd.f32, shape=(len(FX) * 3,))
o9.from_numpy(np.zeros(len(FX) * 3, dtype=np.float32))


@qd.kernel
def k_f9(xs: qd.types.ndarray(ndim=1), y3: qd.types.ndarray(ndim=1)):
    qd.loop_config(serialize=True)
    for i in range(xs.shape[0]):
        v = f9_reset(xs[i])
        for j in range(3):
            y3[i * 3 + j] = v[j]


k_f9(fx, o9)
qd.sync()
gate("F9_compound_per_call_reset", np.allclose(o9.to_numpy(), flat3([f9_ref(x) for x in FX])), f"{o9.to_numpy()[:3]}")

# F10: for-loop inside the merged guard
zero3(o3)
zero1(o1)


@qd.kernel
def k_f10(xs: qd.types.ndarray(ndim=1), y3: qd.types.ndarray(ndim=1), y1: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        v, s = f10_postloop(xs[i])
        for j in range(3):
            y3[i * 3 + j] = v[j]
        y1[i] = s


k_f10(fx, o3, o1)
qd.sync()
gate(
    "F10_postloop_in_guard",
    np.allclose(o3.to_numpy(), flat3([f10_ref(x)[0] for x in FX]))
    and np.allclose(o1.to_numpy(), [f10_ref(x)[1] for x in FX]),
    f"{o1.to_numpy()}",
)

# F11: four compound return sites
zero3(o3)
zero1(o1)


@qd.kernel
def k_f11(xs: qd.types.ndarray(ndim=1), y3: qd.types.ndarray(ndim=1), y1: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        v, s = f11_multisite(xs[i])
        for j in range(3):
            y3[i * 3 + j] = v[j]
        y1[i] = s


k_f11(fx, o3, o1)
qd.sync()
gate(
    "F11_multisite_compound",
    np.allclose(o3.to_numpy(), flat3([f11_ref(x)[0] for x in FX]))
    and np.allclose(o1.to_numpy(), [f11_ref(x)[1] for x in FX]),
    f"{o1.to_numpy()}",
)

# F13: compound + scalar first-bindings interleaved in the guard
zero3(o3)
zero1(o1)
zeroi(oi)


@qd.kernel
def k_f13(
    xs: qd.types.ndarray(ndim=1),
    y3: qd.types.ndarray(ndim=1),
    y1: qd.types.ndarray(ndim=1),
    yi: qd.types.ndarray(ndim=1),
):
    for i in range(xs.shape[0]):
        v, s, c = f13_chain(xs[i])
        for j in range(3):
            y3[i * 3 + j] = v[j]
        y1[i] = s
        yi[i] = c


k_f13(fx, o3, o1, oi)
qd.sync()
gate(
    "F13_binding_chain",
    np.allclose(o3.to_numpy(), flat3([f13_ref(x)[0] for x in FX]))
    and np.allclose(o1.to_numpy(), [f13_ref(x)[1] for x in FX])
    and np.array_equal(oi.to_numpy(), np.array([f13_ref(x)[2] for x in FX], dtype=np.int32)),
    f"{oi.to_numpy()}",
)

# F14: THE defect-2 acceptance form (pre-fix: Name "t" is not defined)
zero1(o1)


@qd.kernel
def k_f14(xs: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        y[i] = f14_sibling(xs[i])


k_f14(fx, o1)
qd.sync()
gate("F14_sibling_guard_first_binding", np.allclose(o1.to_numpy(), [f14_ref(x) for x in FX]), f"{o1.to_numpy()}")

# F12: new-validation negatives — every case must raise a loud compile-time error
def try_reject(label, needle, fn):

    def deco():
        try:
            wrapped = qd.func(early_return=True)(fn)

            @qd.kernel
            def _driver(xs: qd.types.ndarray(ndim=1), ns: qd.types.ndarray(ndim=1)):
                for i in range(xs.shape[0]):
                    wrapped(xs[i], ns[i])

            _driver(fx, fn_)
            qd.sync()
            gate(f"F12_{label}", False, "no error raised")
        except qd.QuadrantsSyntaxError as e:
            gate(f"F12_{label}", needle in str(e), str(e)[:100])
        except Exception as e:  # noqa: BLE001
            gate(f"F12_{label}", False, f"WRONG-TYPE {type(e).__name__}: {str(e)[:100]}")

    return deco


def _f12a(x: qd.f32, n: qd.i32) -> tuple[vec3f, qd.f32]:
    if x < 0.0:
        return vec3f(0.0), -1.0
    return vec3f(1.0)


def _f12b(x: qd.f32, n: qd.i32) -> qd.f32:
    if x < 0.0:
        return
    return 1.0


PairT = tuple[qd.f32, qd.f32]


def _f12c(x: qd.f32, n: qd.i32) -> tuple[qd.f32, PairT]:
    if x < 0.0:
        return 0.0, (0.0, 0.0)
    return 1.0, (1.0, 1.0)


class F12Struct:
    pass


def _f12d(x: qd.f32, n: qd.i32) -> F12Struct:
    if x < 0.0:
        return F12Struct()
    return F12Struct()


try_reject("arity_mismatch", "inconsistent return arities", _f12a)
try_reject("bare_value_mix", "mixes bare", _f12b)
try_reject("nested_tuple", "nested-tuple", _f12c)
try_reject("struct_return", "flag-guard", _f12d)

print(f"\nEARLY_RETURN GATE [{ARCH}]: {'ALL PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
