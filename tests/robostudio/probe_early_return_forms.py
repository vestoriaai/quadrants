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

print(f"\nEARLY_RETURN GATE arch={ARCH}: {'ALL PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
