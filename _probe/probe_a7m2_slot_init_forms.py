"""A7/M2 form-pinning probe: which compound slot-init forms are legal + stable?

Pins (cpu+metal), all independent of the early_return transformer — these are the
candidate prologue forms the transformer fix will synthesize:

  V1  v = vec3(0)                      vector single-scalar broadcast ctor
  V2  v = qd.cast(vec3(0), qd.f32)     ctor + element-dtype cast (call-shaped init,
                                       uniform with the existing scalar-slot discipline)
  M1  m = mat83(0)                     matrix single-scalar broadcast ctor
  M2  m = qd.cast(mat83(0), qd.f32)
  M3  m = mat83(vec8(0), x8)           row-wise fallback (scalar broadcast fails)
  R1  func -> tuple[vec3, f32] with tensor slot + struct return   (foundation path)
  H1  per-call reset under serialized loop for cast-wrapped compound init
      (over-hoisting guard: the :222-226 argument must extend to T(0)-shaped inits)

Run: <venv>/bin/python probe_a7m2_slot_init_forms.py [cpu|metal]   Exit 0 = all PASS.
"""
import sys

import numpy as np

import quadrants as qd

ARCH = sys.argv[1] if len(sys.argv) > 1 else "cpu"
qd.init(arch=qd.metal if ARCH == "metal" else qd.cpu)

fails = []


def gate(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL':4}] {name} {detail}")
    if not ok:
        fails.append(name)


vec3 = qd.types.vector(3, qd.f32)
vec8 = qd.types.vector(8, qd.f32)
mat83 = qd.types.matrix(8, 3, qd.f32)


# ---- V1/V2: vector init forms ------------------------------------------------
@qd.func
def f_vec_forms(x: qd.f32) -> tuple[vec3, vec3]:
    a = vec3(0)
    b = qd.cast(vec3(0), qd.f32)
    a[0] = x
    b[1] = x * 2.0
    return a, b


@qd.kernel
def k_vec(xs: qd.types.ndarray(ndim=1), y0: qd.types.ndarray(ndim=1), y1: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        a, b = f_vec_forms(xs[i])
        for j in range(3):
            y0[i * 3 + j] = a[j]
            y1[i * 3 + j] = b[j]


xs = qd.ndarray(dtype=qd.f32, shape=(2,))
xs.from_numpy(np.array([3.0, -2.0], dtype=np.float32))
y0 = qd.ndarray(dtype=qd.f32, shape=(6,))
y1 = qd.ndarray(dtype=qd.f32, shape=(6,))
y0.from_numpy(np.zeros(6, dtype=np.float32))
y1.from_numpy(np.zeros(6, dtype=np.float32))
try:
    k_vec(xs, y0, y1)
    qd.sync()
    g0 = y0.to_numpy()
    g1 = y1.to_numpy()
    want0 = np.array([3.0, 0.0, 0.0, -2.0, 0.0, 0.0], dtype=np.float32)
    want1 = np.array([0.0, 6.0, 0.0, 0.0, -4.0, 0.0], dtype=np.float32)
    gate("V1_vec3_broadcast_ctor", np.allclose(g0, want0), f"{g0}")
    gate("V2_vec3_cast_wrapped", np.allclose(g1, want1), f"{g1}")
except Exception as e:  # noqa: BLE001
    gate("V1_vec3_broadcast_ctor", False, f"{type(e).__name__}: {str(e)[:150]}")
    gate("V2_vec3_cast_wrapped", False, "same exception")


# ---- M1/M2/M3: matrix init forms (separate funcs: one poison must not kill all) --
@qd.func
def f_m1(x: qd.f32) -> tuple[mat83, mat83]:
    m1 = mat83(0)
    m2 = qd.cast(mat83(0), qd.f32)
    for i in range(8):
        m1[i, 0] = x + i
        m2[i, 1] = x * 2.0 + i
    return m1, m2


@qd.func
def f_m3(x: qd.f32) -> mat83:
    # row-wise via list flattening (vec-expr args do NOT flatten: 8 != 24 entries
    # trips MatrixExpression::type_check QD_ASSERT — pinned as loud-reject evidence)
    m3 = mat83([0.0] * 8, [0.0] * 8, [0.0] * 8)
    for i in range(8):
        m3[i, 2] = x * 3.0 + i
    return m3


@qd.kernel
def k_mat(xs: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=2)):
    for i in range(xs.shape[0]):
        m1, m2 = f_m1(xs[i])
        m3 = f_m3(xs[i])
        for r in range(8):
            y[i, r * 3 + 0] = m1[r, 0]
            y[i, r * 3 + 1] = m2[r, 1]
            y[i, r * 3 + 2] = m3[r, 2]


y = qd.ndarray(dtype=qd.f32, shape=(2, 24))
y.from_numpy(np.zeros((2, 24), dtype=np.float32))
try:
    k_mat(xs, y)
    qd.sync()
    g = y.to_numpy()
    # y[i, r*3+c]: c=0 -> m1[r,0]=x+i, c=1 -> m2[r,1]=x*2+i, c=2 -> m3[r,2]=x*3+i
    want = np.zeros((2, 24), dtype=np.float32)
    for row, xv in enumerate([3.0, -2.0]):
        for r in range(8):
            want[row, r * 3 + 0] = xv + r
            want[row, r * 3 + 1] = xv * 2.0 + r
            want[row, r * 3 + 2] = xv * 3.0 + r
    gate("M1_mat_broadcast_ctor", np.allclose(g[0][0:24:3], want[0][0:24:3]), f"{g[0][0:24:3]}")
    gate("M2_mat_cast_wrapped", np.allclose(g[0][1:24:3], want[0][1:24:3]), f"{g[0][1:24:3]}")
    gate("M3_mat_rowwise_ctor", np.allclose(g[0][2:24:3], want[0][2:24:3]), f"{g[0][2:24:3]}")
    gate("M_num_mat83_values", np.allclose(g[1], want[1]), f"{g[1][:6]}")
except Exception as e:  # noqa: BLE001
    msg = f"{type(e).__name__}: {str(e)[:150]}"
    gate("M1_mat_broadcast_ctor", False, msg)
    gate("M2_mat_cast_wrapped", False, msg)
    gate("M3_mat_rowwise_ctor", False, msg)
    gate("M_num_mat83_values", False, "n/a")


# ---- H1: per-call reset with compound slots (over-hoisting guard) -------------
@qd.func
def f_compound_reset(x: qd.f32) -> vec3:
    v = qd.cast(vec3(0), qd.f32)
    v[0] = x
    return v


@qd.kernel
def k_reset(xs: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=1)):
    qd.loop_config(serialize=True)
    for i in range(4):
        v = f_compound_reset(xs[i])
        y[i * 3] = v[0]
        y[i * 3 + 1] = v[1]
        y[i * 3 + 2] = v[2]


xs4 = qd.ndarray(dtype=qd.f32, shape=(4,))
xs4.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
y4 = qd.ndarray(dtype=qd.f32, shape=(12,))
y4.from_numpy(np.zeros(12, dtype=np.float32))
try:
    k_reset(xs4, y4)
    qd.sync()
    got = y4.to_numpy().reshape(4, 3)
    want = np.array([[v, 0.0, 0.0] for v in [1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    gate("H1_compound_per_call_reset", np.allclose(got, want), f"{got[:, 0]}")
except Exception as e:  # noqa: BLE001
    gate("H1_compound_per_call_reset", False, f"{type(e).__name__}: {str(e)[:150]}")


# ---- R1: bare-T(0) prologue under serialized loop (hoisting hazard shape) ------
@qd.func
def f_compound_bare(x: qd.f32) -> vec3:
    v = vec3(0)
    v[0] = x
    return v


@qd.kernel
def k_reset_bare(xs: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=1)):
    qd.loop_config(serialize=True)
    for i in range(4):
        v = f_compound_bare(xs[i])
        y[i * 3] = v[0]
        y[i * 3 + 1] = v[1]
        y[i * 3 + 2] = v[2]


y4b = qd.ndarray(dtype=qd.f32, shape=(12,))
y4b.from_numpy(np.zeros(12, dtype=np.float32))
try:
    k_reset_bare(xs4, y4b)
    qd.sync()
    got = y4b.to_numpy().reshape(4, 3)
    want = np.array([[v, 0.0, 0.0] for v in [1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    gate("R1_bare_ctor_per_call_reset", np.allclose(got, want), f"{got[:, 0]}")
except Exception as e:  # noqa: BLE001
    gate("R1_bare_ctor_per_call_reset", False, f"{type(e).__name__}: {str(e)[:150]}")


print(f"\nSLOT_INIT_FORMS arch={ARCH}: {'ALL PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
