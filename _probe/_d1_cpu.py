import sys

import numpy as np

import quadrants as qd

ARCH = sys.argv[1] if len(sys.argv) > 1 else "cpu"
qd.init(arch=qd.metal if ARCH == "metal" else qd.cpu)
vec3 = qd.types.vector(3, qd.f32)


@qd.func(early_return=True)
def f_compound_tup(x: qd.f32) -> tuple[vec3, qd.f32]:
    if x < 0.0:
        return vec3([1.0, 0.0, 0.0]), -1.0
    return vec3([0.0, x, 0.0]), x * 2.0


@qd.kernel
def k_d1(xs: qd.types.ndarray(ndim=1), ys: qd.types.ndarray(ndim=1), zs: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        v, s = f_compound_tup(xs[i])
        for j in range(3):
            ys[i * 3 + j] = v[j]
        zs[i] = s


xs = qd.ndarray(dtype=qd.f32, shape=(2,))
ys = qd.ndarray(dtype=qd.f32, shape=(6,))
zs = qd.ndarray(dtype=qd.f32, shape=(2,))
xs.from_numpy(np.array([-1.0, 2.0], dtype=np.float32))
try:
    k_d1(xs, ys, zs)
    qd.sync()
    print("D1 OK", ys.to_numpy(), zs.to_numpy())
except Exception as e:
    print("ERRTYPE:", type(e).__name__)
    print("...HEAD...", str(e)[:600])
    print("...TAIL...", str(e)[-900:])
