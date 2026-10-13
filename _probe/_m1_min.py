import sys

import quadrants as qd

qd.init(arch=qd.cpu)
vec8 = qd.types.vector(8, qd.f32)
mat83 = qd.types.matrix(8, 3, qd.f32)
print("mat83 ndim", mat83.ndim, "n", mat83.n, "m", mat83.m)


@qd.func
def f_m1(x: qd.f32) -> mat83:
    m1 = mat83(0)
    return m1


@qd.func
def f_m1_mut(x: qd.f32) -> mat83:
    m1 = mat83(0)
    for i in range(8):
        m1[i, 0] = x + i
    return m1


@qd.kernel
def k(x: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=2)):
    for i in range(x.shape[0]):
        m = f_m1_mut(x[i])
        for r in range(8):
            y[i, r] = m[r, 0]


xa = qd.ndarray(dtype=qd.f32, shape=(1,))
xa.from_numpy(np_one) if False else None
import numpy as np

xa.from_numpy(np.array([3.0], dtype=np.float32))
ya = qd.ndarray(dtype=qd.f32, shape=(1, 8))
ya.from_numpy(np.zeros((1, 8), dtype=np.float32))
try:
    k(xa, ya)
    qd.sync()
    print("M1+mut OK", ya.to_numpy())
except Exception as e:
    import traceback

    traceback.print_exc()
