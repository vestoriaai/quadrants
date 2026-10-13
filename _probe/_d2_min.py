import numpy as np

import quadrants as qd

qd.init(arch=qd.cpu)


@qd.func(early_return=True)
def f_sibling(x: qd.f32) -> qd.f32:
    if x < 0.0:
        return -1.0
    t = x * 0.5
    return t + 1.0


@qd.kernel
def k(xs: qd.types.ndarray(ndim=1), ys: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        ys[i] = f_sibling(xs[i])


xs = qd.ndarray(dtype=qd.f32, shape=(3,))
ys = qd.ndarray(dtype=qd.f32, shape=(3,))
xs.from_numpy(np.array([-1.0, 0.5, 4.0], dtype=np.float32))
try:
    k(xs, ys)
    qd.sync()
    print("D2 OK", ys.to_numpy())
except Exception as e:
    print("ERRTYPE:", type(e).__name__)
    msg = str(e)
    print("TAIL:", msg[-700:])
