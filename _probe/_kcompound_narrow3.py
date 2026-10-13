import numpy as np
import quadrants as qd

qd.init(arch=qd.cpu)
vec3f = qd.types.vector(3, qd.f32)
FX = [-7.5, -0.5, 0.0, 0.5, 3.0, 20.0, 100.0]
fx = qd.ndarray(dtype=qd.f32, shape=(len(FX),))
fx.from_numpy(np.array(FX, dtype=np.float32))
y = qd.ndarray(dtype=qd.f32, shape=(len(FX),))


@qd.kernel
def kD(xs: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        done = qd.cast(0, qd.i32)
        v = vec3f(0.0)
        if xs[i] < 0.0:
            done = 1
        if done == 0:
            v = vec3f(-1.0, -2.0, -3.0)  # whole-value store, flag condition
        y[i] = v[0] + v[2]


y.from_numpy(np.zeros(len(FX), dtype=np.float32))
kD(fx, y)
qd.sync()
got = y.to_numpy()
want = [-4.0 if x < 0 else 0.0 for x in FX]
print(f"D flag-cond whole-store [{'PASS' if np.allclose(got, want) else 'FAIL'}] got={got}")


@qd.kernel
def kE(xs: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        done = qd.cast(0, qd.i32)
        v = vec3f(0.0)
        if xs[i] < 0.0:
            done = 1
        if done == 0:
            v[0] = xs[i]  # element store, flag condition, read outside
        y[i] = v[0] + v[2]


y.from_numpy(np.zeros(len(FX), dtype=np.float32))
kE(fx, y)
qd.sync()
got = y.to_numpy()
want = [0.0 if x < 0 else x for x in FX]
print(f"E flag-cond elem-store read-out [{'PASS' if np.allclose(got, want) else 'FAIL'}] got={got}")


@qd.kernel
def kF(xs: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        done = qd.cast(0, qd.i32)
        v = vec3f(0.0)
        if xs[i] < 0.0:
            done = 1
        if done == 0:
            v[0] = xs[i]
            y[i] = v[0] + v[2]  # element store + read inside, flag condition


y.from_numpy(np.zeros(len(FX), dtype=np.float32))
kF(fx, y)
qd.sync()
got = y.to_numpy()
want = [0.0 if x < 0 else x for x in FX]
print(f"F flag-cond elem-store read-in [{'PASS' if np.allclose(got, want) else 'FAIL'}] got={got}")
