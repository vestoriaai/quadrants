import sys
import numpy as np
import quadrants as qd

qd.init(arch=qd.cpu)
vec3f = qd.types.vector(3, qd.f32)
FX = [-7.5, -0.5, 0.0, 0.5, 3.0, 20.0, 100.0]
fx = qd.ndarray(dtype=qd.f32, shape=(len(FX),))
fx.from_numpy(np.array(FX, dtype=np.float32))
y = qd.ndarray(dtype=qd.f32, shape=(len(FX),))

# A: store inside guard, READ outside
@qd.kernel
def kA(xs: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        v = vec3f(0.0)
        if xs[i] < 0.0:
            v[0] = -1.0
        y[i] = v[0] + v[2]

y.from_numpy(np.zeros(len(FX), dtype=np.float32))
kA(fx, y)
qd.sync()
got = y.to_numpy()
want = [-1.0 if x < 0 else 0.0 for x in FX]
print(f"A store-in-guard read-out [{'PASS' if np.allclose(got, want) else 'FAIL'}] got={got}")

# B: whole-value store inside guard, read inside
@qd.kernel
def kB(xs: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        v = vec3f(0.0)
        if xs[i] < 0.0:
            v = vec3f(-1.0, -2.0, -3.0)
        y[i] = v[0] + v[2]

y.from_numpy(np.zeros(len(FX), dtype=np.float32))
kB(fx, y)
qd.sync()
got = y.to_numpy()
want = [-4.0 if x < 0 else 0.0 for x in FX]
print(f"B whole-store-in-guard [{'PASS' if np.allclose(got, want) else 'FAIL'}] got={got}")

# C: element store + read both inside guard, SCALAR condition not involving done
@qd.kernel
def kC(xs: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        v = vec3f(0.0)
        if xs[i] < 0.0:
            v[0] = -1.0
            y[i] = v[0] + v[2]

y.from_numpy(np.zeros(len(FX), dtype=np.float32))
kC(fx, y)
qd.sync()
got = y.to_numpy()
want = [-1.0 if x < 0 else 0.0 for x in FX]
print(f"C elem-store+read in-guard [{'PASS' if np.allclose(got, want) else 'FAIL'}] got={got}")
