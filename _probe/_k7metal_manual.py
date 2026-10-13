import sys
import numpy as np
import quadrants as qd

ARCH = sys.argv[1] if len(sys.argv) > 1 else "metal"
qd.init(arch=qd.metal if ARCH == "metal" else qd.cpu)
vec3f = qd.types.vector(3, qd.f32)

FX = [-7.5, -0.5, 0.0, 0.5, 3.0, 20.0, 100.0]
fx = qd.ndarray(dtype=qd.f32, shape=(len(FX),))
fx.from_numpy(np.array(FX, dtype=np.float32))

vec3 = vec3f


@qd.func
def f1(x: qd.f32) -> tuple[vec3f, qd.f32]:
    r0 = vec3f(0.0)
    r1 = 0.0
    done = qd.cast(0, qd.i32)
    if x < 0.0:
        r0 = vec3f(1.0, 0.0, 0.0)
        r1 = -1.0
        done = 1
    if done == 0:
        v = vec3f(0.0)
        v[1] = x
        r0 = v
        r1 = x * 2.0
    return r0, r1


# hand-written equivalent of F7's rewritten kernel (no early_return pass involved)
@qd.kernel
def k7_manual(xs: qd.types.ndarray(ndim=1), y3: qd.types.ndarray(ndim=1), y1: qd.types.ndarray(ndim=1), yg: qd.types.ndarray(ndim=1)):
    __er_done = qd.cast(0, qd.i32)
    for i in range(xs.shape[0]):
        v, s = f1(xs[i])
        for j in range(3):
            y3[i * 3 + j] = v[j]
        y1[i] = s
        if xs[i] > 50.0:
            yg[i] = 1.0
            __er_done = 1
        if __er_done == 0:
            yg[i] = 0.5
            __er_done = 1
    # loop runs out (no break on outermost)


o3 = qd.ndarray(dtype=qd.f32, shape=(len(FX) * 3,))
o1 = qd.ndarray(dtype=qd.f32, shape=(len(FX),))
og = qd.ndarray(dtype=qd.f32, shape=(len(FX),))
o3.from_numpy(np.zeros(len(FX) * 3, dtype=np.float32))
o1.from_numpy(np.zeros(len(FX), dtype=np.float32))
og.from_numpy(np.zeros(len(FX), dtype=np.float32))
k7_manual(fx, o3, o1, og)
qd.sync()
got = og.to_numpy()
want = [1.0 if x > 50.0 else 0.5 for x in FX]
print(f"k7_manual [{ARCH}] [{'PASS' if np.allclose(got, want) else 'FAIL'}] got={got} want={want}")
