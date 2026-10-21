"""A51 semantics probe: what does local variable type rebinding ACTUALLY mean today?

Before deciding "frontend rejects scalar alloca <- Tensor assign", nail down the
current behavior empirically on cpu+metal:

  R1 tensor var <- scalar rebind  : n = vec3(0); if c: n = 1.5; read after merge
  R2 scalar var <- tensor rebind  : l = 0.0;   if c: l = vec3; read after merge
  R3 scalar var <- tensor, same-path consume (the only shape known to work)

If R1/R2 compile anywhere, the stored VALUES show whether the implicit retype
has coherent semantics or is an implementation accident.

Usage: python a50_a51_rebind_semantics.py [cpu|metal]
"""
import sys

import numpy as np

import quadrants as qd

ARCH = sys.argv[1] if len(sys.argv) > 1 else "cpu"
qd.init(arch=getattr(qd, ARCH), offline_cache=False)


def run(name, fn):
    try:
        fn()
        qd.sync()
        print(f"  [{ARCH}] {name:46s} OK")
        return True
    except Exception as e:
        print(f"  [{ARCH}] {name:46s} FAIL {str(e).splitlines()[0][:90]}")
        return False


def case_r1():
    @qd.kernel
    def k(flag: qd.types.ndarray(dtype=qd.f32, ndim=1), out: qd.types.ndarray(dtype=qd.math.vec3, ndim=1)):
        n = qd.math.vec3(0.0)
        if flag[0] > 0.0:
            n = 1.5
        out[0] = n

    fl = qd.ndarray(dtype=qd.f32, shape=[1])
    out = qd.ndarray(dtype=qd.math.vec3, shape=[1])
    fl.from_numpy(np.array([1.0], dtype="float32"))  # then path: the scalar assign executes
    k(fl, out)
    print(f"      R1 then-path value = {out.to_numpy()[0]}  (broadcast would be [1.5 1.5 1.5])")


def case_r2():
    @qd.kernel
    def k(flag: qd.types.ndarray(dtype=qd.f32, ndim=1), out: qd.types.ndarray(dtype=qd.math.vec3, ndim=1)):
        l = 0.0
        if flag[0] > 0.0:
            l = qd.math.vec3(1.0, 2.0, 3.0)
        out[0] = l

    fl = qd.ndarray(dtype=qd.f32, shape=[1])
    out = qd.ndarray(dtype=qd.math.vec3, shape=[1])
    fl.from_numpy(np.array([-1.0], dtype="float32"))  # else path
    k(fl, out)
    print(f"      R2 else-path value = {out.to_numpy()[0]}")
    fl.from_numpy(np.array([1.0], dtype="float32"))  # then path
    k(fl, out)
    print(f"      R2 then-path value = {out.to_numpy()[0]}")


def case_r3():
    @qd.kernel
    def k(flag: qd.types.ndarray(dtype=qd.f32, ndim=1), out: qd.types.ndarray(dtype=qd.math.vec3, ndim=1)):
        l = 0.0
        if flag[0] > 0.0:
            l = qd.math.vec3(1.0, 2.0, 3.0)
            out[0] = l  # consumed on the same path it was rebound

    fl = qd.ndarray(dtype=qd.f32, shape=[1])
    out = qd.ndarray(dtype=qd.math.vec3, shape=[1])
    fl.from_numpy(np.array([1.0], dtype="float32"))
    k(fl, out)
    print(f"      R3 then-path value = {out.to_numpy()[0]}")


def case_r4():
    # ray_ellipsoid's exact shape: rebind to a COMPUTED tensor expr, same-path consume
    @qd.kernel
    def k(flag: qd.types.ndarray(dtype=qd.f32, ndim=1), out: qd.types.ndarray(dtype=qd.math.vec3, ndim=1)):
        lv = qd.math.vec3(0.3, 0.5, 0.7)
        lpnt = qd.math.vec3(0.1, 0.2, 0.3)
        s = qd.math.vec3(1.0, 2.0, 3.0)
        l = 0.0
        if flag[0] > 0.0:
            l = lpnt + lv * flag[0]  # = (0.4, 0.7, 1.0) at flag=1
            out[0] = s * l           # want (0.4, 1.4, 3.0)

    fl = qd.ndarray(dtype=qd.f32, shape=[1])
    out = qd.ndarray(dtype=qd.math.vec3, shape=[1])
    fl.from_numpy(np.array([1.0], dtype="float32"))
    k(fl, out)
    print(f"      R4 then-path value = {out.to_numpy()[0]}  want [0.4 1.4 3.0]")


if __name__ == "__main__":
    run("R1 tensor var <- scalar, read after merge", case_r1)
    run("R2 scalar var <- tensor, read after merge", case_r2)
    run("R3 scalar var <- tensor, same-path consume", case_r3)
    run("R4 rebind to computed expr (ray_ellipsoid shape)", case_r4)
