"""A50 follow-up: scalar-initialized local rebound to a Tensor inside a branch.

ray.py ray_ellipsoid has `l = 0.0` then `l = lpnt + lvec * sol` (vec3) inside
an `if`. CPU/LLVM tolerates the retype silently; SPIR-V dies with
'get_primitive_type: Type [Tensor (3) f32] not supported'. This probe isolates
that exact shape from the ray code. Usage: python <file> [metal|cpu]
"""
import sys

import quadrants as qd

ARCH = sys.argv[1] if len(sys.argv) > 1 else "metal"
qd.init(arch=getattr(qd, ARCH), offline_cache=False)


def run(name, fn):
    try:
        fn()
        qd.sync()
        print(f"  [{ARCH}] {name:50s} OK")
    except Exception as e:
        print(f"  [{ARCH}] {name:50s} FAIL {str(e).splitlines()[0][:100]}")


def scalar_init_then_vec():
    @qd.kernel
    def k(flag: qd.types.ndarray(dtype=qd.f32, ndim=1), out: qd.types.ndarray(dtype=qd.math.vec3, ndim=1)):
        s = qd.math.vec3(1.0, 2.0, 3.0)
        normal = qd.math.vec3(0.0)
        l = 0.0
        if flag[0] >= 0.0:
            l = qd.math.vec3(0.5, 0.5, 0.5) + s * flag[0]
            normal = s * l
        out[0] = normal

    fl = qd.ndarray(dtype=qd.f32, shape=[1])
    out = qd.ndarray(dtype=qd.math.vec3, shape=[1])
    k(fl, out)


def vec_init_then_vec():
    @qd.kernel
    def k(flag: qd.types.ndarray(dtype=qd.f32, ndim=1), out: qd.types.ndarray(dtype=qd.math.vec3, ndim=1)):
        s = qd.math.vec3(1.0, 2.0, 3.0)
        normal = qd.math.vec3(0.0)
        l = qd.math.vec3(0.0)
        if flag[0] >= 0.0:
            l = qd.math.vec3(0.5, 0.5, 0.5) + s * flag[0]
            normal = s * l
        out[0] = normal

    fl = qd.ndarray(dtype=qd.f32, shape=[1])
    out = qd.ndarray(dtype=qd.math.vec3, shape=[1])
    k(fl, out)


if __name__ == "__main__":
    run("scalar-init local rebound to vec3 in branch", scalar_init_then_vec)
    run("vec3-init local (control)", vec_init_then_vec)
