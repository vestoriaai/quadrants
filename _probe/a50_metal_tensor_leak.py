"""A50 metal follow-up: which Tensor(3)-into-scalar-slot shape leaks in SPIR-V?

The _ray kernel (with func-inlined ray_geom) fails on metal at
spirv_ir_builder.cpp:get_primitive_type@367 'Type [Tensor (3) f32] not
supported'. Bisect the suspect forms standalone (each compiles+runs in
seconds, unlike the full ray kernel):

  P1 unpack func tuple, store WHOLE vec3 to ndarray(vec3)
  P2 unpack func tuple, store ELEMENTWISE
  P3 unpack + vec4 carrier rebuild (ray_geom call-site shape)
  P6 vec3 local -> whole-tensor ndarray store, no func (baseline)
  P7 bare vec6 kernel arg + element reads (geomgroup shape)
  P8 vec3 local carried across if/else then whole-store

Usage: python a50_metal_tensor_leak.py [metal|cpu]
"""
import sys

import quadrants as qd
from typing import Tuple

ARCH = sys.argv[1] if len(sys.argv) > 1 else "metal"
qd.init(arch=getattr(qd, ARCH), offline_cache=False)

fails = []


def run_case(name, fn):
    try:
        fn()
        qd.sync()
        print(f"  [{ARCH}] {name:52s} OK")
    except Exception as e:
        msg = str(e).splitlines()[0][:110]
        print(f"  [{ARCH}] {name:52s} FAIL {msg}")
        fails.append(name)


def case_p1():
    @qd.func
    def leaf() -> Tuple[float, qd.math.vec3]:
        return 1.5, qd.math.vec3(1.0, 2.0, 3.0)

    @qd.kernel
    def k(out: qd.types.ndarray(dtype=qd.math.vec3, ndim=1)):
        d, n = leaf()
        out[0] = n * d

    out = qd.ndarray(dtype=qd.math.vec3, shape=[2])
    k(out)


def case_p2():
    @qd.func
    def leaf() -> Tuple[float, qd.math.vec3]:
        return 1.5, qd.math.vec3(1.0, 2.0, 3.0)

    @qd.kernel
    def k(out: qd.types.ndarray(dtype=qd.math.vec3, ndim=1)):
        d, n = leaf()
        for i in range(3):
            out[0][i] = n[i] * d

    out = qd.ndarray(dtype=qd.math.vec3, shape=[2])
    k(out)


def case_p3():
    @qd.func
    def leaf() -> Tuple[float, qd.math.vec3]:
        return 1.5, qd.math.vec3(1.0, 2.0, 3.0)

    @qd.kernel
    def k(out: qd.types.ndarray(dtype=qd.math.vec3, ndim=1)):
        d, n = leaf()
        carrier = qd.math.vec4(d, n[0], n[1], n[2])
        out[0] = qd.math.vec3(carrier[1], carrier[2], carrier[3])

    out = qd.ndarray(dtype=qd.math.vec3, shape=[2])
    k(out)


def case_p6():
    @qd.kernel
    def k(out: qd.types.ndarray(dtype=qd.math.vec3, ndim=1)):
        n = qd.math.vec3(1.0, 2.0, 3.0)
        out[0] = n

    out = qd.ndarray(dtype=qd.math.vec3, shape=[2])
    k(out)


def case_p7():
    @qd.kernel
    def k(g: vec6, out: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        s = 0.0
        for i in range(6):
            if g[i] > 0.0:
                s += g[i]
        out[0] = s

    out = qd.ndarray(dtype=qd.f32, shape=[2])
    k(vec6(-1, -1, 1.0, 1.0, -1, -1), out)


def case_p8():
    @qd.kernel
    def k(flag: qd.types.ndarray(dtype=qd.f32, ndim=1), out: qd.types.ndarray(dtype=qd.math.vec3, ndim=1)):
        n = qd.math.vec3(0.0)
        if flag[0] > 0.0:
            n = qd.math.vec3(1.0, 0.0, 0.0)
        else:
            n = qd.math.vec3(0.0, 1.0, 0.0)
        out[0] = n

    out = qd.ndarray(dtype=qd.math.vec3, shape=[2])
    fl = qd.ndarray(dtype=qd.f32, shape=[1])
    fl.from_numpy(__import__("numpy").ones(1, dtype="float32"))
    k(fl, out)


vec6 = qd.types.vector(6, qd.f32)  # ray.py's geomgroup type

if __name__ == "__main__":
    run_case("P1 unpack tuple -> whole vec3 store", case_p1)
    run_case("P2 unpack tuple -> elementwise store", case_p2)
    run_case("P3 unpack -> vec4 carrier rebuild", case_p3)
    run_case("P6 vec3 local -> whole store (no func)", case_p6)
    run_case("P7 bare vec6 kernel arg", case_p7)
    run_case("P8 vec3 across if/else -> whole store", case_p8)
    print(f"A50 METAL TENSOR LEAK [{ARCH}]: {6 - len(fails)}/6 PASS")
