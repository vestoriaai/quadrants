"""A50-(2) trigger matrix: @qd.func(early_return=True) compound return slots.

Isolates the make_matrix_expr@1445 failure (dt->is<PrimitiveType>()) seen when
converting ray_geom real_func->func for metal. Minimal bodies: constructors
only, no scalar*tensor binops, so the promoted_type (A50-(1)) layer is NOT
exercised - whatever fires here is the layer-2 mechanism itself. The failure
is a compile-time frontend assert, so CPU repro is sufficient.

Cases:
  C1 early-return tuple(float, vec3), unpacked at kernel call site  (ray_geom form)
  C2 done-flag single-exit tuple(float, vec3)  (= probe_realfunc_composite, expect OK)
  C3 early-return vec3 only (single compound slot)
  C4 early-return float only (single primitive slot)
  C5 early-return tuple(float, float)          (sanity)
  C6 C1 body consumed through a second plain-func dispatcher

Usage: python a50_er_tuple_matrix.py   (prints one OK/FAIL line per case)
"""
import traceback
from typing import Tuple

import quadrants as qd

qd.init(arch=qd.cpu, offline_cache=False)

out0 = qd.field(dtype=qd.f32, shape=())
out1 = qd.field(dtype=qd.f32, shape=())


def run_case(name, fn):
    try:
        fn()
        print(f"  {name:58s} OK")
        return True
    except Exception:
        print(f"  {name:58s} FAIL")
        traceback.print_exc()
        return False


def case_c1():
    @qd.func(early_return=True)
    def f(x: qd.math.vec3) -> Tuple[float, qd.math.vec3]:
        if x[0] > 0.0:
            return 1.0, qd.math.vec3(1.0, 0.0, 0.0)
        return 0.0, qd.math.vec3(0.0, 0.0, 1.0)

    @qd.kernel
    def k():
        d, n = f(qd.math.vec3(1.0, 0.0, 0.0))
        out0[None] = d + n[0]

    k()
    assert abs(out0[None] - 2.0) < 1e-6, out0[None]


def case_c2():
    @qd.func
    def f(x: qd.math.vec3) -> Tuple[float, qd.math.vec3]:
        d = 0.0
        n = qd.math.vec3(0.0, 0.0, 1.0)
        if x[0] > 0.0:
            d = 1.0
            n = qd.math.vec3(1.0, 0.0, 0.0)
        return d, n

    @qd.kernel
    def k():
        d, n = f(qd.math.vec3(1.0, 0.0, 0.0))
        out0[None] = d + n[0]

    k()
    assert abs(out0[None] - 2.0) < 1e-6, out0[None]


def case_c3():
    @qd.func(early_return=True)
    def f(x: qd.math.vec3) -> qd.math.vec3:
        if x[0] > 0.0:
            return qd.math.vec3(1.0, 0.0, 0.0)
        return qd.math.vec3(0.0, 0.0, 1.0)

    @qd.kernel
    def k():
        n = f(qd.math.vec3(1.0, 0.0, 0.0))
        out0[None] = n[0]

    k()
    assert abs(out0[None] - 1.0) < 1e-6, out0[None]


def case_c4():
    @qd.func(early_return=True)
    def f(x: qd.math.vec3) -> float:
        if x[0] > 0.0:
            return 1.0
        return 0.0

    @qd.kernel
    def k():
        out0[None] = f(qd.math.vec3(1.0, 0.0, 0.0))

    k()
    assert abs(out0[None] - 1.0) < 1e-6, out0[None]


def case_c5():
    @qd.func(early_return=True)
    def f(x: qd.math.vec3) -> Tuple[float, float]:
        if x[0] > 0.0:
            return 1.0, 2.0
        return 3.0, 4.0

    @qd.kernel
    def k():
        a, b = f(qd.math.vec3(1.0, 0.0, 0.0))
        out0[None] = a + b

    k()
    assert abs(out0[None] - 3.0) < 1e-6, out0[None]


def case_c6():
    @qd.func(early_return=True)
    def leaf(x: qd.math.vec3) -> Tuple[float, qd.math.vec3]:
        if x[0] > 0.0:
            return 1.0, qd.math.vec3(1.0, 0.0, 0.0)
        return 0.0, qd.math.vec3(0.0, 0.0, 1.0)

    @qd.func
    def disp(x: qd.math.vec3) -> Tuple[float, qd.math.vec3]:
        return leaf(x)

    @qd.kernel
    def k():
        d, n = disp(qd.math.vec3(1.0, 0.0, 0.0))
        out1[None] = d + n[1]

    k()
    assert abs(out1[None] - 1.0) < 1e-6, out1[None]


if __name__ == "__main__":
    results = [
        run_case("C1 early-ret tuple(f32,vec3) unpacked", case_c1),
        run_case("C2 done-flag single-exit tuple", case_c2),
        run_case("C3 early-ret vec3 only", case_c3),
        run_case("C4 early-ret f32 only", case_c4),
        run_case("C5 early-ret tuple(f32,f32)", case_c5),
        run_case("C6 C1 via plain-func dispatcher", case_c6),
    ]
    print(f"A50 ER-TUPLE-MATRIX [cpu]: {sum(results)}/{len(results)} PASS")
