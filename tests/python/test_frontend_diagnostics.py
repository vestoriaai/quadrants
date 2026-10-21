from typing import Tuple

import pytest

import quadrants as qd

from tests import test_utils

vec3 = qd.types.vector(3, qd.f32)

# Frontend diagnostics (ledger A50/A51): two failure shapes used to die two C++
# asserts deep in the frontend / SPIR-V codegen, or worse, silently compute
# wrong values (scalar <- Tensor truncated to its first component on LLVM).
# Both are now clear Python-side errors; the legal shapes must keep working.


@test_utils.test()
def test_assign_scalar_var_tensor_value_rejected():
    # A51: `l = 0.0` then `l = <vec3>` in a branch - LLVM truncated, SPIR-V
    # refused; now a clear type error at the assign site.
    flag = qd.field(qd.i32, shape=())

    @qd.kernel
    def a50a51_broken():
        l = 0.0
        if flag[None] > 0:
            l = qd.math.vec3(1.0, 2.0, 3.0)

    flag[None] = 1
    with pytest.raises(
        qd.QuadrantsTypeError, match=r"was created as a scalar but is assigned a Tensor"
    ):
        a50a51_broken()


@test_utils.test()
def test_assign_tensor_var_scalar_broadcast_kept():
    # The reverse direction is a legal broadcast and must keep working.
    flag = qd.field(qd.i32, shape=())
    out = qd.field(dtype=vec3, shape=())

    @qd.kernel
    def a50a51_ok():
        n = qd.math.vec3(0.0)
        if flag[None] > 0:
            n = 1.5
        out[None] = n

    flag[None] = 1
    a50a51_ok()
    assert out.to_numpy() == pytest.approx([1.5, 1.5, 1.5])


@test_utils.test()
def test_assign_tensor_shape_mismatch_rejected():
    # vec3 variable assigned a vec4 used to assert deep inside scalarize.
    flag = qd.field(qd.i32, shape=())

    @qd.kernel
    def a50a51_broken():
        n = qd.math.vec3(0.0)
        if flag[None] > 0:
            n = qd.math.vec4(1.0, 2.0, 3.0, 4.0)

    flag[None] = 1
    with pytest.raises(
        qd.QuadrantsTypeError,
        match=r"has shape \(3,\) but is assigned a Tensor with shape \(4,\)",
    ):
        a50a51_broken()


@test_utils.test()
def test_whole_tuple_func_assign_rejected():
    # A50, the exact ray.py shape: `x` is created in one branch and re-assigned
    # a multi-value func return in the other. The re-assign routes the tuple
    # through ops.assign -> Expr(tuple) -> make_matrix; say "unpack it" instead
    # of dying in promoted_type / make_matrix_expr asserts. (A FRESH
    # `x = f(...)` silently binds a python list - separate quirk, out of scope.)
    flag = qd.field(qd.i32, shape=())

    @qd.func
    def two_vals() -> Tuple[float, vec3]:
        return 1.0, qd.math.vec3(1.0, 2.0, 3.0)

    @qd.kernel
    def a50a51_broken():
        x = 0.0
        if flag[None] > 0:
            x = two_vals()

    flag[None] = 1
    with pytest.raises(qd.QuadrantsTypeError, match=r"unpack it instead"):
        a50a51_broken()


@test_utils.test()
def test_nested_matrix_concatenation_still_works():
    # A50 anti-footgun: building a vector from a scalar + a smaller vector is
    # legal concatenation (entries flatten) - the diagnostics must not reject it.
    out = qd.field(dtype=qd.types.vector(4, qd.f32), shape=())

    @qd.kernel
    def a50a51_ok():
        v = qd.math.vec3(1.0, 2.0, 3.0)
        out[None] = qd.math.vec4(0.5, v)

    a50a51_ok()
    assert out.to_numpy() == pytest.approx([0.5, 1.0, 2.0, 3.0])


@test_utils.test()
def test_func_multi_value_unpack_still_works():
    # The legal form the diagnostics point to: unpacking a func's
    # (float, vec3) return, incl. early_return - must be unaffected.
    @qd.func(early_return=True)
    def leaf() -> Tuple[float, vec3]:
        return 1.5, qd.math.vec3(1.0, 2.0, 3.0)

    out_d = qd.field(qd.f32, shape=())
    out_n = qd.field(dtype=vec3, shape=())

    @qd.kernel
    def a50a51_ok():
        d, n = leaf()
        out_d[None] = d
        out_n[None] = n

    a50a51_ok()
    assert out_d[None] == pytest.approx(1.5)
    assert out_n.to_numpy() == pytest.approx([1.0, 2.0, 3.0])
