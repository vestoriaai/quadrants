"""Offline structure check for the rewritten early_return_transformer (no compile/run).

Loads the NEW transformer module directly by path (its only quadrants import is the
exception class, supplied by the installed wheel) and asserts the emitted AST shapes.
"""
import ast
import importlib.util
import os
import sys
import textwrap

_DEFAULT_ERT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "python",
    "quadrants",
    "lang",
    "ast",
    "ast_transformers",
    "early_return_transformer.py",
)

spec = importlib.util.spec_from_file_location(
    "ert_new",
    sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_ERT,
)
ert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ert)

vec3 = None  # annotation elements are evaluated via global_vars below

import quadrants as _qd  # noqa: E402

GLOBALS = {"qd": _qd}


class FakeVec:
    """stand-in for a VectorType instance: isinstance-checkable only via real MatrixType."""

    pass


def run(src, globals_=None, is_kernel=False):
    tree = ast.parse(textwrap.dedent(src))
    fn = tree.body[0]
    out = ert.rewrite(fn, is_kernel=is_kernel, global_vars=globals_ or GLOBALS)
    return ast.unparse(out)


fails = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL':4}] {name} {detail}")
    if not cond:
        fails.append(name)


def expect_reject(name, src, needle, globals_=None, is_kernel=False):
    try:
        run(src, globals_, is_kernel)
    except ert.QuadrantsSyntaxError as e:
        check(name, needle in str(e), f"msg: {str(e)[:110]}")
        return
    except Exception as e:  # noqa: BLE001
        check(name, False, f"WRONG-TYPE {type(e).__name__}: {str(e)[:110]}")
        return
    check(name, False, "no error raised")


# 1. scalar guard: unchanged classic shape, single trailing guard merged
out = run(
    """
    def f(a: qd.f32) -> qd.f32:
        if a < 0:
            return -1.0
        b = a * 0.5
        return b + 1.0
    """
)
# b must be bound and read in the SAME guard block (8-space indent inside the fn-level guard)
check(
    "merge_sibling_binding_same_block",
    "if __er_done == 0:\n        b = a * 0.5\n        __er_ret0 = b + 1.0" in out,
    out,
)

# 2. three guards collapse: two post-return statements + final return -> ONE guard
n_guards = out.count("if __er_done == 0:")
check("single_merged_guard", n_guards == 1, f"count={n_guards}")

# 2b. sibling return-sites keep per-site guards (run split) -- must equal the OLD
#     pass output for the f_guard shape exactly
out = run(
    """
    def f(a: qd.f32) -> qd.f32:
        if a < 0.0:
            return -1.0
        if a < 1.0:
            return 0.5
        return 1.0
    """
)
check(
    "sibling_return_sites_own_guards",
    "if __er_done == 0:\n        if a < 1.0:\n            __er_ret0 = 0.5\n            __er_done = 1\n    if __er_done == 0:\n        __er_ret0 = 1.0\n        __er_done = 1" in out,
    out,
)

# 3. dead code after literal return is dropped (both pre- and post-seen_return)
out = run(
    """
    def f(a: qd.f32) -> qd.f32:
        if a < 0:
            return -1.0
        return 1.0
        dead = a * 100.0
    """
)
check("dead_code_dropped", "dead" not in out, out)

out = run(
    """
    def f(a: qd.f32) -> qd.f32:
        return 3.0
        dead2 = a
    """
)
check("dead_code_dropped_top", "dead2" not in out, out)

# 4. void function: no slots, guard merge on side effects
out = run(
    """
    def f(a: qd.f32, acc: list) -> None:
        if a < 0:
            return
        acc[0] = a
    """
)
check("void_no_slots", "__er_ret0" not in out and "if __er_done == 0:" in out, out)

# 5. loop: return-in-loop emits break (func), tail merged at loop level
out = run(
    """
    def f(a: qd.f32) -> qd.f32:
        for i in range(4):
            if a > 2.0:
                return a
            a = a + 1.0
        return a
    """
)
check("loop_break_emitted", "break" in out, out)
check("loop_tail_merged", out.count("if __er_done == 0:") == 2, out)  # loop-body guard + fn-level guard

# 6. static subtree untouched: the ORIGINAL return statement must survive verbatim
out = run(
    """
    def f(a: qd.f32) -> qd.f32:
        if qd.static(True):
            return a * 2.0
        return a
    """
)
check(
    "static_untouched",
    "if qd.static(True):\n        return a * 2.0" in out,
    out,
)

# 7. rejections
expect_reject(
    "reject_mixed_bare_value",
    """
    def f(a: qd.f32) -> qd.f32:
        if a < 0:
            return
        return 1.0
    """,
    "mixes bare",
)
expect_reject(
    "reject_arity_mismatch",
    """
    def f(a: qd.f32) -> tuple[qd.f32, qd.f32]:
        if a < 0:
            return 0.0, -1.0
        return 1.0
    """,
    "inconsistent return arities",
)
expect_reject(
    "reject_ann_slot_mismatch",
    """
    def f(a: qd.f32) -> tuple[qd.f32, qd.f32]:
        return 1.0
    """,
    "describes 2 slot(s)",
)


class NotAType:
    pass


expect_reject(
    "reject_struct_return",
    """
    def f(a: qd.f32) -> MyStruct:
        return a
    """,
    "flag-guard",
    globals_={"qd": GLOBALS["qd"], "MyStruct": NotAType},
)
expect_reject(
    "reject_unresolvable_ann",
    """
    def f(a: qd.f32) -> NoSuchName:
        return a
    """,
    "cannot resolve",
)

# 8. compound slot: needs a REAL MatrixType instance for classification
from quadrants.lang.matrix import MatrixType  # noqa: E402  (installed wheel)

vec3 = __import__("quadrants").types.vector(3, __import__("quadrants").f32)
mat83 = __import__("quadrants").types.matrix(8, 3, __import__("quadrants").f32)
G2 = {"qd": __import__("quadrants"), "vec3": vec3, "mat83": mat83}

out = run(
    """
    def f(x: qd.f32) -> tuple[vec3, qd.f32]:
        if x < 0.0:
            return vec3(0), -1.0
        v = vec3(0)
        v[1] = x
        return v, x * 2.0
    """,
    globals_=G2,
)
check(
    "compound_slot_init_form",
    "__er_ret0 = qd.cast(vec3(0), vec3.dtype)" in out,
    out.split("\n")[0],
)
check(
    "compound_first_binding_same_guard",
    "if __er_done == 0:\n        v = vec3(0)\n        v[1] = x\n        __er_ret0 = v\n        __er_ret1 = x * 2.0" in out,
    out,
)

out = run(
    """
    def m(x: qd.f32) -> mat83:
        if x < 0.0:
            return mat83(0)
        return mat83(1)
    """,
    globals_=G2,
)
check("mat_slot_init_form", "__er_ret0 = qd.cast(mat83(0), mat83.dtype)" in out, out.split("\n")[0])

# 9. nested tuple via typing origin
from typing import Tuple as _T  # noqa: E402

expect_reject(
    "reject_nested_tuple",
    """
    def f(a: qd.f32) -> tuple[qd.f32, PairT]:
        return a, a
    """,
    "nested-tuple",
    globals_={"qd": GLOBALS["qd"], "PairT": _T[float, float]},
)

# 10. kernel flag: break suppressed in outermost loop only
out = run(
    """
    def k(xs: list, ys: list):
        for i in range(4):
            if xs[i] < 0.0:
                return
            ys[i] = 1.0
    """,
    is_kernel=True,
)
check("kernel_outermost_no_break", "break" not in out, out)

out = run(
    """
    def k(xs: list, ys: list):
        for i in range(4):
            for j in range(4):
                if xs[i] < 0.0:
                    return
                ys[i] = 1.0
    """,
    is_kernel=True,
)
check("kernel_inner_break_kept", "break" in out, out)

print(f"\nSTRUCTURE CHECK: {'ALL PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
