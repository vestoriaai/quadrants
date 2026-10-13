#!/usr/bin/env python3
"""A20 probe/gate: actionable NameError for names bound in closed block scopes.

Root cause behind the mujoco_qd io_test geom1 NameError family (A20): a name
FIRST bound inside a runtime `if` branch (tuple-unpack from a func return, the
warp-port style) is not visible in later sibling blocks -- the block scope
popped at the join, so get_var_by_name fell through to the builtins fallback
and surfaced as `AttributeError: module 'builtins' has no attribute 'geom1'`
chained into a bare `Name "geom1" is not defined`, which read like a typo and
sent the hunt down a wrong path (registered as A20, then re-attributed: the
shape never compiled on ANY wheel, v1.3.3 through dev95 -- not a regression).
The convention is predeclaration (assign a default before the block), exactly
what the mujoco_qd sensor kernels already do.

Pinned behaviour:

  D1  actionable diagnostic -- the branch-bind shape raises QuadrantsNameError
      whose text names the variable, states the closed-block cause and the
      predeclaration convention; the misleading chained builtins
      AttributeError is gone (`from None`).
  D2  generic typo unchanged -- a name that was never bound anywhere still
      raises the plain 'Name "x" is not defined' (test_name_error contract).
  R1  predecl recipe compiles -- the same kernel with the branch-bound names
      predeclared compiles and computes the correct value on cpu (this is the
      fix recipe handed to the mujoco_qd collision_primitive owner).

Usage: probe_a20_branch_scope_diag.py [arch]   (default cpu; text-level pin)
Verdict: RESULT: PASS / RESULT: FAIL (exit 1)
"""
import sys
import traceback

import numpy as np

import quadrants as qd

qd.init(arch=getattr(qd, sys.argv[1] if len(sys.argv) > 1 else "cpu"))

fails = []


def check(name, ok):
    print(f"  {name:44s} {'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(name)


@qd.func
def _pair(sel: qd.types.ndarray(dtype=qd.i32, ndim=1), tid: int) -> tuple:
    g1 = sel[tid]
    g2 = sel[tid] + 1
    return g1, g2


@qd.func
def _params(cutoff_in: qd.types.ndarray(dtype=qd.f32, ndim=1), tid: int) -> tuple:
    m = cutoff_in[tid]
    return m, m * 0.5


@qd.func
def _worker(g1: int, g2: int, m: float, out: qd.types.ndarray(dtype=qd.f32, ndim=1)) -> None:
    out[0] = out[0] + qd.cast(g1 * 10 + g2, qd.f32) + m


@qd.kernel
def k_branch_bind(
    sel: qd.types.ndarray(dtype=qd.i32, ndim=1),
    cutoff_in: qd.types.ndarray(dtype=qd.f32, ndim=1),
    out: qd.types.ndarray(dtype=qd.f32, ndim=1),
):
    for t in range(1):
        done = False
        tid = t
        if tid >= 1:
            done = True
        if not done:
            margin, gap = _params(cutoff_in, tid)
        if not done:
            geom1, geom2 = _pair(sel, tid)
        if not done:
            for i in qd.static(range(2)):
                if i == 1:
                    _worker(geom1, geom2, margin, out)


@qd.kernel
def k_predecl(
    sel: qd.types.ndarray(dtype=qd.i32, ndim=1),
    cutoff_in: qd.types.ndarray(dtype=qd.f32, ndim=1),
    out: qd.types.ndarray(dtype=qd.f32, ndim=1),
):
    for t in range(1):
        done = False
        tid = t
        geom1 = 0
        geom2 = 0
        margin = 0.0
        gap = 0.0
        if tid >= 1:
            done = True
        if not done:
            margin, gap = _params(cutoff_in, tid)
        if not done:
            geom1, geom2 = _pair(sel, tid)
        if not done:
            for i in qd.static(range(2)):
                if i == 1:
                    _worker(geom1, geom2, margin, out)


@qd.kernel
def k_typo():
    a + 1


sel = np.array([2, 0], dtype=np.int32)
cutoff_in = np.array([0.5], dtype=np.float32)
out = np.zeros(1, dtype=np.float32)

# D1 -- actionable diagnostic, no misleading chained AttributeError
try:
    k_branch_bind(sel, cutoff_in, out)
    check("D1 branch-bind kernel raises", False)
except qd.QuadrantsNameError as e:
    msg = str(e)
    check(
        "D1 actionable text (cause + predecl)",
        'Name "geom1" is not defined' in msg
        and "Predeclare it before the block" in msg
        and "geom1 = 0" in msg,
    )
    check(
        "D1 no chained builtins AttributeError",
        "no attribute 'geom1'" not in msg and "During handling" not in msg,
    )
except Exception:  # noqa: BLE001
    traceback.print_exc()
    check("D1 raises QuadrantsNameError", False)

# D2 -- true typo keeps the generic message
try:
    k_typo()
    check("D2 typo kernel raises", False)
except qd.QuadrantsNameError as e:
    msg = str(e)
    check(
        "D2 generic message unchanged",
        'Name "a" is not defined' in msg and "Predeclare it before the block" not in msg,
    )

# R1 -- predecl recipe compiles and computes correctly
# sel[0]=2 -> geom1=2, geom2=3; margin=0.5; one worker call -> 2*10+3+0.5 = 23.5
out = np.zeros(1, dtype=np.float32)
try:
    k_predecl(sel, cutoff_in, out)
    qd.sync()
    got = float(out[0])
    check("R1 predecl kernel compiles + value", abs(got - 23.5) < 1e-5)
except Exception:  # noqa: BLE001
    traceback.print_exc()
    check("R1 predecl kernel compiles", False)

print(f"RESULT: {'PASS' if not fails else 'FAIL'}")
sys.exit(1 if fails else 0)
