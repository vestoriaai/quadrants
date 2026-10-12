#!/usr/bin/env python3
"""A10 probe: nested serial loops, cross-iteration RAW through global memory.

2026-10-12 root-cause review (mujocoqd/a10-nested-loop-raw): the original A10
evidence (mujoqo _probe/cs_evidence/cs_dbg_chol2.py) attributed column-order
NaNs to a quadrants miscompile ("nan on BOTH backends = our own codegen bug").
That form is algorithmically INVALID, not miscompiled: its i<j branch divides
by U[j,j], but the i==j branch stores U[j,j] LATER in the same middle loop, so
sequential semantics read the initial fill (0) -> ±inf -> NaN.  The numpy twin
of that exact program NaNs identically, and qd's output matches the twin
exactly (NaN pattern included) on cpu AND metal AND across wheels (dev36 and
current): both backends compiled the program faithfully.  A corrected
column-oriented form (diagonal of column j first -- it depends only on earlier
columns) passes on both backends in the porting shape (func + 3-D arrays +
serialize directives, n=8/24).

This probe pins, per arch:
  A  corrected col-order Cholesky (inline, 2-D)  == f32 twin, tol 1e-5
  B  corrected col-order Cholesky (func, 3-D)    == f32 twin, tol 1e-5
  C  row-order L-form Cholesky (control)         == f32 twin, tol 1e-5
  D  invalid-order form output == numpy twin EXACTLY
     (NaN pattern + finite values) -- guards the
     faithful-compilation contract on a program whose
     sequential semantics are degenerate; a future
     codegen change that alters this output (without a
     semantics change) fails here loudly.

Porting note: the row-order law for loop-carried algorithms stands, but its
correct justification is algorithm data-dependence structure (row forms see
the diagonal/earlier-row values they divide by), not a compiler defect.

Usage: probe_a10_nested_loop_order.py [arch]   (default cpu)
Verdict: RESULT: PASS / RESULT: FAIL
"""
import sys

import numpy as np
import quadrants as qd

ARCH = sys.argv[1] if len(sys.argv) > 1 else "cpu"
qd.init(arch=getattr(qd, ARCH))
N = 8


def make_H(n):
    rng = np.random.default_rng(3)
    B = rng.standard_normal((n, n)).astype(np.float32) * 0.3
    return (B @ B.T + n * np.eye(n, dtype=np.float32)).astype(np.float32)


# --- corrected column-oriented U Cholesky: diagonal of column j first -------
@qd.kernel
def k_col_fixed_inline(A: qd.types.ndarray(dtype=float, ndim=2), U: qd.types.ndarray(dtype=float, ndim=2), n: qd.i32):
    qd.loop_config(serialize=True)
    for j in range(n):
        s = A[j, j]
        qd.loop_config(serialize=True)
        for k in range(j):
            s -= U[j, k] * U[j, k]
        U[j, j] = qd.sqrt(s)
        qd.loop_config(serialize=True)
        for i in range(j):
            si = A[i, j]
            qd.loop_config(serialize=True)
            for k in range(i):
                si -= U[i, k] * U[j, k]
            U[i, j] = si / U[j, j]


@qd.func
def chol_col_fixed_func(A: qd.types.ndarray(dtype=float, ndim=3), U: qd.types.ndarray(dtype=float, ndim=3), n: int):
    qd.loop_config(serialize=True)
    for j in range(n):
        s = A[0, j, j]
        qd.loop_config(serialize=True)
        for k in range(j):
            s -= U[0, j, k] * U[0, j, k]
        U[0, j, j] = qd.sqrt(s)
        qd.loop_config(serialize=True)
        for i in range(j):
            si = A[0, i, j]
            qd.loop_config(serialize=True)
            for k in range(i):
                si -= U[0, i, k] * U[0, j, k]
            U[0, i, j] = si / U[0, j, j]
    return 0.0


@qd.kernel
def k_col_fixed_func(A: qd.types.ndarray(dtype=float, ndim=3), U: qd.types.ndarray(dtype=float, ndim=3), n: qd.i32):
    chol_col_fixed_func(A, U, n)


# --- row-order L-form Cholesky (control) ------------------------------------
@qd.kernel
def k_row_L(A: qd.types.ndarray(dtype=float, ndim=2), U: qd.types.ndarray(dtype=float, ndim=2), n: qd.i32):
    qd.loop_config(serialize=True)
    for i in range(n):
        qd.loop_config(serialize=True)
        for j in range(i + 1):
            s = A[i, j]
            qd.loop_config(serialize=True)
            for k in range(j):
                s -= U[i, k] * U[j, k]
            if i == j:
                U[i, j] = qd.sqrt(s)
            else:
                U[i, j] = s / U[j, j]


# --- original (invalid-order) form: faithful-compilation pin ----------------
@qd.kernel
def k_col_invalid(A: qd.types.ndarray(dtype=float, ndim=2), U: qd.types.ndarray(dtype=float, ndim=2), n: qd.i32):
    qd.loop_config(serialize=True)
    for j in range(n):
        qd.loop_config(serialize=True)
        for i in range(j + 1):
            s = A[i, j]
            qd.loop_config(serialize=True)
            for k in range(i):
                s -= U[k, i] * U[k, j]
            if i == j:
                U[i, j] = qd.sqrt(s)
            else:
                U[i, j] = s / U[j, j]


# --- numpy twins (exact sequential semantics, f32) ---------------------------
def twin_col_fixed(H):
    n = H.shape[0]
    U = np.zeros_like(H)
    for j in range(n):
        s = H[j, j]
        for k in range(j):
            s -= U[j, k] * U[j, k]
        U[j, j] = np.sqrt(s)
        for i in range(j):
            si = H[i, j]
            for k in range(i):
                si -= U[i, k] * U[j, k]
            U[i, j] = si / U[j, j]
    return U


def twin_row_L(H):
    n = H.shape[0]
    U = np.zeros_like(H)
    for i in range(n):
        for j in range(i + 1):
            s = H[i, j]
            for k in range(j):
                s -= U[i, k] * U[j, k]
            if i == j:
                U[i, j] = np.sqrt(s)
            else:
                U[i, j] = s / U[j, j]
    return U


def twin_col_invalid(H):
    n = H.shape[0]
    U = np.zeros_like(H)
    for j in range(n):
        for i in range(j + 1):
            s = H[i, j]
            for k in range(i):
                s -= U[k, i] * U[k, j]
            if i == j:
                U[i, j] = np.sqrt(s)
            else:
                U[i, j] = s / U[j, j]
    return U


FAILURES = 0


def check(name, ok, detail=""):
    global FAILURES
    if not ok:
        FAILURES += 1
    print(f"[{ARCH}] [{name}] {'PASS' if ok else 'FAIL'} {detail}")


def run2d(kern, H):
    n = H.shape[0]
    A = qd.ndarray(dtype=qd.f32, shape=(n, n)); A.from_numpy(H.copy())
    U = qd.ndarray(dtype=qd.f32, shape=(n, n)); U.fill(0)
    kern(A, U, n)
    qd.sync()
    return U.to_numpy()


def run3d(kern, H):
    n = H.shape[0]
    A = qd.ndarray(dtype=qd.f32, shape=(1, n, n)); A.from_numpy(H[None].copy())
    U = qd.ndarray(dtype=qd.f32, shape=(1, n, n)); U.fill(0)
    kern(A, U, n)
    qd.sync()
    return U.to_numpy()[0]


# A: corrected col-order, inline 2-D, n=8 and n=24
for n in (8, 24):
    H = make_H(n)
    got = run2d(k_col_fixed_inline, H)
    ref = twin_col_fixed(H)
    err = np.abs(got - ref).max()
    check(f"A col_fixed inline n={n}", (not np.isnan(got).any()) and err < 1e-5, f"err={err:.2e}")

# B: corrected col-order, func + 3-D, n=8 and n=24
for n in (8, 24):
    H = make_H(n)
    got = run3d(k_col_fixed_func, H)
    ref = twin_col_fixed(H)
    err = np.abs(got - ref).max()
    check(f"B col_fixed func+3d n={n}", (not np.isnan(got).any()) and err < 1e-5, f"err={err:.2e}")

# C: row-order control, n=8
H = make_H(N)
got = run2d(k_row_L, H)
ref = twin_row_L(H)
err = np.abs(got - ref).max()
check("C row_L control", (not np.isnan(got).any()) and err < 1e-5, f"err={err:.2e}")

# D: invalid-order form == numpy twin exactly (NaN pattern + finite values)
got = run2d(k_col_invalid, H)
ref = twin_col_invalid(H)
same_nan = np.array_equal(np.isnan(got), np.isnan(ref))
finite_eq = np.array_equal(got[~np.isnan(got)], ref[~np.isnan(ref)])
check("D invalid==twin nan-pattern", same_nan)
check("D invalid==twin finite-values", finite_eq)
if not (np.isnan(ref).any()):
    check("D twin-selfcheck", False, "twin unexpectedly finite; probe assumptions broken")

print(f"[{ARCH}] RESULT: {'PASS' if FAILURES == 0 else 'FAIL'} ({FAILURES} failures)")
sys.exit(0 if FAILURES == 0 else 1)
