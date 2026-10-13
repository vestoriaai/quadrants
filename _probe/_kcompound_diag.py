"""A7/M2 side-finding evidence: KERNEL loop-invariant flag + compound work in runtime-if.

EXISTING CORE BUG (not introduced by the early_return line): identical failure on the
pre-fix baseline wheel (0de907cd) and the dev104 fix wheel — dual-wheel attribution.
Same LICM/over-hoisting family as the documented inlined-func prologue-store hazard
(early_return_transformer NOTE), CPU included, so not the Metal MSL family.

Trigger (pinned repros below; exact minimal trigger needs its own attribution pass):
a scalar flag conditionally assigned inside a kernel-loop iteration and used as a
runtime-if condition over a block containing compound work. Observed matrix:

  k_plain    no guard, element store                                      -> PASS
  k_manual7  flag OUTSIDE loop; element store + read inside guard         -> FAIL
  k_manual7b flag OUTSIDE loop; predeclared v, store + read inside guard  -> FAIL
  k_manual8  flag OUTSIDE loop; compound in post-loop region              -> FAIL
  kD         flag INSIDE loop; whole-value store of CONSTANT entries      -> FAIL
             (2-iteration delay pattern in got[])

Controls that PASS: element store under a plain data condition (narrow2 A/C),
whole-value store under a plain data condition (narrow2 B), flag-INSIDE-loop +
element store with/without in-guard read (_kcompound_narrow3.py kE/kF). Note the
func-side forms of the early_return F-matrix (F10/F13: whole-value stores of
VARIABLE entries inside the func's merged guard) pass — the boundary is
entry-constant-ness / inlining sensitive; treat as open for the compiler owner.

Impact on the A7/M2 line: kernel-side early_return generates the flag at function
level (outside the outermost loop), so guard regions containing compound work hit
THIS bug — the pass output is correct there (hand-written identical shape fails
the same way). Func-side compound forms are unaffected (F1-F6, F9-F14 pass
cpu+metal). Kernel-side F7/F8 therefore pin the guard-merge with compound funcs
in the UNGUARDED prefix and scalar-only guard regions.
Run: <venv>/bin/python _kcompound_diag.py [cpu|metal]
"""
import sys

import numpy as np

import quadrants as qd

ARCH = sys.argv[1] if len(sys.argv) > 1 else "cpu"
qd.init(arch=qd.metal if ARCH == "metal" else qd.cpu)
vec3f = qd.types.vector(3, qd.f32)

FX = [-7.5, -0.5, 0.0, 0.5, 3.0, 20.0, 100.0]
fx = qd.ndarray(dtype=qd.f32, shape=(len(FX),))
fx.from_numpy(np.array(FX, dtype=np.float32))

results = []


def check(name, got, want):
    ok = np.allclose(got, want)
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL':4}] {name} got={got} want={want}")


# sanity: no guard
@qd.kernel
def k_plain(xs: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        v = vec3f(0.0)
        v[0] = xs[i]
        y[i] = v[0] + v[2]


y = qd.ndarray(dtype=qd.f32, shape=(len(FX),))
y.from_numpy(np.zeros(len(FX), dtype=np.float32))
k_plain(fx, y)
qd.sync()
check("k_plain_no_guard", y.to_numpy(), [x for x in FX])  # v[0]+v[2] = x + 0

# guarded, branch-local compound
@qd.kernel
def k_manual7(xs: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=1)):
    __er_done = qd.cast(0, qd.i32)
    for i in range(xs.shape[0]):
        if xs[i] < 0.0:
            __er_done = 1
        if __er_done == 0:
            v = vec3f(0.0)
            v[0] = xs[i]
            y[i] = v[0] + v[2]


y.from_numpy(np.zeros(len(FX), dtype=np.float32))
k_manual7(fx, y)
qd.sync()
check("k_manual7_guard_local", y.to_numpy(), [max(x, 0.0) for x in FX])

# guarded, function-level predeclared compound
@qd.kernel
def k_manual7b(xs: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=1)):
    __er_done = qd.cast(0, qd.i32)
    for i in range(xs.shape[0]):
        v = vec3f(0.0)
        if xs[i] < 0.0:
            __er_done = 1
        if __er_done == 0:
            v[0] = xs[i]
            y[i] = v[0] + v[2]


y.from_numpy(np.zeros(len(FX), dtype=np.float32))
k_manual7b(fx, y)
qd.sync()
check("k_manual7b_guard_predecl", y.to_numpy(), [max(x, 0.0) for x in FX])

# compound in the post-loop fall-through region
@qd.kernel
def k_manual8(xs: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=1), z: qd.types.ndarray(ndim=1)):
    __er_done = qd.cast(0, qd.i32)
    for i in range(xs.shape[0]):
        s = 0.0
        for j in range(8):
            s += 0.5
            if s > xs[i]:
                y[i] = s
                __er_done = 1
                break
        if __er_done == 0:
            v = vec3f(s, 0.0, 0.0)
            z[i] = v[0]


y2 = qd.ndarray(dtype=qd.f32, shape=(len(FX),))
z = qd.ndarray(dtype=qd.f32, shape=(len(FX),))
y2.from_numpy(np.zeros(len(FX), dtype=np.float32))
z.from_numpy(np.zeros(len(FX), dtype=np.float32))
k_manual8(fx, y2, z)
qd.sync()
want_z = []
for x in FX:
    s = 0.0
    hit = False
    for _ in range(8):
        s += 0.5
        if s > x:
            hit = True
            break
    want_z.append(0.0 if hit else s)
check("k_manual8_postloop", z.to_numpy(), want_z)

# kD: flag INSIDE the loop, whole-value compound store under the flag condition
@qd.kernel
def k_d(xs: qd.types.ndarray(ndim=1), y: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        done = qd.cast(0, qd.i32)
        v = vec3f(0.0)
        if xs[i] < 0.0:
            done = 1
        if done == 0:
            v = vec3f(-1.0, -2.0, -3.0)
        y[i] = v[0] + v[2]


y.from_numpy(np.zeros(len(FX), dtype=np.float32))
k_d(fx, y)
qd.sync()
check("kD_flag_inside_whole_store", y.to_numpy(), [-4.0 if x < 0 else 0.0 for x in FX])

bad = [n for n, ok in results if not ok]
print(f"\nKCOMPOUND DIAG arch={ARCH}: {'bug PRESENT' if bad else 'bug ABSENT (' + str(len(results)) + ' sanity forms pass)'} — failing: {bad}")
# This evidence probe exits 0 when the BUG IS PRESENT (it documents the defect);
# flip the expectation when the compiler is fixed.
sys.exit(0 if bad else 1)
