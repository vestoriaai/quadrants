"""A7/M2 side-finding evidence: KERNEL loop-external flag semantics (ERRATUM 2026-10-13).

ERRATUM (supersedes the original "compound work in runtime-if miscompiles"
reading): the wants below originally assumed the flag resets every iteration.
The kernels DECLARE the flag OUTSIDE the loop; after offload such an alloca is
demoted to a GlobalTemporary (launch-shared scratch), so its true sequential
semantics are once-set-always, and under a PARALLEL element loop the shared
slot races across elements (metal wave scheduling widens the window; cpu pool
is near-sequential). The real boundary between the passing and failing forms
was never compound-vs-scalar work — it is the flag declaration position
(loop-external GlobalTemporary race vs in-loop per-element local). Controls
(narrow2 A/B/C, narrow3 kE/kF) PASS for the same reason; their original
"compound vs element store" explanation is retracted. Mechanism + production
impact: _probe/eval_kflag/ (eval1-eval6, census, offload IR).

Original observed matrix, re-read under the erratum (wants fixed below to the
code's sequential semantics):

  k_plain    no guard, element store                                      -> PASS
  k_manual7  flag OUTSIDE loop; element store + read inside guard         -> want was wrong
             (once-set-always: first negative element sets the flag and its
             own guard read already sees 1 -> nothing is ever written)
  k_manual7b flag OUTSIDE loop; predeclared v, store + read inside guard  -> want was wrong (same)
  k_manual8  flag OUTSIDE loop; compound in post-loop region              -> want was wrong
             (z guard can never pass once iteration 0 set the flag)
  kD         flag INSIDE loop; whole-value store of CONSTANT entries      -> want was INVERTED
             (done=1 for xs<0 keeps v at zero; done=0 for xs>=0 stores the
             negative vec; got=[0,0,-4,-4,...] matches the code bit-exactly;
             the "2-iteration delay" reading was a misparse)

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
# ERRATUM: flag is loop-external (once-set-always). FX[0] < 0 sets __er_done in
# iteration 0, and that same iteration's guard already reads 1, so no element
# ever writes y: sequential semantics = all zeros (not per-element max(x, 0)).
check("k_manual7_guard_local", y.to_numpy(), [0.0 for _ in FX])

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
# ERRATUM: same loop-external flag as k_manual7 -> sequential semantics = zeros.
check("k_manual7b_guard_predecl", y.to_numpy(), [0.0 for _ in FX])

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
# ERRATUM: the flag is loop-external, so iteration 0 (x=-7.5) sets __er_done and
# from then on the z guard never passes: sequential semantics for z = all zeros
# (the original want modeled a per-iteration flag reset the code does not have).
# y still receives the inner-loop hits themselves:
# [0.5, 0.5, 0.5, 1.0, 3.5, 0.0, 0.0] (x=20/100 outrun the 8 x 0.5 budget).
want_y8 = []
for x in FX:
    s = 0.0
    hit = False
    for _ in range(8):
        s += 0.5
        if s > x:
            hit = True
            break
    want_y8.append(s if hit else 0.0)
check("k_manual8_postloop_y", y2.to_numpy(), want_y8)
check("k_manual8_postloop_z", z.to_numpy(), [0.0 for _ in FX])

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
# ERRATUM: the original want was inverted (copied from kB, ignoring that the
# flag polarity here is done=1 for xs<0). done=1 keeps v at the zero vec, and
# done=0 (xs>=0) stores the negative vec: y = 0 for xs<0, -4 for xs>=0.
# got=[0, 0, -4, -4, ...] matches the code bit-exactly; the earlier
# "2-iteration delay" reading was a misparse.
check("kD_flag_inside_whole_store", y.to_numpy(),
      [0.0 if x < 0 else -4.0 for x in FX])

bad = [n for n, ok in results if not ok]
print(f"\nKCOMPOUND DIAG arch={ARCH}: "
      f"{'SEMANTICS DEVIATION (loop-external flag race visible)' if bad else 'all ' + str(len(results)) + ' forms match sequential semantics'}"
      f" — failing: {bad}")
# Post-erratum convention: exit 0 when every form matches the sequential
# semantics of its code (near-sequential pools: loop-external flags behave
# once-set-always deterministically); nonzero lists the deviating forms,
# which under parallel scheduling is the GlobalTemporary cross-element race.
sys.exit(0 if not bad else 1)
