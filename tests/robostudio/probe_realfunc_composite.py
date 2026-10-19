"""Family R gate probe: @qd.func x composite values across real_func callers (A41/A44/A33).

Regression surface for the Family R fix (ledger A41/A44/A33; builder resolution
via compiling_callable in Func.__call__'s plain-func path):

  M1 direct          : early_return leaf called directly in a kernel        -> OK
  M2 func dispatcher : leaf via @qd.func dispatcher                          -> OK
  M3 real_func disp  : leaf via @qd.real_func dispatcher (THE A41/A44 bug)   -> OK
  C1 scalar dispatch : real_func if/else over scalar literal returns         -> OK
  C2 cross-vals      : real_func returning func-produced composite (A33)     -> OK

All cases computed against numpy-twin wants; CPU-only (real_func on SPIR-V
stays loudly rejected per the M3 fallback).

Usage: python probe_realfunc_composite.py [cpu]   (default cpu)
Prints "REALFUNC COMPOSITE GATE [cpu]: ALL PASS" on success.
"""
import sys

import numpy as np

import quadrants as qd
from typing import Tuple

ARCH = sys.argv[1] if len(sys.argv) > 1 else "cpu"
qd.init(arch=getattr(qd, ARCH), offline_cache=False, cfg_optimization=False)

fails = []


def check(name, ok, detail=""):
    print(f"  [{ARCH}] {name:44s} {'PASS' if ok else 'FAIL'} {detail}")
    if not ok:
        fails.append(name)


@qd.func(early_return=True)
def leaf_er(pnt: qd.math.vec3) -> Tuple[float, qd.math.vec3]:
    d = -1.0
    n = qd.math.vec3(0.0, 0.0, 0.0)
    done = False
    if not done and pnt[0] > -100.0:
        d = 2.0
        n = qd.math.vec3(0.0, 1.0, 0.0)
        done = True
    if not done and pnt[1] > 100.0:
        d = 3.0
        n = qd.math.vec3(1.0, 0.0, 0.0)
        done = True
    return d, n


@qd.real_func
def disp_er(pnt: qd.math.vec3) -> Tuple[float, qd.math.vec3]:
    return leaf_er(pnt)


@qd.func
def disp_f(pnt: qd.math.vec3) -> Tuple[float, qd.math.vec3]:
    return leaf_er(pnt)


@qd.kernel
def k_direct(pnt_in: qd.types.ndarray(dtype=qd.math.vec3, ndim=1),
             out_dist: qd.types.ndarray(dtype=qd.f32, ndim=1),
             out_nrm: qd.types.ndarray(dtype=qd.math.vec3, ndim=1), __n0: int):
    for i in range(__n0):
        d, n = leaf_er(pnt_in[i])
        out_dist[i] = d
        out_nrm[i] = n


@qd.kernel
def k_via_f(pnt_in: qd.types.ndarray(dtype=qd.math.vec3, ndim=1),
            out_dist: qd.types.ndarray(dtype=qd.f32, ndim=1),
            out_nrm: qd.types.ndarray(dtype=qd.math.vec3, ndim=1), __n0: int):
    for i in range(__n0):
        d, n = disp_f(pnt_in[i])
        out_dist[i] = d
        out_nrm[i] = n


@qd.kernel
def k_via_r(pnt_in: qd.types.ndarray(dtype=qd.math.vec3, ndim=1),
            out_dist: qd.types.ndarray(dtype=qd.f32, ndim=1),
            out_nrm: qd.types.ndarray(dtype=qd.math.vec3, ndim=1), __n0: int):
    for i in range(__n0):
        d, n = disp_er(pnt_in[i])
        out_dist[i] = d
        out_nrm[i] = n


@qd.func
def leaf_a(pnt: qd.math.vec3) -> Tuple[float, qd.math.vec3]:
    return 1.0, qd.math.vec3(0.0, 0.0, 1.0)


@qd.func
def leaf_b2(pnt: qd.math.vec3) -> Tuple[float, qd.math.vec3]:
    return 2.0, qd.math.vec3(0.0, 1.0, 0.0)


@qd.real_func
def disp_pick(geomtype: int, pnt: qd.math.vec3) -> Tuple[float, qd.math.vec3]:
    if geomtype == 0:
        return leaf_a(pnt)
    else:
        return leaf_b2(pnt)


@qd.kernel
def k_pick(gt: qd.types.ndarray(dtype=qd.i32, ndim=1),
           pnt_in: qd.types.ndarray(dtype=qd.math.vec3, ndim=1),
           out_dist: qd.types.ndarray(dtype=qd.f32, ndim=1),
           out_nrm: qd.types.ndarray(dtype=qd.math.vec3, ndim=1), __n0: int):
    for i in range(__n0):
        d, n = disp_pick(gt[i], pnt_in[i])
        out_dist[i] = d
        out_nrm[i] = n


pnt = qd.ndarray(qd.math.vec3, (1,))
pnt.from_numpy(np.array([[0.1, 0.2, 0.3]], dtype=np.float32))
gt = qd.ndarray(qd.i32, (1,))
gt.from_numpy(np.array([1], dtype=np.int32))
od = qd.ndarray(qd.f32, (1,))
on = qd.ndarray(qd.math.vec3, (1,))
WANT = (2.0, [0.0, 1.0, 0.0])


def run(kf, with_gt=False):
    od.from_numpy(np.zeros(1, dtype=np.float32))
    on.from_numpy(np.zeros((1, 3), dtype=np.float32))
    if with_gt:
        kf(gt, pnt, od, on, 1)
    else:
        kf(pnt, od, on, 1)
    qd.sync()
    return float(od.to_numpy()[0]), on.to_numpy()[0].tolist()


def ok_tuple(got):
    return abs(got[0] - WANT[0]) < 1e-6 and all(abs(got[1][i] - WANT[1][i]) < 1e-6 for i in range(3))


d, n = run(k_direct)
check("M1 direct early_return leaf", ok_tuple((d, n)), f"got {d}/{n}")
d, n = run(k_via_f)
check("M2 via @qd.func dispatcher", ok_tuple((d, n)), f"got {d}/{n}")
d, n = run(k_via_r)
check("M3 via @qd.real_func dispatcher", ok_tuple((d, n)), f"got {d}/{n}")

# C1 scalar literal dispatch in a real_func
@qd.real_func
def pick_s(geomtype: int, x: qd.f32) -> qd.f32:
    if geomtype == 0:
        return x * 1.0
    else:
        return x * 2.0


@qd.kernel
def k_s(gt: qd.types.ndarray(dtype=qd.i32, ndim=1), x_in: qd.types.ndarray(dtype=qd.f32, ndim=1),
        out: qd.types.ndarray(dtype=qd.f32, ndim=1), __n0: int):
    for i in range(__n0):
        out[i] = pick_s(gt[i], x_in[i])


x1 = qd.ndarray(qd.f32, (1,))
x1.from_numpy(np.array([3.0], dtype=np.float32))
os1 = qd.ndarray(qd.f32, (1,))
ok_s = True
for g, want in ((0, 3.0), (1, 6.0)):
    gt.from_numpy(np.array([g], dtype=np.int32))
    os1.from_numpy(np.zeros(1, dtype=np.float32))
    k_s(gt, x1, os1, 1)
    qd.sync()
    got = os1.to_numpy()[0]
    if abs(got - want) > 1e-6:
        ok_s = False
        print(f"    C1 scalar dispatch gt={g}: got {got} want {want}")
check("C1 scalar literal dispatch in real_func", ok_s)

# C2 real_func returning func-produced composite (A33 direction)
@qd.real_func
def disp_pick2(geomtype: int, pnt: qd.math.vec3) -> Tuple[float, qd.math.vec3]:
    if geomtype == 0:
        return leaf_a(pnt)
    else:
        return leaf_b2(pnt)


@qd.kernel
def k_pick2(gt: qd.types.ndarray(dtype=qd.i32, ndim=1),
            pnt_in: qd.types.ndarray(dtype=qd.math.vec3, ndim=1),
            out_dist: qd.types.ndarray(dtype=qd.f32, ndim=1),
            out_nrm: qd.types.ndarray(dtype=qd.math.vec3, ndim=1), __n0: int):
    for i in range(__n0):
        d, n = disp_pick2(gt[i], pnt_in[i])
        out_dist[i] = d
        out_nrm[i] = n


ok_c2 = True
for g, wd, wn in ((0, 1.0, [0.0, 0.0, 1.0]), (1, 2.0, [0.0, 1.0, 0.0])):
    gt.from_numpy(np.array([g], dtype=np.int32))
    od.from_numpy(np.zeros(1, dtype=np.float32))
    on.from_numpy(np.zeros((1, 3), dtype=np.float32))
    k_pick2(gt, pnt, od, on, 1)
    qd.sync()
    got_d = od.to_numpy()[0]
    got_n = on.to_numpy()[0].tolist()
    if abs(got_d - wd) > 1e-6 or any(abs(got_n[i] - wn[i]) > 1e-6 for i in range(3)):
        ok_c2 = False
        print(f"    C2 dispatch gt={g}: got {got_d}/{got_n} want {wd}/{wn}")
check("C2 real_func composite dispatch (A33 dir)", ok_c2)

if fails:
    print(f"REALFUNC COMPOSITE GATE [{ARCH}]: FAIL ({len(fails)} case(s))")
    sys.exit(1)
print(f"REALFUNC COMPOSITE GATE [{ARCH}]: ALL PASS")
