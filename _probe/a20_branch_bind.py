"""A20 repro #3: faithful shape -- first binding happens by tuple-unpack INSIDE
a runtime-if branch; later sibling runtime-if (inside a static-unrolled for)
reads those names. This is collision_primitive.py primitive_narrowphase:

    if not doneqd:                      # branch A (closes)
        _, margin, ... = contact_params(...)
    if not doneqd:                      # branch B (closes)
        geom1, geom2 = geom_collision_pair(...)
    if not doneqd:
        for i in qd.static(range(N)):
            t1 = qd.static(types[i][0])
            if t1 == type1 and t2 == type2:
                workers[i](geom1, geom2, margin, ...)   # reads across closed branches

No predeclaration of geom1/geom2/margin at kernel top (that's the real code).
Run: <venv python> a20_branch_bind.py
exit 0 = compiles+runs correct (fixed/never-broken), 1 = A20 signature, 2 = other.
"""
import sys
import traceback

import numpy as np

import quadrants as qd


@qd.func
def contact_params_fake(cutoff_in: qd.types.ndarray(dtype=qd.f32, ndim=1), tid: int) -> tuple:
    z = qd.cast(0, qd.i32)
    m = cutoff_in[tid]
    g = m * 0.5
    c = z
    fr = m * 0.1
    sr = m * 0.2
    sfr = m * 0.3
    si = m * 0.4
    ad = m * 0.5
    return z, m, g, c, fr, sr, sfr, si, ad


@qd.func
def pair_fake(sel: qd.types.ndarray(dtype=qd.i32, ndim=1), tid: int) -> tuple:
    g1 = sel[tid]
    g2 = sel[tid] + 1
    return g1, g2


@qd.func
def worker(
    g1: int,
    g2: int,
    w: int,
    m: float,
    out: qd.types.ndarray(dtype=qd.f32, ndim=1),
) -> None:
    out[0] = out[0] + qd.cast(g1 * 10 + g2, qd.f32) + m


@qd.kernel
def probe(
    sel: qd.types.ndarray(dtype=qd.i32, ndim=1),
    cutoff_in: qd.types.ndarray(dtype=qd.f32, ndim=1),
    out: qd.types.ndarray(dtype=qd.f32, ndim=1),
):
    for t in range(1):
        doneqd = False
        tid = t
        if tid >= 1:
            doneqd = True
        if not doneqd:
            _, margin, gap, condim, friction, solref, solreffriction, solimp, adhesion = contact_params_fake(cutoff_in, tid)
        if not doneqd:
            geom1, geom2 = pair_fake(sel, tid)
        if not doneqd:
            for i in qd.static(range(2)):
                type1 = qd.static(0)
                type2 = qd.static(i)
                if type2 == 1:
                    worker(geom1, geom2, tid, margin, out)


def main() -> int:
    qd.init()
    sel = np.array([2, 0], dtype=np.int32)
    cutoff_in = np.array([0.5], dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)
    try:
        probe(sel, cutoff_in, out)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        if "geom1" in tb and (
            "is not defined" in tb or "no attribute 'geom1'" in tb
        ):
            actionable = "Predeclare it before the block" in tb
            chained = "During handling" in tb or "no attribute 'geom1'" in tb
            print(
                "REPRO: geom1 name-resolution failure "
                f"(actionable diagnostic: {actionable}; misleading chained AttributeError: {chained})"
            )
            if "-v" in sys.argv:
                print(tb)
            return 1
        print("REPRO: kernel failed with an UNEXPECTED error (not the A20 signature)")
        print(tb)
        return 2
    qd.sync()
    got = float(out[0])
    # worker called once with g1=sel[0]=2, g2=3, w=0, m=0.5 -> 23 + 0.5 = 23.5
    if abs(got - 23.5) > 1e-5:
        print(f"REPRO: compiled but WRONG VALUE: out[0]={got}, expected 23.5")
        return 2
    print(f"ok: A20 branch-bind probe compiles and runs (fixed state); out[0] = {got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
