"""A20 probe #4: probe #3's kernel WITH the mujoco_qd-style predeclarations
added (the convention sensor.py kernels already follow). Mirrors the fix recipe
for collision_primitive.py `_primitive_narrowphase`: predeclare every name that
the kernel first binds inside a runtime-if branch (geom1/geom2 + the
contact_params tuple members) at the top of the enclosing block, so later
sibling blocks can read them via the assign-to-declared path.

Run: <venv python> a20_predecl_fix.py
exit 0 = compiles and computes correctly (proves the predecl recipe),
1 = A20 signature still fires (recipe insufficient), 2 = other unexpected error.
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
        # predecl: every branch-first-bound name (the fix recipe)
        _ = 0
        geom1 = 0
        geom2 = 0
        margin = 0.0
        gap = 0.0
        condim = 0
        friction = 0.0
        solref = 0.0
        solreffriction = 0.0
        solimp = 0.0
        adhesion = 0.0
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
            print("REPRO: geom1 failure still fires despite predecls (recipe insufficient)")
            return 1
        print("REPRO: kernel failed with an UNEXPECTED error")
        print(tb)
        return 2
    qd.sync()
    got = float(out[0])
    if abs(got - 23.5) > 1e-5:
        print(f"REPRO: compiled but WRONG VALUE: out[0]={got}, expected 23.5")
        return 2
    print(f"ok: predecl recipe compiles and computes correctly; out[0] = {got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
