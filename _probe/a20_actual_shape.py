"""A20 repro #2: the actual collision_primitive.py shape.

    if not done:                      # runtime if, branch A
        geom1, geom2 = pair_func(...)  # tuple-unpack binds geom1/geom2 in branch scope
    ...
    if not done:                      # separate runtime if
        for i in qd.static(range(2)):  # static-unrolled loop
            if type_match:             # runtime if
                worker(...geom1...)    # READ of geom1 -- name resolution crosses the closed branch

Run: <venv python> a20_actual_shape.py
exit 0 = compiles (fixed/never-broken state), 1 = A20 signature reproduced,
2 = other unexpected error.
"""
import sys
import traceback

import numpy as np

import quadrants as qd


@qd.func
def pair_func(sel: int) -> tuple:
    g1 = sel * 2
    g2 = sel * 2 + 1
    return g1, g2


@qd.func
def worker(g1: int, g2: int, w: int, out: qd.types.ndarray(dtype=qd.f32, ndim=1)) -> None:
    out[0] = out[0] + qd.cast(g1 * 10 + g2 + w, qd.f32)


@qd.kernel
def probe(sel: int, done_flag: qd.types.ndarray(dtype=qd.i32, ndim=1), out: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    done = False
    geom1 = 0
    geom2 = 0
    for t in range(1):
        if done_flag[t] == 0:
            geom1, geom2 = pair_func(sel)
        if not done:
            for i in qd.static(range(2)):
                type_match = (i == sel)
                if type_match:
                    worker(geom1, geom2, t, out)


def main() -> int:
    qd.init()
    done_flag = np.zeros(1, dtype=np.int32)
    out = np.zeros(1, dtype=np.float32)
    try:
        probe(1, done_flag, out)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        if "geom1" in tb and (
            "is not defined" in tb or "no attribute 'geom1'" in tb
        ):
            print("REPRO: geom1 name-resolution failure (regressed state)")
            return 1
        print("REPRO: kernel failed with an UNEXPECTED error (not the A20 signature)")
        print(tb)
        return 2
    qd.sync()
    got = float(out[0])
    if got != 23.0:  # sel=1 -> geom1=2, geom2=3, w=0 -> 2*10+3+0
        print(f"REPRO: compiled but WRONG VALUE: out[0]={got}, expected 23.0")
        return 2
    print(f"ok: A20 actual-shape probe compiles and runs (fixed state); out[0] = {got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
