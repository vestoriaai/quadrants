"""A20 minimal repro: loop variable referenced inside a nested range-for that
sits in a runtime `elif` branch of an outer range-for, with outer-scope locals
predeclared at kernel top (mujoco_qd sensor.py _sensor_pos shape, minimized).

Asserts the failure mode (QuadrantsNameError: Name "geom1" is not defined /
builtins AttributeError) at the unfixed state, or successful compile at the
fixed state. Never prints "ok" unconditionally.

Run: <venv python> a20_min_repro.py
"""
import sys
import traceback

import numpy as np
import quadrants as qd


@qd.kernel
def probe(
    ngeom: int,
    body_geomnum: qd.types.ndarray(dtype=qd.i32, ndim=1),
    body_geomadr: qd.types.ndarray(dtype=qd.i32, ndim=1),
    geom_type: qd.types.ndarray(dtype=qd.i32, ndim=1),
    col_in: qd.types.ndarray(dtype=qd.f32, ndim=3),
    out: qd.types.ndarray(dtype=qd.f32, ndim=1),
    __n0: int,
    __n1: int,
):
    geomid1 = 0
    geomid2 = 0
    dist = 0.0
    dist_new = 0.0
    id1 = 0
    id2 = 0
    n1 = 0
    n2 = 0
    objid = 0
    for __t in range(__n0 * __n1):
        worldid = __t // __n1
        objid = worldid
        pass
        if objid < 0:
            dist = -1.0
        elif objid >= 0:
            n1 = body_geomnum[objid]
            id1 = body_geomadr[objid]
            n2 = 1
            id2 = 0
            for geom1 in range(n1):
                geomid1 = id1 + geom1
                for geom2 in range(n2):
                    geomid2 = id2 + geom2
                    collisionid = geomid1 * ngeom + geomid2
                    for i in range(2):
                        dist_new = col_in[worldid, collisionid, i]
                        if dist_new < dist:
                            dist = dist_new
                            if geom_type[geomid1] > geom_type[geomid2]:
                                dist = dist_new + 1.0
                            else:
                                dist = dist_new
        pass
    out[0] = dist


def main() -> int:
    qd.init()
    ngeom, nbody = 4, 2
    body_geomnum = np.zeros(nbody, dtype=np.int32)
    body_geomadr = np.zeros(nbody, dtype=np.int32)
    geom_type = np.zeros(ngeom, dtype=np.int32)
    col_in = np.zeros((2, ngeom * ngeom, 2), dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)
    body_geomnum[0], body_geomnum[1] = 2, 1
    body_geomadr[0], body_geomadr[1] = 0, 2
    for g in range(ngeom):
        geom_type[g] = g % 3
    for w in range(2):
        for p in range(ngeom * ngeom):
            col_in[w, p, 0] = 0.5 + p * 0.01
            col_in[w, p, 1] = 0.25 + p * 0.01
    try:
        probe(ngeom, body_geomnum, body_geomadr, geom_type, col_in, out, 2, 1)
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
    # compile+run fine
    qd.sync()
    print("ok: A20 repro compiles and runs (fixed state); out[0] =", float(out[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
