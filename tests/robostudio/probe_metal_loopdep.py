"""Gate probe (A5): Metal/SPIR-V miscompile — loop-carried memory dependence violated.

A conditional read-modify-write of the SAME array element inside a runtime loop loses
updates on Metal: the element load is hoisted/GVN'd across the loop-carried dependency.
CPU is correct on the identical code (6.0); Metal returns 4.0 (only the last iteration's
store survives, each iteration reads the stale pre-loop value 0).

This is the minimal form. Compound finding: A2 (Tile16x16 in runtime loops, m>=3) shares
the "loop-carried dependency broken on SPIR-V/Metal" signature — investigate together.

Run: python probe_metal_loopdep.py   (exit 0 = backend correct; Metal expected FAIL
until fixed — this gate pins the bug so the fix shows as a flip to PASS)
"""
import sys

import numpy as np

import quadrants as qd

qd.init(arch=qd.metal if (len(sys.argv) > 1 and sys.argv[1] == "metal") else qd.cpu)

xs = qd.ndarray(dtype=qd.f32, shape=(4,))
xs.from_numpy(np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32))


# variant C (minimal): plain conditional accumulate, no flags, no tiles
@qd.kernel
def cond_accum(xs: qd.types.ndarray(ndim=1), acc: qd.types.ndarray(ndim=1)):
    for _ in range(1):
        for i in range(xs.shape[0]):
            if xs[i] >= 0.0:
                acc[0] = acc[0] + xs[i]


acc = qd.ndarray(dtype=qd.f32, shape=(1,))
acc.from_numpy(np.zeros(1, dtype=np.float32))
cond_accum(xs, acc)
qd.sync()
got = float(acc.to_numpy()[0])
ok = np.isclose(got, 6.0)
print(f"A5 loop-dep gate: acc={got} (expect 6.0) -> {'PASS' if ok else 'FAIL (miscompile present)'}")
sys.exit(0 if ok else 1)
