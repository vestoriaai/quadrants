"""C1a fork-matrix gates: scalar change / array death / correctness."""
import sys
sys.path.insert(0, '/tmp/c1site')
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf')
import quadrants as qd
qd.init(arch=qd.cpu, offline_cache=False, cfg_optimization=False)
import numpy as np
from mujoco_qd._src.warp_util import zeros

fails = []

# 1. scalar value change (same id impossible for distinct small ints, but
#    large floats are freshly allocated each call - id may recycle)
x = zeros((4,), dtype=float)
y = zeros((4,), dtype=float)

@qd.kernel
def k(a_in: qd.types.ndarray(dtype=float, ndim=1), b_out: qd.types.ndarray(dtype=float, ndim=1), s: float):
    for __t in range(4):
        b_out[__t] = a_in[__t] * s

for s in (2.0, 3.5, 2.0, 100.25, 2.0):
    x.fill(2.0)
    k(x, y, s)
    qd.sync()
    if not np.allclose(y.to_numpy(), 2.0 * s):
        fails.append(f'scalar {s}: got {y.to_numpy()[0]}')
print('scalar-change gate:', 'PASS' if not fails else fails)

# 2. array death + id reuse: launch on temp arrays, drop them, new arrays
def run_temp():
    t1 = zeros((4,), dtype=float)
    t2 = zeros((4,), dtype=float)
    t1.fill(5.0)
    k(t1, t2, 2.0)
    qd.sync()
    return t2.to_numpy().copy()

r1 = run_temp()
r2 = run_temp()  # may reuse ids
x.fill(5.0)
k(x, y, 2.0)
qd.sync()
ref = y.to_numpy()
if not (np.allclose(r1, ref) and np.allclose(r2, ref)):
    fails.append(f'array-death: {r1} {r2} vs {ref}')
print('array-death gate:', 'PASS' if np.allclose(r1, ref) and np.allclose(r2, ref) else 'FAIL')

# 3. float id-recycle adversarial: many temp floats cycling through one call site
bad = False
seen = {}
for trial in range(200):
    sval = 1.0 + trial * 0.001
    x.fill(1.0)
    k(x, y, sval)
    qd.sync()
    got = float(y.to_numpy()[0])
    if not np.isclose(got, sval, rtol=1e-5):
        bad = True
        fails.append(f'float-recycle trial {trial}: want {sval} got {got}')
        break
print('float-recycle gate:', 'PASS' if not bad else 'FAIL')

print('ALL:', 'PASS' if not fails else f'{len(fails)} FAILURES')
