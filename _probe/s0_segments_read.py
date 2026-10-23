"""S0a v3: read segment accumulator after instrumented trivial launches."""
import sys
sys.path.insert(0, '/Users/mini0/Code/quadrants-c1wt/python')
import quadrants as qd
qd.init(arch=qd.cpu, offline_cache=False, cfg_optimization=False)
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf')
from mujoco_qd._src.warp_util import zeros
import time

KMOD = sys.modules['quadrants.lang.kernel']

@qd.kernel
def trivial(x_out: qd.types.ndarray(dtype=float, ndim=1), __n0: int):
    for __t in range(__n0):
        x_out[__t] = 1.0

x = zeros((2,), dtype=float)
trivial(x, 2)
qd.sync()
KMOD._SEG_ACC.clear()

N = 3000
t0 = time.perf_counter()
for _ in range(N):
    trivial(x, 2)
qd.sync()
wall = (time.perf_counter() - t0) / N * 1e6
print(f'wall (instrumented): {wall:.2f} us/launch')
tot = 0.0
for k, v in sorted(KMOD._SEG_ACC.items(), key=lambda kv: -kv[1]):
    print(f'  {k:12s} {v/N*1e6:>7.2f} us/launch')
    tot += v / N * 1e6
print(f'  segments sum {tot:.2f}; outside launch_kernel body approx {wall - tot:.2f}')
