"""S0a v4: is the 13us C++ segment the thread-pool barrier? Compare serial kernel."""
import sys
sys.path.insert(0, '/Users/mini0/Code/quadrants-c1wt/python')
import quadrants as qd
qd.init(arch=qd.cpu, offline_cache=False, cfg_optimization=False)
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf')
from mujoco_qd._src.warp_util import zeros
import time

KMOD = sys.modules['quadrants.lang.kernel']

@qd.kernel
def k_par(x_out: qd.types.ndarray(dtype=float, ndim=1), __n0: int):
    for __t in range(__n0):
        x_out[__t] = 1.0

@qd.kernel
def k_ser(x_out: qd.types.ndarray(dtype=float, ndim=1), __n0: int):
    qd.loop_config(serialize=True)
    for __t in range(__n0):
        x_out[__t] = 1.0

x = zeros((2,), dtype=float)
for k in (k_par, k_ser):
    k(x, 2)
qd.sync()

N = 3000
for name, k in (('parallel', k_par), ('serial', k_ser)):
    KMOD._SEG_ACC.clear()
    t0 = time.perf_counter()
    for _ in range(N):
        k(x, 2)
    qd.sync()
    wall = (time.perf_counter() - t0) / N * 1e6
    cpp = KMOD._SEG_ACC.get('cpp', 0.0) / N * 1e6
    print(f'{name:9s} wall {wall:6.2f} us  cpp {cpp:6.2f} us  python {wall - cpp:5.2f} us')
