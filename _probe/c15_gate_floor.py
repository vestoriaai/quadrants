"""C1.5 gate: serial-fallback correctness + floor measurement (dev136 wheel)."""
import os
import sys
sys.path.insert(0, '/tmp/c1site')
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf')
mode = os.environ.get('AS', '1')
os.environ['QD_AUTO_SERIAL'] = mode
# env read happens at ThreadPool construction inside qd.init -> set before init
import quadrants as qd
qd.init(arch=qd.cpu, offline_cache=False, cfg_optimization=False)
import numpy as np

from mujoco_qd._src.warp_util import zeros
import timeit

x = zeros((27,), dtype=float)
y = zeros((27,), dtype=float)

@qd.kernel
def k(a_in: qd.types.ndarray(dtype=float, ndim=1), b_out: qd.types.ndarray(dtype=float, ndim=1), __n0: int):
    for __t in range(__n0):
        b_out[__t] = a_in[__t] * 2.0 + 1.0

x.fill(3.0)
k(x, y, 27)
qd.sync()
exp = x.to_numpy() * 2.0 + 1.0
print(f'QD_AUTO_SERIAL={mode}: correct={np.allclose(y.to_numpy(), exp)}')

N = 3000
t = timeit.timeit(lambda: k(x, y, 27), number=N)
print(f'27-elem launch+exec: {t*1e6/N:.2f} us')
qd.sync()
