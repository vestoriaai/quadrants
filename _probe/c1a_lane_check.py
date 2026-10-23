"""C1a lane fill verification."""
import sys
sys.path.insert(0, '/tmp/c1site')
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf')
import quadrants as qd
qd.init(arch=qd.cpu, offline_cache=False, cfg_optimization=False)
from mujoco_qd._src.warp_util import zeros
import timeit

x = zeros((27,), dtype=float); y = zeros((27,), dtype=float)

@qd.kernel
def k(a_in: qd.types.ndarray(dtype=float, ndim=1), b_out: qd.types.ndarray(dtype=float, ndim=1), __n0: int):
    for __t in range(__n0):
        b_out[__t] = a_in[__t] * 2.0 + 1.0

x.fill(3.0)
k(x, y, 27); k(x, y, 27); k(x, y, 27)
qd.sync()
kc = k._primal if hasattr(k, '_primal') else k
lane = getattr(kc, '_c1_lane', 'MISSING')
print('lane type:', type(lane).__name__ if lane is not None else 'None')
print('lane entries:', len(lane) if isinstance(lane, dict) else lane)
print('mutable_cached_val:', getattr(kc, '_mutable_nd_cached_val', '?'))
print('tensor_indices:', getattr(kc, '_tensor_unwrap_indices', '?'))
N = 3000
t = timeit.timeit(lambda: k(x, y, 27), number=N)
print(f'launch: {t*1e6/N:.2f} us')
qd.sync()
