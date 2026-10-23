"""S0a v2: clean line-level attribution — Python-function wraps only, no prog shim.
C++ segment known from prior clean probe: ~2us of the 15.6us trivial floor.
"""
import sys, time
sys.path.insert(0, '/Users/mini0/Code/quadrants-c1wt/python')
import quadrants as qd
qd.init(arch=qd.cpu, offline_cache=False, cfg_optimization=False)
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf')
from mujoco_qd._src.warp_util import zeros

stats = {}
def wrap(cls, methname):
    orig = getattr(cls, methname)
    key = f'{cls.__name__}.{methname}'
    def timed(self, *a, **k):
        t0 = time.perf_counter()
        r = orig(self, *a, **k)
        s = stats.setdefault(key, [0, 0.0])
        s[0] += 1
        s[1] += time.perf_counter() - t0
        return r
    setattr(cls, methname, timed)
    return orig

import quadrants.lang.kernel_impl as K
import sys as _sys
KMOD = _sys.modules['quadrants.lang.kernel']
import quadrants.lang._template_mapper as TM
wrap(K.Kernel, '__call__') if hasattr(K, 'Kernel') else None
wrap(K.Kernel, 'ensure_compiled') if hasattr(K, 'Kernel') else None
wrap(K.Kernel, 'launch_kernel') if hasattr(K, 'Kernel') else None
wrap(KMOD.LaunchContextBufferCache, 'populate_launch_ctx_from_cache')
wrap(TM.TemplateMapper, 'lookup')

@qd.kernel
def trivial(x_out: qd.types.ndarray(dtype=float, ndim=1), __n0: int):
    for __t in range(__n0):
        x_out[__t] = 1.0

x = zeros((2,), dtype=float)
trivial(x, 2)
qd.sync()
stats.clear()

N = 3000
t0 = time.perf_counter()
for _ in range(N):
    trivial(x, 2)
_impl_wall = time.perf_counter() - t0
qd.sync()
wall = _impl_wall / N * 1e6

cpp_us = 2.0  # from clean prior probe (mujoqo-perf _probe/perf_c0_cpp_split: C++ ~2us for trivial)
print(f'wall: {wall:.2f} us/launch (N={N}, wrapped; unwrapped baseline 15.6)')
for k, (c, t) in sorted(stats.items(), key=lambda kv: -kv[1][1]):
    print(f'  {k:46s} cum {t/N*1e6:>6.2f} us/launch')
lk = stats.get('Kernel.launch_kernel', [0, 0.0])[1] / N * 1e6
pop = stats.get('LaunchContextBufferCache.populate_launch_ctx_from_cache', [0, 0.0])[1] / N * 1e6
lk_own = lk - pop - cpp_us
print(f'  launch_kernel own-body (excl populate+cpp): {lk_own:.2f} us')
print(f'  __call__ rest (excl launch_kernel+ensure_compiled): {stats.get("Kernel.__call__",[0,0])[1]/N*1e6 - stats.get("Kernel.launch_kernel",[0,0])[1]/N*1e6 - stats.get("Kernel.ensure_compiled",[0,0])[1]/N*1e6:.2f} us')
