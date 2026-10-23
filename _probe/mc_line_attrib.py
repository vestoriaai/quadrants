"""Line-level attribution inside make_constraint (temporary source patch)."""
import sys, time
sys.path.insert(0, '/tmp/c1site')
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf')
import quadrants as qd
qd.init(arch=qd.cpu, offline_cache=True, cfg_optimization=False)
import numpy as np
from mujoco_qd import test_data
import mujoco_qd as mjw
import mujoco_qd._src.constraint as CONS
import time as T

# 临时行级: 用源码读取获取函数体并按语句块计时不可行; 改为统计其内部 kernel 调用与 host 段
from quadrants.lang.kernel_impl import Kernel
kstats = {}
korig = Kernel.__call__
def kspy(self, *a, **k):
    name = ''
    for holder, attr in ((self, 'name'), (self, 'func')):
        try:
            v = getattr(holder, attr)
            if isinstance(v, str) and v and v != 'kernel':
                name = v
                break
            if callable(v) and getattr(v, '__name__', 'kernel') != 'kernel':
                name = v.__name__
                break
        except Exception:
            pass
    t0 = T.perf_counter()
    r = korig(self, *a, **k)
    s = kstats.setdefault(name, [0, 0.0])
    s[0] += 1
    s[1] += T.perf_counter() - t0
    return r
Kernel.__call__ = kspy

orig_mc = CONS.make_constraint
def mc_timed(m, d):
    t0 = T.perf_counter()
    orig_mc(m, d)
    s = kstats.setdefault('__MAKE_CONSTRAINT_WALL__', [0, 0.0])
    s[0] += 1
    s[1] += T.perf_counter() - t0
CONS.make_constraint = mc_timed

xml = open('/Users/mini0/Code/mujoqo-perf/mujoco_qd/test_data/humanoid/humanoid.xml').read()
mjm, mjd, m, d = test_data.fixture(xml=xml)
mjd.qpos[:] = mjm.key_qpos[0]
d.qpos.from_numpy(np.tile(mjd.qpos.astype(np.float32), (1, 1)))
mjw.forward(m, d)
for _ in range(5):
    mjw.step(m, d)
kstats.clear()

N = 100
for _ in range(N):
    mjw.step(m, d)
qd.sync()
wall = kstats.get('__MAKE_CONSTRAINT_WALL__', [0, 0.0])
print(f"make_constraint wall: {wall[1]/N*1000:.3f} ms/step")
launch_python = 0.0
for name, (c, t) in sorted(kstats.items(), key=lambda kv: -kv[1][1]):
    if name == '__MAKE_CONSTRAINT_WALL__':
        continue
    us = t / N * 1e6
    launch_python += us * c / N
    print(f'  {name[:44]:44s} {c/N:>6.1f}/step {us:>8.2f} us/call {t/N*1000:>8.3f} ms')
print(f'  kernel-call python total ≈ {launch_python/1000:.3f} ms; host-only ≈ {wall[1]/N*1000 - launch_python/1000:.3f} ms')
