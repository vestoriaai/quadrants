"""Which passive launches hit the C1a lane vs fall through?"""
import sys, time
sys.path.insert(0, '/tmp/c1site')
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf')
import quadrants as qd
qd.init(arch=qd.cpu, offline_cache=True, cfg_optimization=False)
import numpy as np
from mujoco_qd import test_data
import mujoco_qd as mjw
import mujoco_qd._src.passive as PAS
import quadrants.lang.kernel_impl as K
KMOD = sys.modules['quadrants.lang.kernel']
from collections import Counter

xml = SCEN = None
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf/benchmark3way')
from scenarios import SCENARIOS
xml = SCENARIOS['S5_cloth']['xml']

mjm, mjd, m, d = test_data.fixture(xml=xml)
d.qpos.from_numpy(np.tile(mjd.qpos.astype(np.float32), (1, 1)))
mjw.forward(m, d)
for _ in range(10):
    mjw.step(m, d)
qd.sync()

# wrap ensure_compiled: fast-lane hits skip it entirely
ec = Counter()
_ensure = K.Kernel.ensure_compiled
def ec_spy(self, *a):
    ec['called'] += 1
    return _ensure(self, *a)
K.Kernel.ensure_compiled = ec_spy

fn = PAS.passive
# warm: first calls populate lanes
for _ in range(3):
    fn(m, d)
qd.sync()

lanes = {}
ec.clear()
t0 = time.perf_counter()
for _ in range(200):
    fn(m, d)
qd.sync()
wall = (time.perf_counter() - t0) / 200 * 1000

# inspect all kernels' lanes
import gc
for obj in gc.get_objects():
    if isinstance(obj, K.Kernel):
        lane = getattr(obj, '_c1_lane', None)
        if lane is not None and len(lane):
            name = getattr(obj, 'name', None) or (obj.func.__name__ if getattr(obj, 'func', None) else '?')
            lanes[name] = len(lane)
print(f'passive: {wall:.3f} ms/call; ensure_compiled calls/200: {ec["called"]} ({ec["called"]/200:.1f}/call)')
print('non-empty lanes:', lanes if lanes else 'NONE')
