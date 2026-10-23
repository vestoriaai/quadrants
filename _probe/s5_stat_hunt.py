"""Who calls posix.stat in the S5 hot path?"""
import sys, traceback, collections
sys.path.insert(0, '/tmp/c1site')
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf')
import posix
_orig = posix.stat
sites = collections.Counter()
armed = {'n': 0}
def spy(path, *a, **k):
    if armed['n'] < 8:
        f = sys._getframe(1)
        chain = []
        while f is not None and len(chain) < 5:
            chain.append(f'{f.f_code.co_filename.split("/")[-1]}:{f.f_lineno}:{f.f_code.co_name}')
            f = f.f_back
        sites[' <- '.join(reversed(chain))] += 1
        armed['n'] += 1
    return _orig(path, *a, **k)
posix.stat = spy
import os
os.stat = spy

import quadrants as qd
qd.init(arch=qd.cpu, offline_cache=True, cfg_optimization=False)
import numpy as np
from mujoco_qd import test_data
import mujoco_qd as mjw
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf/benchmark3way')
from scenarios import SCENARIOS
xml = SCENARIOS['S5_cloth']['xml']
mjm, mjd, m, d = test_data.fixture(xml=xml)
d.qpos.from_numpy(np.tile(mjd.qpos.astype(np.float32), (1, 1)))
mjw.forward(m, d)
for _ in range(10):
    mjw.step(m, d)
qd.sync()
armed['n'] = 0
sites.clear()
for _ in range(20):
    mjw.step(m, d)
qd.sync()
for chain, n in sites.most_common(5):
    print(f'{n:>4}  {chain[:150]}')
