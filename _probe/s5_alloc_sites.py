"""C0d: attribute per-step ndarray allocations to call sites."""
import sys, collections
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf')
import quadrants as qd
qd.init(arch=qd.cpu, offline_cache=True, cfg_optimization=False)

from quadrants.lang._ndarray import ScalarNdarray
sites = collections.Counter()
_orig_init = ScalarNdarray.__init__
def tracked_init(self, *a, **k):
    f = sys._getframe(2)
    chain = []
    while f is not None and len(chain) < 4:
        name = f.f_code.co_name
        fn = f.f_code.co_filename.split('/')[-1]
        if 'quadrants' not in fn:
            chain.append(f'{fn}:{name}:{f.f_lineno}')
        f = f.f_back
    sites[' <- '.join(chain) or 'unknown'] += 1
    return _orig_init(self, *a, **k)
ScalarNdarray.__init__ = tracked_init

import numpy as np
from mujoco_qd import test_data
import mujoco_qd as mjw

sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf/benchmark3way')
from scenarios import SCENARIOS
xml = SCENARIOS['S5_cloth']['xml']
mjm, mjd, m, d = test_data.fixture(xml=xml)
d.qpos.from_numpy(np.tile(mjd.qpos.astype(np.float32), (1, 1)))
mjw.forward(m, d)
for _ in range(5):
    mjw.step(m, d)

sites.clear()
N = 30
for _ in range(N):
    mjw.step(m, d)
qd.sync()
for site, n in sites.most_common(12):
    print(f'{n/N:5.1f}/step  {site[:110]}')
