"""Post-C1.5 structure probe: work-size distribution of S6's 133 launches.

Determines how many launches still exceed the serial threshold (64) and pay
the 13us barrier, vs how many are now serial-fast — this sizes C1b/C2 gains.
"""
import sys, collections
sys.path.insert(0, '/tmp/c1site')
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf')
import quadrants as qd
qd.init(arch=qd.cpu, offline_cache=True, cfg_optimization=False)
import numpy as np
from mujoco_qd import test_data
import mujoco_qd as mjw
from quadrants.lang.kernel_impl import Kernel

xml = open('/Users/mini0/Code/mujoqo-perf/mujoco_qd/test_data/humanoid/humanoid.xml').read()
mjm, mjd, m, d = test_data.fixture(xml=xml)
mjd.qpos[:] = mjm.key_qpos[0]
d.qpos.from_numpy(np.tile(mjd.qpos.astype(np.float32), (1, 1)))
mjw.forward(m, d)
for _ in range(3):
    mjw.step(m, d)

dist = collections.Counter()
orig = Kernel.__call__
def spy(self, *a, **k):
    # __n0 trailing scalar args = grid sizes; work = product
    work = 1
    for v in reversed(a):
        if isinstance(v, int):
            work = v
            break
        break
    bucket = ('<=64' if work <= 64 else '65-256' if work <= 256 else '257-1024' if work <= 1024 else '>1024')
    dist[bucket] += 1
    return orig(self, *a, **k)
Kernel.__call__ = spy

N = 30
for _ in range(N):
    mjw.step(m, d)
qd.sync()
total = sum(dist.values())
print(f'{total/N:.0f} launches/step; work-size distribution (last int arg = n0):')
for b in ('<=64', '65-256', '257-1024', '>1024'):
    print(f'  {b:10s} {dist[b]/N:>6.1f}/step')
over = sum(dist[b] for b in ('65-256', '257-1024', '>1024')) / N
print(f'over-threshold(>64): {over:.1f}/step -> residual barrier ~{over*13:.0f} us/step (if parallel)')
