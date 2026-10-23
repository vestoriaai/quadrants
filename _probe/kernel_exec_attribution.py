"""Kernel compute attribution: per-kernel C++ exec time on C1.5 wheel (serial path)."""
import sys
sys.path.insert(0, '/tmp/c1site')
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf')
import quadrants as qd
qd.init(arch=qd.cpu, offline_cache=True, cfg_optimization=False)
import numpy as np
from mujoco_qd import test_data
import mujoco_qd as mjw
import quadrants.lang.kernel as KMOD

xml = open('/Users/mini0/Code/mujoqo-perf/mujoco_qd/test_data/humanoid/humanoid.xml').read()
mjm, mjd, m, d = test_data.fixture(xml=xml)
mjd.qpos[:] = mjm.key_qpos[0]
d.qpos.from_numpy(np.tile(mjd.qpos.astype(np.float32), (1, 1)))
mjw.forward(m, d)
for _ in range(3):
    mjw.step(m, d)

vars(sys.modules['quadrants.lang.kernel'])['_C1_SEG'].clear()
N = 50
for _ in range(N):
    mjw.step(m, d)
qd.sync()

rows = sorted(vars(sys.modules['quadrants.lang.kernel'])['_C1_SEG'].items(), key=lambda kv: -kv[1][1])
tot = sum(t for _, (c, t) in rows) / N * 1000
print(f'TOTAL kernel exec (C++ seg): {tot:.3f} ms/step across {sum(entry[0] for _, entry in rows)/N:.0f} launches')
print(f"{'kernel':46s} {'calls/step':>10} {'us/exec':>8} {'ms/step':>8}")
for name, entry in rows[:22]:
    cnt, t = entry
    print(f'{name[:46]:46s} {cnt/N:>10.1f} {t/cnt*1e6:>8.2f} {t/N*1000:>8.3f}')
