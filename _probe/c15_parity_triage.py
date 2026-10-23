"""C1.5 parity triage: serial-vs-parallel divergence vs semantic break.

1. AS=0 1000-step: must match dev134 baseline bitwise (regression check)
2. AS=1 vs AS=0: first divergence step (chaos point) — early steps must match
"""
import os, sys
sys.path.insert(0, '/tmp/c1site')
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf')
mode = os.environ.get('AS', '1')
os.environ['QD_AUTO_SERIAL'] = mode
import quadrants as qd
qd.init(arch=qd.cpu, offline_cache=True, cfg_optimization=False)
import numpy as np
from mujoco_qd import test_data
import mujoco_qd as mjw

xml = open('/Users/mini0/Code/mujoqo-perf/mujoco_qd/test_data/humanoid/humanoid.xml').read()
mjm, mjd, m, d = test_data.fixture(xml=xml)
mjd.qpos[:] = mjm.key_qpos[0]
d.qpos.from_numpy(np.tile(mjd.qpos.astype(np.float32), (1, 1)))
mjw.forward(m, d)
qpos_log = [d.qpos.to_numpy().copy()]
for step in range(1000):
    mjw.step(m, d)
    qpos_log.append(d.qpos.to_numpy().copy())
np.save(f'/tmp/c15_traj_{mode}.npy', np.array(qpos_log))
print(f'AS={mode}: done, final qvel_norm={np.linalg.norm(d.qvel.to_numpy()):.4f}')
