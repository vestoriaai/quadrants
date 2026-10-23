"""Per-kernel exec attribution for any scenario. Usage: kernel_exec_attribution2.py <scenario> [steps_per_sample]"""
import sys
sys.path.insert(0, '/tmp/c1site')
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf')
import quadrants as qd
qd.init(arch=qd.cpu, offline_cache=True, cfg_optimization=False)
import numpy as np
from mujoco_qd import test_data
import mujoco_qd as mjw
import quadrants.lang.kernel as KMOD

scenario = sys.argv[1] if len(sys.argv) > 1 else 'S6_humanoid'
CFG = {
  'S5_cloth': None,
  'S6_humanoid': None,
  'S3_slide_spring': None,
}
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf/benchmark3way')
from scenarios import SCENARIOS
xml = SCENARIOS[scenario]['xml']
qpos0 = SCENARIOS[scenario]['qpos0']

mjm, mjd, m, d = test_data.fixture(xml=xml)
if qpos0 is not None:
    mjd.qpos[:] = np.array(qpos0)
d.qpos.from_numpy(np.tile(mjd.qpos.astype(np.float32), (1, 1)))
mjw.forward(m, d)
for _ in range(3):
    mjw.step(m, d)

vars(sys.modules['quadrants.lang.kernel'])['_C1_SEG'].clear()
N = 50
t0 = time.perf_counter() if False else None
import time
t0 = time.perf_counter()
for _ in range(N):
    mjw.step(m, d)
qd.sync()
wall = (time.perf_counter() - t0) / N * 1000

rows = sorted(vars(sys.modules['quadrants.lang.kernel'])['_C1_SEG'].items(), key=lambda kv: -kv[1][1])
tot = sum(entry[1] for _, entry in rows) / N * 1000
n_launch = sum(entry[0] for _, entry in rows) / N
print(f'{scenario}: wall {wall:.3f} ms/step | kernel exec {tot:.3f} ms | {n_launch:.0f} launches')
print(f"{'kernel':46s} {'calls/step':>10} {'us/exec':>8} {'ms/step':>8}")
for name, entry in rows[:18]:
    cnt, t = entry
    print(f'{name[:46]:46s} {cnt/N:>10.1f} {t/cnt*1e6:>8.2f} {t/N*1000:>8.3f}')
