"""Standalone cProfile of one host function for a scenario.
Usage: fn_profile.py <scenario> <module> <func>   e.g. S5_cloth passive passive
"""
import sys, cProfile, pstats, io, importlib
sys.path.insert(0, '/tmp/c1site')
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf')
import quadrants as qd
qd.init(arch=qd.cpu, offline_cache=True, cfg_optimization=False)
import numpy as np
from mujoco_qd import test_data
import mujoco_qd as mjw

scenario, modname, fname = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf/benchmark3way')
from scenarios import SCENARIOS
xml = SCENARIOS[scenario]['xml']
q0 = SCENARIOS[scenario]['qpos0']
MOD = importlib.import_module(f'mujoco_qd._src.{modname}')
fn = getattr(MOD, fname)

mjm, mjd, m, d = test_data.fixture(xml=xml)
if q0 is not None:
    mjd.qpos[:] = np.array(q0)
d.qpos.from_numpy(np.tile(mjd.qpos.astype(np.float32), (1, 1)))
mjw.forward(m, d)
for _ in range(10):
    mjw.step(m, d)
qd.sync()

import time
t0 = time.perf_counter()
for _ in range(200):
    fn(m, d)
qd.sync()
print(f'{modname}.{fname} plain: {(time.perf_counter()-t0)/200*1000:.3f} ms/call')

pr = cProfile.Profile()
pr.enable()
for _ in range(200):
    fn(m, d)
pr.disable()
qd.sync()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats(16)
for line in s.getvalue().splitlines():
    print(line.replace('/Users/mini0/Code/mujoqo-perf/', '').replace('/tmp/c1site/quadrants/', '')[:135])
