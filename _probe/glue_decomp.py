"""Glue decomposition per scenario. Usage: glue_decomp.py [scenario]"""
import sys
sys.path.insert(0, '/tmp/c1site')
sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf')
import quadrants as qd
qd.init(arch=qd.cpu, offline_cache=True, cfg_optimization=False)
import numpy as np
from mujoco_qd import test_data
import mujoco_qd as mjw
import time

sys.path.insert(0, '/Users/mini0/Code/mujoqo-perf/benchmark3way')
from scenarios import SCENARIOS
scenario = sys.argv[1] if len(sys.argv) > 1 else 'S6_humanoid'
xml = SCENARIOS[scenario]['xml']
q0 = SCENARIOS[scenario]['qpos0']

import mujoco_qd._src.forward as FWD
import mujoco_qd._src.constraint as CONS
import mujoco_qd._src.sensor as SEN
import mujoco_qd._src.smooth as SM
import mujoco_qd._src.collision_driver as CD
import mujoco_qd._src.collision_flex as CFLEX
import mujoco_qd._src.collision_gjk as GJK
import mujoco_qd._src.passive as PAS
import mujoco_qd._src.support as SUP

stats = {}
def wrap(mod, name):
    if not hasattr(mod, name):
        return
    fn = getattr(mod, name)
    key = f'{mod.__name__.split(".")[-1]}.{name}'
    def timed(*a, **k):
        t0 = time.perf_counter()
        r = fn(*a, **k)
        s = stats.setdefault(key, [0, 0.0])
        s[0] += 1
        s[1] += time.perf_counter() - t0
        return r
    setattr(mod, name, timed)

for mod, names in (
    (FWD, ('fwd_position', 'fwd_velocity', 'fwd_actuation', '_energy_pos', 'fwd_acceleration')),
    (CONS, ('make_constraint',)),
    (SEN, ('sensor_pos', 'sensor_vel', 'sensor_acc')),
    (SM, ('factor_solve_i', 'solve_m')),
    (CD, ('collision', 'nxn_broadphase')),
    (CFLEX, ('collide', 'collide_flex')),
    (GJK, ('collide',)),
    (PAS, ('passive', 'flex_hessian', 'flex_stretch_mul', 'build_efm_contact')),
    (SUP, ('apply_ft',)),
    (SM, ('com_vel', 'rne', 'tendon_bias')),
):
    for n in names:
        wrap(mod, n)

mjm, mjd, m, d = test_data.fixture(xml=xml)
if q0 is not None:
    mjd.qpos[:] = np.array(q0)
d.qpos.from_numpy(np.tile(mjd.qpos.astype(np.float32), (1, 1)))
mjw.forward(m, d)
for _ in range(10):
    mjw.step(m, d)
qd.sync()
stats.clear()

N = 100
t0 = time.perf_counter()
for _ in range(N):
    mjw.step(m, d)
qd.sync()
wall = (time.perf_counter() - t0) / N * 1000
print(f'{scenario}: step wall {wall:.3f} ms')
for k, (c, t) in sorted(stats.items(), key=lambda kv: -kv[1][1])[:14]:
    print(f'  {k:38s} {t/N*1000:>7.3f} ms/step')
