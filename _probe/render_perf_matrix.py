"""render-perf spec step A: kernel-time isolation matrix for the 6.98ms GPU
account — {shadows on/off} x {nworld 1/64}, explicit-sync decomposition.

Usage (from mujoqo worktree root):
    PYTHONPATH=. QD_ARCH=metal _port.venv/bin/python _probe/render_perf_matrix.py
"""

import os
import time

os.environ.setdefault("QD_ARCH", "metal")

from mujoco_qd._src.conftest import _init_qd  # noqa: E402

_init_qd()

import mujoco_qd as mjw  # noqa: E402
from mujoco_qd import test_data  # noqa: E402
from quadrants.lang import impl as _impl  # noqa: E402


def measure(nworld, shadows, reps=10):
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=nworld)
    rc = mjw.create_render_context(
        mjm, nworld=nworld, cam_res=(84, 84), render_rgb=True, use_shadows=shadows,
    )
    for _ in range(3):
        mjw.render(m, d, rc)
        _ = rc.rgb_data.to_numpy()
    sub, sync = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        mjw.render(m, d, rc)
        t1 = time.perf_counter()
        _impl.get_runtime().sync()
        t2 = time.perf_counter()
        _ = rc.rgb_data.to_numpy()
        t3 = time.perf_counter()
        sub.append(t1 - t0)
        sync.append(t2 - t1)
    med = lambda v: sorted(v)[len(v) // 2] * 1000.0
    print(
        f"RESULT nworld={nworld:>2} shadows={int(shadows)}: "
        f"submit={med(sub):.2f}ms gpu_wait={med(sync):.2f}ms "
        f"kernel_account={med(sub) + med(sync):.2f}ms",
        flush=True,
    )


for nw in (1, 64):
    for sh in (False, True):
        measure(nw, sh)
