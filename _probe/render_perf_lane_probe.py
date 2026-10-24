"""render-perf spec step B: does the render megakernel launch hit the C1a
fast lane, and where does submit 3.18ms go?

C1DBG2-style instrumentation is already built into kernel.py; this probe turns
it on, renders at the bench shape, and reports lane hit/miss per launch plus
the submit-time split. Run from the mujoqo worktree root:

    C1DBG=1 C1DBG2=1 PYTHONPATH=. QD_ARCH=metal \
        _port.venv/bin/python _probe/render_perf_lane_probe.py 2>&1 | grep DBG
"""

import os
import sys
import time

os.environ.setdefault("QD_ARCH", "metal")

from mujoco_qd._src.conftest import _init_qd  # noqa: E402

_init_qd()

import mujoco_qd as mjw  # noqa: E402
from mujoco_qd import test_data  # noqa: E402

mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=64)
rc = mjw.create_render_context(
    mjm, nworld=64, cam_res=(84, 84), render_rgb=True, use_shadows=True,
)

# warm: first render compiles + fills lane on the slow path
mjw.render(m, d, rc)
_ = rc.rgb_data.to_numpy()

for i in range(3):
    t0 = time.perf_counter()
    mjw.render(m, d, rc)
    t1 = time.perf_counter()
    print(f"[PROBE] render #{i} submit={1000 * (t1 - t0):.2f}ms", flush=True)

print("[PROBE] done - DBG lines above show lane gate PASS/HIT per kernel launch")
