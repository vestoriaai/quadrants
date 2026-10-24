"""render-perf spec step B: cProfile the render() host path at the bench shape
to find where submit 3.18ms actually goes (guard chain vs launch-context
assembly vs pybind launch).

Usage (from mujoqo worktree root):
    PYTHONPATH=. QD_ARCH=metal _port.venv/bin/python _probe/render_perf_cprofile.py
"""

import cProfile
import os
import pstats
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
mjw.render(m, d, rc)
_ = rc.rgb_data.to_numpy()

prof = cProfile.Profile()
prof.enable()
for _ in range(6):
    mjw.render(m, d, rc)
    rc.rgb_data.to_numpy()
prof.disable()

stats = pstats.Stats(prof)
stats.sort_stats("cumulative")
print("===== top 22 by cumulative =====")
stats.print_stats(22)
