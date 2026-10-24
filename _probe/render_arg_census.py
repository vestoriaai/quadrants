"""Count struct-ndarray params among the render megakernel's args (fast-lane
eligibility census, render-perf spec step B follow-up).

Usage (from mujoqo worktree root):
    PYTHONPATH=. QD_ARCH=metal _port.venv/bin/python _probe/render_arg_census.py
"""

import os

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

primal = rc._megakernel._primal
key = next(iter(primal._struct_ndarray_launch_info_by_key))
info = primal._struct_ndarray_launch_info_by_key[key]
print(f"total args: {len(primal.arg_metas)}")
print(f"struct_ndarray_launch_info entries: {len(info)}")
for e in info:
    print("  ", e)

# map arg id -> name via arg_metas if the shape allows
metas = primal.arg_metas
try:
    names = {}
    for i, meta in enumerate(metas):
        names[i] = getattr(meta, "name", None) or str(meta)
    print("=== arg table ===")
    for i, meta in enumerate(metas):
        print(f"  [{i}] {names[i]}")
except Exception as e:  # noqa: BLE001
    print("arg_metas walk failed:", repr(e))
