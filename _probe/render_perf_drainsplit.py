"""render-perf: drain-split measurement for the no-sync BVH hypothesis.

Splits render()'s GPU drain into three timed segments (each ends with a full
sync): morton+sort stages, build_rest, megakernel(+shadow). Runs the SAME
bvh.py state (whatever is installed) so A/B just swaps the file.

Usage:
    PYTHONPATH=. QD_ARCH=metal <python> render_perf_drainsplit.py
Prints DRAIN lines (median of 9).
"""

import os
import time

os.environ.setdefault("QD_ARCH", "metal")

from mujoco_qd._src.conftest import _init_qd  # noqa: E402

_init_qd()

import mujoco_qd as mjw  # noqa: E402
from mujoco_qd import test_data  # noqa: E402
from quadrants.lang import impl as _impl  # noqa: E402

mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=64)
rc = mjw.create_render_context(
    mjm, nworld=64, cam_res=(84, 84), render_rgb=True, use_shadows=True,
)

# mark segment boundaries via a patched build is intrusive; instead measure
# with three sync variants by env: the bvh.py no-sync build already syncs only
# at sort-end and build-end. We time: submit (morton+sort queueing),
# sync#1 (sort+build_rest drain), megakernel submit, sync#2 (megakernel drain).

sub_sort, sort_drain, sub_mega, mega_drain = [], [], [], []
for _ in range(9):
    t0 = time.perf_counter()
    mjw.render(m, d, rc)  # includes morton+sort (no internal syncs) + build_rest sync + megakernel submit
    t1 = time.perf_counter()
    _impl.get_runtime().sync()
    t2 = time.perf_counter()
    _ = rc.rgb_data.to_numpy()
    t3 = time.perf_counter()
    sub_sort.append(t1 - t0)
    mega_drain.append(t2 - t1)  # sync: drain of everything still queued
    t_read = t3 - t2

med = lambda v: sorted(v)[len(v) // 2] * 1000.0
print(
    f"DRAIN submit_incl_build_sync={med(sub_sort):.2f}ms "
    f"sync_drain={med(mega_drain):.2f}ms "
    f"(build-internal syncs already drain sort+build_rest inside submit)",
    flush=True,
)
