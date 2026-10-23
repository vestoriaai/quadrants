"""A56 shutdown-path probe: does explicit runtime clear() make the C++ dump happen?

Usage:
    python a56_shutdown_probe.py <cache_dir> <mode>

Modes:
    plain      - return from main() normally (baseline, known to not dump)
    exit       - sys.exit(0) at the end
    clear      - get_runtime().clear() at the end, then return
    clear_exit - clear() then sys.exit(0)
"""

import sys
import os

CACHE_DIR = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "plain"

import numpy as np  # noqa: E402
import quadrants as qd  # noqa: E402


@qd.data_oriented
class _FastcacheYieldOnSelfCheckpoint:
    def __init__(self, n):
        self.x = qd.ndarray(qd.i32, shape=(n,))
        self.flag = qd.ndarray(qd.i32, shape=())

    @qd.kernel(graph=True, checkpoints=True, fastcache=True)
    def step(self):
        with qd.checkpoint(0, yield_on=self.flag):
            for i in range(self.x.shape[0]):
                self.x[i] = self.x[i] + 1


def main():
    qd.init(
        arch=qd.arm64,
        offline_cache=True,
        offline_cache_file_path=CACHE_DIR,
        src_ll_cache=True,
    )
    n = 8
    sim = _FastcacheYieldOnSelfCheckpoint(n)
    sim.x.from_numpy(np.zeros(n, dtype=np.int32))
    sim.flag.from_numpy(np.array(0, dtype=np.int32))
    sim.step()
    x = sim.x.to_numpy()
    assert (x == 1).all(), f"kernel result wrong: {x}"
    primal = type(sim).step._primal
    obs = primal.src_ll_cache_observations
    print(f"PROBE_OK mode={MODE} loaded={obs.cache_loaded}")
    if "clear" in MODE:
        qd.lang.impl.get_runtime().clear()
        print("CLEARED")
    if "exit" in MODE:
        sys.exit(0)


main()
