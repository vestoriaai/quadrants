"""A56 gc referrer probe: who pins the PyQuadrants runtime at interpreter shutdown?

Usage:
    python a56_gc_refprobe.py <cache_dir>

Runs the checkpoint kernel once (no explicit clear), registers an atexit
diagnostic that gc.collect()s and prints the referrer histogram of the runtime
singleton plus the pinning chain for non-module referrers. Compare output
between the torch-installed venv and the torch-free venv.
"""

import atexit
import gc
import sys
import types

CACHE_DIR = sys.argv[1]

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


def _describe_referrer(r, rt):
    """Return a short string locating rt inside referrer object r."""
    tn = type(r).__name__
    try:
        if isinstance(r, dict):
            keys = [k for k, v in r.items() if v is rt]
            mod = r.get("__name__", "")
            return f"dict<{mod}>" + (f" keys={keys[:4]}" if keys else "")
        if isinstance(r, (list, tuple)):
            idxs = [i for i, v in enumerate(r) if v is rt]
            return f"{tn} idx={idxs[:4]}"
        if isinstance(r, types.FunctionType):
            return f"function {r.__module__}.{r.__qualname__}"
        if isinstance(r, types.FrameType):
            return f"frame {r.f_code.co_filename.split('/')[-1]}:{r.f_lineno} {r.f_code.co_name}"
        if hasattr(r, "__dict__"):
            attrs = [k for k, v in vars(r).items() if v is rt]
            return f"{tn}.{attrs[:4]}" if attrs else tn
    except Exception as e:  # noqa: BLE001
        return f"{tn}(<{e!r}>)"
    return tn


def diagnose():
    from quadrants.lang import impl

    rt = getattr(impl, "pyquadrants", None)
    print(f"[refprobe] runtime alive at atexit: {rt is not None}")
    if rt is None:
        return
    print(f"[refprobe] _prog set: {getattr(rt, '_prog', None) is not None}")
    n = gc.collect()
    print(f"[refprobe] gc.collect freed {n}")
    rt2 = getattr(impl, "pyquadrants", None)
    print(f"[refprobe] runtime alive after collect: {rt2 is not None}")
    if rt2 is None:
        return
    referrers = gc.get_referrers(rt2)
    hist = {}
    for r in referrers:
        k = type(r).__name__
        hist[k] = hist.get(k, 0) + 1
    print(f"[refprobe] {len(referrers)} referrers: {sorted(hist.items(), key=lambda kv: -kv[1])}")
    shown = 0
    for r in referrers:
        desc = _describe_referrer(r, rt2)
        if "dict<" in desc and ("impl" in desc or "builtins" in desc or "sys" in desc):
            continue
        print(f"[refprobe]   - {desc}")
        shown += 1
        if shown >= 25:
            break


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
    print("PROBE_OK")
    atexit.register(diagnose)


main()
