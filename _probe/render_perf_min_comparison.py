"""render-perf: min-of-N sort-drain comparison, noise-robust re-measurement.

Three bvh variants (pristine 448-sync / nosync 3-sync / radix composable),
each measured with min-of-9 sync_drain at the 64envs shadows=on cell,
interleaved 3 rounds. Min statistics are robust to GPU contention spikes.

Usage: python render_perf_min_comparison.py   (generates variants itself)
"""

import os
import shutil
import subprocess
import sys

SP_BVH = "/tmp/a53b_venv/lib/python3.12/site-packages/quadrants/bvh.py"
PY = "/Users/mini0/Code/mujoqo/_port.venv/bin/python"
PROBE = "/Users/mini0/Code/quadrants-rpwt/_probe/render_perf_drainsplit.py"
CWD = "/Users/mini0/Code/mujoqo-metal"
PRISTINE = "/tmp/bvh_pristine.py"


def make_nosync():
    s = open(PRISTINE).read()
    old = """                _bvh_sort_stage(self.keys, self.primitive_indices, n, p, k, invocations)
                qd.sync()
                k = int(k / 2)"""
    new = """                _bvh_sort_stage(self.keys, self.primitive_indices, n, p, k, invocations)
                k = int(k / 2)"""
    assert old in s
    open("/tmp/bvh_nosync.py", "w").write(s.replace(old, new))


def make_radix():
    s = open(PRISTINE).read()
    old = """        qd.sync()
        # Batcher odd-even merge schedule (quadrants parallel_sort form)
        p = 1
        while p < n:
            k = p
            while k >= 1:
                invocations = int((n - k - k % p) / (2 * k)) + 1
                _bvh_sort_stage(self.keys, self.primitive_indices, n, p, k, invocations)
                qd.sync()
                k = int(k / 2)
            p = int(p * 2)"""
    new = """        _bvh_sort_radix(self.keys, self._tmp_keys, self.primitive_indices,
                        self._tmp_values, self._scratch, self._sort_n)"""
    assert old in s, "pristine must have the per-stage-sync Batcher form for this patch"
    s = s.replace(old, new)
    # add radix workspace in __init__ + kernel + imports (minimal forms)
    old_init = """        self.root = _bvh_zeros(1, qd.i32, np.int32)"""
    new_init = """        self._tmp_keys = _bvh_zeros(buf_n, qd.i32, np.int32)
        self._tmp_values = _bvh_zeros(buf_n, qd.i32, np.int32)
        self._scratch = _bvh_zeros(_sort_scratch_api(buf_n, 3), qd.u32, np.uint32)
        self._sort_n = qd.ndarray(dtype=qd.i32, shape=())
        self._sort_n.from_numpy(np.asarray(n, dtype=np.int32))
        self.root = _bvh_zeros(1, qd.i32, np.int32)"""
    assert old_init in s
    s = s.replace(old_init, new_init)
    old_kernel = '''@qd.kernel
def _bvh_sort_stage(keys: _i32_1d, values: _i32_1d, n: int, p: int, k: int, invocations: int):'''
    new_kernel = '''@qd.kernel
def _bvh_sort_radix(
    keys: _i32_1d,
    tmp_keys: _i32_1d,
    values: _i32_1d,
    tmp_values: _i32_1d,
    scratch: qd.types.ndarray(dtype=qd.u32, ndim=1),
    n: _i32_0d,
):
    _radix_sort_api(keys, tmp_keys, values, tmp_values, scratch, n, i32, True, 32, 3)


@qd.kernel
def _bvh_sort_stage(keys: _i32_1d, values: _i32_1d, n: int, p: int, k: int, invocations: int):'''
    assert old_kernel in s
    s = s.replace(old_kernel, new_kernel)
    old_imp = "import quadrants as qd\n"
    new_imp = "import quadrants as qd\nfrom quadrants.algorithms import sort as _radix_sort_api\nfrom quadrants.algorithms import sort_scratch_slots as _sort_scratch_api\nfrom quadrants.types.primitive_types import i32 as _i32_t\n"
    assert old_imp in s
    s = s.replace(old_imp, new_imp)
    # fix template dtype reference: use the imported alias
    s = s.replace("scratch, n, i32, True, 32, 3)", "scratch, n, _i32_t, True, 32, 3)")
    open("/tmp/bvh_radix.py", "w").write(s)


VARIANTS = {
    "pristine(448 sync)": PRISTINE,
    "nosync(3 sync)": "/tmp/bvh_nosync.py",
    "radix(chain)": "/tmp/bvh_radix.py",
}


def run_min(variant, path):
    shutil.copy(path, SP_BVH)
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    env["QD_ARCH"] = "metal"
    proc = subprocess.run([PY, PROBE], capture_output=True, text=True, env=env, cwd=CWD, timeout=900)
    for line in proc.stdout.splitlines():
        if line.startswith("DRAIN"):
            return line
    return f"NO OUTPUT rc={proc.returncode} {proc.stderr[-200:]}"


def main():
    make_nosync()
    make_radix()
    # compile warm for radix first (first run includes kernel compilation)
    run_min("radix warmup", VARIANTS["radix(chain)"])
    for rnd in range(2):
        for tag, path in VARIANTS.items():
            print(f"r{rnd} {tag}: {run_min(tag, path)}", flush=True)


if __name__ == "__main__":
    main()
