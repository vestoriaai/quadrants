"""Probe 3: one-arch-per-process — what exactly breaks on double qd.init / mixed arch?

Each scenario runs in its own process (CLI selects). Families are tiny coll_scan-like
classes (NS=8). Scenario matrix:
  base_cpu        init(cpu), launch family A                       (control, expect OK)
  base_metal      init(metal), launch family A                     (control, expect OK)
  abandon         init(cpu) A -> init(metal) B, launch B only      (is abandoning the old family safe?)
  reuse           init(cpu) A -> init(metal) B -> launch A again   (expected UB: dangling fields)
  reuse_metal_cpu mirror of `reuse` with metal first
  same_arch       init(cpu) A -> init(cpu) B -> launch A again     (double init, same arch)
  abandon_nd      init(cpu) A(ndarray-only kernel) -> init(metal) -> launch A again
                  (kernel survives reset; ndarray args only — recompiled under new program)
Prints stage markers with flush so crash points are exact; exit code carries the outcome.
"""
import argparse
import sys
import time

import numpy as np


def make_family(tag="A"):
    import quadrants as qd

    @qd.data_oriented
    class Fam:
        def __init__(self, n=8):
            self.NS = n
            self.s_rad = qd.field(dtype=qd.f32, shape=(n,))
            self.coef = qd.Vector.field(4, qd.f32, shape=(1,))
            self.s_rad.from_numpy(np.ones(n, np.float32) * 0.1)
            self.coef.from_numpy(np.array([[0.5, 0, 0, 0.01]], np.float32))

        @qd.kernel
        def scan(self, fr: qd.types.ndarray(ndim=1, dtype=qd.f32),
                 out: qd.types.ndarray(ndim=1, dtype=qd.f32), n: int):
            acc = 0.0
            for i in range(n):
                for s in qd.static(range(self.NS)):
                    acc += self.s_rad[s] * fr[0] * self.coef[0][0]
                out[i] = acc

        @qd.kernel
        def scan_nd(self, fr: qd.types.ndarray(ndim=1, dtype=qd.f32),
                    out: qd.types.ndarray(ndim=1, dtype=qd.f32), n: int):
            acc = 0.0
            for i in range(n):
                for s in qd.static(range(self.NS)):
                    acc += fr[0] * (s + 1)
                out[i] = acc * self.NS

    return Fam


def run_scenario(name, arch1, arch2, reuse, nd_only):
    import quadrants as qd

    def T(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    cache = args.cache
    T(f"init {arch1}")
    qd.init(arch={"cpu": qd.cpu, "metal": qd.metal}[arch1], offline_cache_file_path=cache)
    Fam = make_family("A")
    a = Fam(8)
    fr = np.array([1.0, 2.0], np.float32)
    out = np.zeros(4, np.float32)
    T("family A first launch")
    a.scan_nd(fr, out, 4) if nd_only else a.scan(fr, out, 4)
    qd.sync()
    T(f"family A OK out[:2]={out[:2].tolist()}")

    T(f"SECOND init {arch2}")
    qd.init(arch={"cpu": qd.cpu, "metal": qd.metal}[arch2], offline_cache_file_path=cache)
    T("second init returned")

    if reuse:
        T("RE-launch family A on surviving objects")
        if nd_only:
            a.scan_nd(fr, out, 4)
        else:
            a.scan(fr, out, 4)
        qd.sync()
        T(f"family A re-launch OK out[:2]={out[:2].tolist()}")

    FamB = make_family("B")
    b = FamB(8)
    out2 = np.zeros(4, np.float32)
    T("family B first launch (fresh fields under new program)")
    b.scan(fr, out2, 4)
    qd.sync()
    T(f"family B OK out[:2]={out2[:2].tolist()}")
    T("SCENARIO PASS")



def run_realloc(arch):
    """C0-D realloc path, single program: launch -> create field -> relaunch old kernel."""
    import quadrants as qd

    def T(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    qd.init(arch={"cpu": qd.cpu, "metal": qd.metal}[arch], offline_cache_file_path=args.cache)
    Fam = make_family("A")
    a = Fam(8)
    fr = np.array([1.0, 2.0], np.float32)
    out = np.zeros(4, np.float32)
    T("launch #1")
    a.scan(fr, out, 4)
    qd.sync()
    T(f"launch #1 OK out[:2]={out[:2].tolist()}")
    T("creating a NEW field after launch (triggers root grow on cpu)")
    _ = qd.field(dtype=qd.f32, shape=(1024,))
    _.from_numpy(np.zeros(1024, np.float32))
    T("new field created; re-launching OLD kernel")
    a.scan(fr, out, 4)
    qd.sync()
    T(f"launch #2 OK out[:2]={out[:2].tolist()}  (no crash)")
    T("SCENARIO PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=[
        "base_cpu", "base_metal", "abandon", "reuse", "reuse_metal_cpu", "same_arch", "abandon_nd",
        "realloc_cpu", "realloc_metal"])
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()
    s = args.scenario
    if s == "base_cpu":
        run_scenario(s, "cpu", "cpu", reuse=False, nd_only=False)
    elif s == "base_metal":
        run_scenario(s, "metal", "metal", reuse=False, nd_only=False)
    elif s == "abandon":
        run_scenario(s, "cpu", "metal", reuse=False, nd_only=False)
    elif s == "reuse":
        run_scenario(s, "cpu", "metal", reuse=True, nd_only=False)
    elif s == "reuse_metal_cpu":
        run_scenario(s, "metal", "cpu", reuse=True, nd_only=False)
    elif s == "same_arch":
        run_scenario(s, "cpu", "cpu", reuse=True, nd_only=False)
    elif s == "abandon_nd":
        run_scenario(s, "cpu", "metal", reuse=True, nd_only=True)
    elif s == "realloc_cpu":
        run_realloc("cpu")
    elif s == "realloc_metal":
        run_realloc("metal")
