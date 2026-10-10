"""Probe 2: where does the static-unroll compile cost live, per fresh process?

For the coll_scan-like variant A kernel at growing NS, run TWO processes against one
private cache:
  p1 = compile process: t_total = python-frontend AST + key-derivation + C++ compile + launch
  p2 = hit-or-miss process: same script, same cache.
Key equality => C++ disk HIT for p2 => p2's t_total is what ANY fresh process pays even when
the artifact is cached (the per-process fixed cost: python AST build + AST serialization for
the key). (t_p1 - t_p2) approximates the C++ backend compile cost.

Small NS only (<=256): the real coll_scan body is far heavier; we fit the scaling law,
never reproducing the 22-minute compile.
"""
import argparse
import json
import time

import numpy as np
import quadrants as qd

import probe_keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, default=32)
    ap.add_argument("--arch", default="metal")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", default="p1")
    args = ap.parse_args()

    arch = {"cpu": qd.cpu, "metal": qd.metal}[args.arch]
    qd.init(arch=arch, offline_cache_file_path=args.cache)

    A, *_ = probe_keys.build_variants(False)
    kin = A(ns=args.ns, m=8)
    fr = np.random.RandomState(0).rand(4, 3).astype(np.float32)
    out = np.zeros(4, np.float32)

    t0 = time.perf_counter()
    kin.scan(fr, out, 4)
    t_total = time.perf_counter() - t0
    qd.sync()
    t1 = time.perf_counter()
    kin.scan(fr, out, 4)
    t_warm = time.perf_counter() - t1
    qd.sync()

    key = None
    prog = qd.lang.impl.get_runtime().prog
    primal = type(kin).__dict__["scan"]._primal
    for k, t_kernel in primal.materialized_kernels.items():
        key = prog.compile_kernel(prog.config(), prog.get_device_caps(), t_kernel).cache_key

    print("PHASE " + json.dumps({
        "tag": args.tag, "ns": args.ns, "arch": args.arch,
        "t_total_s": round(t_total, 3),
        "t_second_launch_s": round(t_warm, 6),
        "cxx_key16": key[:16] if key else None,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
