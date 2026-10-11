#!/usr/bin/env python3
"""Minimal repro for upstream issue #1: same-process second qd.init +
cross-family JIT corrupts compile state (quadrants 1.3.3).

Quadrants-only (numpy aside).  Two independent data_oriented families,
each calling qd.init in its ctor (the production QdKin / EsdfKin
pattern), then interleaved launches.

  python -u probe_multi_init.py                # full battery, PASS/FAIL per arm
  python -u probe_multi_init.py --scenario a2b # single arm: A, re-init, B
"""
import argparse
import traceback

import numpy as np
import quadrants as qd

AP = argparse.ArgumentParser()
AP.add_argument("--scenario", default="all",
                choices=["all", "a2b", "ab_a", "ab_newk", "jit_heavy"])
AP.add_argument("--arch", default="cpu")
AP.add_argument("--single-init", action="store_true",
                help="S5 experiment: families share ONE qd.init (contract lift test)")
args = AP.parse_args()
ARCH = getattr(qd, args.arch)


_INIT_DONE = {"flag": False}


def maybe_init(arch=ARCH):
    # double-init mode: every family inits (the defect scenario);
    # single-init mode (--single-init): exactly one init for the process
    # (the S5 contract-lift experiment).
    if args.single_init:
        if not _INIT_DONE["flag"]:
            qd.init(arch=arch)
            _INIT_DONE["flag"] = True
    else:
        qd.init(arch=arch)


@qd.data_oriented
class FamA:
    """Family A: 3x3 matrix field, dim 3."""

    def __init__(self, n):
        maybe_init()
        self.n = n
        self.fr = qd.Matrix.field(3, 3, qd.f32, shape=(n,))
        self.fr.from_numpy(np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)))

    @qd.kernel
    def k(self, out: qd.types.ndarray()):
        for i in range(self.n):
            acc = 0.0
            for a in qd.static(range(3)):
                for b in qd.static(range(3)):
                    acc += self.fr[i][a, b]
            out[i] = acc

    def expect(self):
        return 3.0


@qd.data_oriented
class FamB:
    """Family B: 4x4 matrix field (different dim on purpose)."""

    def __init__(self, n):
        maybe_init()
        self.n = n
        self.fr = qd.Matrix.field(4, 4, qd.f32, shape=(n,))
        self.fr.from_numpy(np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)))

    @qd.kernel
    def k(self, out: qd.types.ndarray()):
        for i in range(self.n):
            acc = 0.0
            for a in qd.static(range(4)):
                for b in qd.static(range(4)):
                    acc += self.fr[i][a, b]
            out[i] = acc

    def expect(self):
        return 4.0


@qd.data_oriented
class FamJit:
    """JIT-heavy family (static unrolled inner product, many loads)."""

    def __init__(self, n):
        maybe_init()
        self.n = n
        self.x = qd.Vector.field(12, qd.f32, shape=(n,))
        self.x.from_numpy(np.ones((n, 12), dtype=np.float32))

    @qd.kernel
    def k(self, out: qd.types.ndarray()):
        for i in range(self.n):
            acc = 0.0
            for j in qd.static(range(12)):
                acc += self.x[i][j] * (j + 1)
            out[i] = acc

    def expect(self):
        return float(sum(j + 1 for j in range(12)))


def run_pair(name, mk1, mk2, relaunches=3):
    """mk1()/mk2() construct the two families (each calls qd.init)."""
    print(f"--- arm {name} ---", flush=True)
    try:
        f1 = mk1()
        o1 = np.zeros(f1.n, dtype=np.float32)
        f1.k(o1)
        ok1 = bool(np.allclose(o1, f1.expect(), atol=1e-5))
        print(f"  launch1: {'ok' if ok1 else f'WRONG vals max={o1.max()}'}", flush=True)

        f2 = mk2()
        o2 = np.zeros(f2.n, dtype=np.float32)
        f2.k(o2)
        ok2 = bool(np.allclose(o2, f2.expect(), atol=1e-5))
        print(f"  launch2: {'ok' if ok2 else f'WRONG vals max={o2.max()}'}", flush=True)

        verdict = "PASS"
        for r in range(relaunches):
            o1[:] = 0
            f1.k(o1)
            okr = bool(np.allclose(o1, f1.expect(), atol=1e-5))
            print(f"  relaunch{r}: {'ok' if okr else f'WRONG vals max={o1.max()}'}", flush=True)
            if not okr:
                verdict = "FAIL"
        if not (ok1 and ok2):
            verdict = "FAIL"
        print(f"  ARM {name}: {verdict}", flush=True)
        return verdict
    except Exception:
        print("  EXCEPTION:", flush=True)
        traceback.print_exc()
        print(f"  ARM {name}: RAISED", flush=True)
        return "RAISED"


def arm_a2b():
    # A compile+launch, second init + B, then relaunch A.
    return run_pair("a2b", lambda: FamA(512), lambda: FamB(256), relaunches=3)


def arm_ab_a():
    # Construct BOTH first (no launch), then interleave launches.
    f1 = FamA(512)
    f2 = FamB(256)
    o1 = np.zeros(f1.n, dtype=np.float32)
    o2 = np.zeros(f2.n, dtype=np.float32)
    verdict = "PASS"
    for r in range(3):
        f1.k(o1)
        f2.k(o2)
        ok1 = bool(np.allclose(o1, f1.expect(), atol=1e-5))
        ok2 = bool(np.allclose(o2, f2.expect(), atol=1e-5))
        print(f"  pair{r}: A {'ok' if ok1 else 'WRONG'} B {'ok' if ok2 else 'WRONG'}", flush=True)
        if not (ok1 and ok2):
            verdict = "FAIL"
    print(f"  ARM ab_a: {verdict}", flush=True)
    return verdict


@qd.data_oriented
class FamAExtra:
    """A fresh kernel compiled in family A AFTER family B was initialized."""

    def __init__(self, n):
        maybe_init()
        self.n = n
        self.fr = qd.Matrix.field(3, 3, qd.f32, shape=(n,))
        self.fr.from_numpy(np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)))

    @qd.kernel
    def k3(self, out: qd.types.ndarray()):
        for i in range(self.n):
            acc = 0.0
            for a in qd.static(range(3)):
                acc += self.fr[i][a, a]
            out[i] = acc


def arm_ab_newk():
    # Cross-family JIT: after B exists, define and launch a NEW kernel in A.
    f1 = FamA(512)
    f2 = FamB(256)
    o2 = np.zeros(f2.n, dtype=np.float32)
    f2.k(o2)

    f3 = FamAExtra(128)
    o3 = np.zeros(f3.n, dtype=np.float32)
    try:
        f3.k3(o3)
        ok = bool(np.allclose(o3, 3.0, atol=1e-5))
        print(f"  new-kernel-after-cross-init: {'ok' if ok else 'WRONG vals'}", flush=True)
        return "PASS" if ok else "FAIL"
    except Exception:
        traceback.print_exc()
        return "RAISED"


def arm_jit_heavy():
    # JIT-heavy first family, then second family (upstream #1 shape).
    return run_pair("jit_heavy", lambda: FamJit(1024), lambda: FamB(256), relaunches=3)


BATTERY = {
    "a2b": arm_a2b,
    "ab_a": arm_ab_a,
    "ab_newk": arm_ab_newk,
    "jit_heavy": arm_jit_heavy,
}

if args.scenario == "all":
    results = {}
    for name, fn in BATTERY.items():
        try:
            results[name] = fn()
        except Exception:
            traceback.print_exc()
            results[name] = "RAISED"
    print("=== battery summary (arch=%s) ===" % args.arch, flush=True)
    for k, v in results.items():
        print(f"  {k}: {v}", flush=True)
else:
    print(f"ARM {args.scenario}: {BATTERY[args.scenario]()}", flush=True)
