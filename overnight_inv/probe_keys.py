"""Overnight probe 1: cross-process compile-cache key stability for class kernels.

Variants:
  A  coll_scan-like: data_oriented kernel, reads instance FIELDS + qd.static(range(self.NS)) unroll,
     calls ONE @qd.func helper.
  B  same class shape but kernel reads NO fields (ndarray args + self.NS int only, same static unroll).
  C  reads fields but NO static NS unroll (dynamic loop bound via kernel arg).
  D  like A but calls TWO @qd.func helpers (real-func pointer-order sensitivity).
  E  spline_eval-like: static range(self.M) over a basis field Wp.
  P  module-scope-field kernel (classic expected-to-persist control).

Each run prints, for every compiled kernel: the C++ offline-cache key (CompileResult.cache_key),
the instance's SNode ids, wall time, and whether this was the first compile in-process.

CLI controls: --variant --ns --m --perturb N --two-inst --pure --arch --cache
"""
import argparse
import os
import sys
import time

import numpy as np
import quadrants as qd


def build_variants(pure: bool):
    dec = qd.kernel

    @qd.data_oriented
    class A:
        def __init__(self, ns=8, m=8):
            self.NS = int(ns)
            self.M = int(m)
            self.s_ctr = qd.Vector.field(3, qd.f32, shape=(self.NS,))
            self.s_rad = qd.field(dtype=qd.f32, shape=(self.NS,))
            self.s_par = qd.field(dtype=qd.i32, shape=(self.NS,))
            self.coef = qd.Vector.field(4, qd.f32, shape=(1,))
            self.grid = qd.field(dtype=qd.f32, shape=(4, 4, 4))
            self.s_ctr.from_numpy(np.arange(self.NS * 3, dtype=np.float32).reshape(self.NS, 3))
            self.s_rad.from_numpy(np.ones(self.NS, np.float32) * 0.1)
            self.s_par.from_numpy(np.zeros(self.NS, np.int32))
            self.coef.from_numpy(np.array([[0.5, 0, 0, 0.01]], np.float32))

        @qd.func
        def helper1(self, s):
            return self.s_rad[s] * 0.5

        @qd.func
        def helper2(self, x):
            return x + 1.0

        @dec
        def scan(self, fr: qd.types.ndarray(ndim=2, dtype=qd.f32),
                 out: qd.types.ndarray(ndim=1, dtype=qd.f32), n: int):
            acc = 0.0
            for i in range(n):
                for s in qd.static(range(self.NS)):
                    c = self.s_ctr[s]
                    r = self.helper1(s)
                    acc += (fr[i, 0] * c[0] + r) * self.coef[0][0] + self.grid[0, 0, 0]
                out[i] = acc

    @qd.data_oriented
    class D(A):
        @dec
        def scan2(self, fr: qd.types.ndarray(ndim=2, dtype=qd.f32),
                  out: qd.types.ndarray(ndim=1, dtype=qd.f32), n: int):
            acc = 0.0
            for i in range(n):
                for s in qd.static(range(self.NS)):
                    c = self.s_ctr[s]
                    r = self.helper1(s)
                    r2 = self.helper2(r)
                    acc += (fr[i, 0] * c[0] + r2) * self.coef[0][0]
                out[i] = acc

    @qd.data_oriented
    class B:
        def __init__(self, ns=8, m=8):
            self.NS = int(ns)
            self.M = int(m)

        @dec
        def scan(self, fr: qd.types.ndarray(ndim=2, dtype=qd.f32),
                 out: qd.types.ndarray(ndim=1, dtype=qd.f32), n: int):
            acc = 0.0
            for i in range(n):
                for s in qd.static(range(self.NS)):
                    acc += fr[i, s % 3] * (s + 1)
                out[i] = acc * self.NS

    @qd.data_oriented
    class C:
        def __init__(self, ns=8, m=8):
            self.NS = int(ns)
            self.s_rad = qd.field(dtype=qd.f32, shape=(self.NS,))
            self.coef = qd.Vector.field(4, qd.f32, shape=(1,))
            self.s_rad.from_numpy(np.ones(self.NS, np.float32) * 0.1)
            self.coef.from_numpy(np.array([[0.5, 0, 0, 0.01]], np.float32))

        @dec
        def scan(self, fr: qd.types.ndarray(ndim=2, dtype=qd.f32),
                 out: qd.types.ndarray(ndim=1, dtype=qd.f32), n: int):
            acc = 0.0
            for i in range(n):
                for s in range(self.NS):
                    acc += self.s_rad[s] * fr[i, 0] + self.coef[0][0]
                out[i] = acc

    @qd.data_oriented
    class E:
        def __init__(self, ns=8, m=8):
            self.M = int(m)
            self.Wp = qd.field(dtype=qd.f32, shape=(m, m))

        @dec
        def eval(self, C: qd.types.ndarray(ndim=2, dtype=qd.f32),
                 Q: qd.types.ndarray(ndim=1, dtype=qd.f32), n: int):
            for i in range(n):
                acc = 0.0
                for j in qd.static(range(self.M)):
                    acc += self.Wp[i, j] * C[i, j]
                Q[i] = acc

    if pure:
        # post-hoc is_pure=True (qd.kernel(fastcache=True) as a nested decorator breaks
        # the class-stackframe detection); is_pure is read at call time via
        # Kernel.quadrants_callable.is_pure (kernel.py::_try_load_fastcache).
        for kls, names in ((A, ["scan"]), (B, ["scan"]), (C, ["scan"]), (D, ["scan", "scan2"]), (E, ["eval"])):
            for nm in names:
                if hasattr(kls, nm):
                    getattr(kls, nm)._primal.quadrants_callable.is_pure = True
    return A, B, C, D, E


MODULE_FIELD = None


@qd.kernel
def kern_module(f: qd.template(), out: qd.types.ndarray(ndim=1, dtype=qd.f32)):
    for i in range(4):
        out[i] = f[0] + MODULE_FIELD[0]


def snode_report(obj, names):
    rep = {}
    for n in names:
        f = getattr(obj, n, None)
        if f is None or not hasattr(f, "snode"):
            rep[n] = None
        else:
            sn = f.snode
            rep[n] = {"id": sn._id, "tree": sn._snode_tree_id}
    return rep


def compile_key_of(t_kernel):
    from quadrants.lang import impl as _impl
    prog = _impl.get_runtime().prog
    res = prog.compile_kernel(prog.config(), prog.get_device_caps(), t_kernel)
    return res.cache_key, res.cache_hit


def launch_and_report(tag, kin, kernel_attr, args, extra_names=()):
    t0 = time.perf_counter()
    getattr(kin, kernel_attr)(*args)
    dt = time.perf_counter() - t0
    primal = type(kin).__dict__[kernel_attr]._primal
    key = None
    for k, t_kernel in primal.materialized_kernels.items():
        ck, hit = compile_key_of(t_kernel)
        key = ck
    info = {"tag": tag, "kernel": kernel_attr, "wall_s": round(dt, 3), "cxx_key": key,
            "snode_ids": snode_report(kin, ["s_ctr", "s_rad", "s_par", "coef", "grid", "Wp"])}
    print("RESULT " + json_dumps(info), flush=True)
    return key


def json_dumps(d):
    import json
    return json.dumps(d, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A", choices=["A", "B", "C", "D", "E", "P"])
    ap.add_argument("--ns", type=int, default=8)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--perturb", type=int, default=0)
    ap.add_argument("--two-inst", action="store_true")
    ap.add_argument("--pure", action="store_true")
    ap.add_argument("--arch", default="cpu")
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()

    arch = {"cpu": qd.cpu, "metal": qd.metal}[args.arch]
    qd.init(arch=arch, offline_cache_file_path=args.cache)
    print(f"CFG variant={args.variant} ns={args.ns} perturb={args.perturb} two_inst={args.two_inst} "
          f"pure={args.pure} arch={args.arch}", flush=True)

    # perturbation: unrelated fields created BEFORE the target instance shifts global snode-id counter
    for i in range(args.perturb):
        _ = qd.field(dtype=qd.f32, shape=(3,))

    A, B, C, D, E = build_variants(args.pure)

    if args.variant == "P":
        global MODULE_FIELD
        MODULE_FIELD = qd.field(dtype=qd.f32, shape=(1,))
        MODULE_FIELD[0] = 1.5
        out = np.zeros(4, np.float32)
        t0 = time.perf_counter()
        kern_module(MODULE_FIELD, out)
        dt = time.perf_counter() - t0
        t_kernel = list(kern_module._primal.materialized_kernels.values())[0]
        ck, hit = compile_key_of(t_kernel)
        print(f"RESULT {json_dumps({'tag': 'P', 'kernel': 'kern_module', 'wall_s': round(dt, 3), 'cxx_key': ck})}",
              flush=True)
        print("OUT", out.tolist(), flush=True)
        return

    cls = {"A": A, "B": B, "C": C, "D": D, "E": E}[args.variant]
    kin1 = cls(ns=args.ns, m=args.m)
    fr = np.random.RandomState(0).rand(4, 3).astype(np.float32)
    out = np.zeros(4, np.float32)

    if args.variant == "D":
        launch_and_report("inst1", kin1, "scan2", (fr, out, 4))
    elif args.variant == "E":
        Cc = np.random.RandomState(1).rand(8, args.m).astype(np.float32)
        Q = np.zeros(8, np.float32)
        launch_and_report("inst1", kin1, "eval", (Cc, Q, 8))
    else:
        launch_and_report("inst1", kin1, "scan", (fr, out, 4))

    if args.two_inst:
        kin2 = cls(ns=args.ns, m=args.m)
        out2 = np.zeros(4, np.float32)
        if args.variant == "D":
            launch_and_report("inst2", kin2, "scan2", (fr, out2, 4))
        elif args.variant == "E":
            Cc = np.random.RandomState(1).rand(8, args.m).astype(np.float32)
            Q2 = np.zeros(8, np.float32)
            launch_and_report("inst2", kin2, "eval", (Cc, Q2, 8))
        else:
            launch_and_report("inst2", kin2, "scan", (fr, out2, 4))

    qd.sync()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
