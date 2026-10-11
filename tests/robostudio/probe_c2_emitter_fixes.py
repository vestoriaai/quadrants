"""Gate probe (mujocoqd/c2-emitter-fixes): three HEAD C2 emitter/analyze defects.

Each defect gets its own hard gate arm here; the flags-OFF path must stay
untouched (the scalar arm below is plain scalar emission, and the union
runner's gates 7/9 re-pin OFF-path bitwise parity independently).

Defect map (all HEAD-existing, reproducible on the dev59/dev71 wheels):

  1. arg-in-DAG env miss: a vectorized loop body whose DAG references a
     value defined OUTSIDE the body (task-level scalar kernel-arg load,
     or a loop-invariant global load / pure arithmetic over them hoisted
     out by cache-loop-invariant-global-vars) crashed the emitter env
     lookup (unordered_map::at / env-end assert). vec_analyze admitted
     these bodies (the "argload used as value" rule only sees IN-body
     argloads, which the hoister has already moved out), so the crash -
     not a rejection - was the observable. Gate G1: the three reduction
     forms below must run in the vec arm, log no IR-builder errors, and
     stay bitwise identical to the scalar arm. This includes the
     fast-path hazard: a reduction whose accumulated operand has a
     loop-invariant uniform leaf gets refolded by Apple's MSL fast-math
     into (x0+x1+x2+x3)*c (measured ~2 ulp on `s += x[j]*c`), so such
     reductions must keep the order-barrier lane loop.

  2. coeff-0 remainder duplicate registration: names the vector walk
     registers (ScalarMemLoad / ScalarCompute re-emissions) leaked into
     the remainder walk's register_value calls -> "already exists"
     IR-builder error. Gate G3 (lands with the fix commit): the
     multi-site coeff-0 read form must run clean and bitwise.

  3. k_elementwise never vectorized: vec_analyze took the memory-access
     dtype from mem->element_type(), which for a GlobalStoreStmt is the
     meaningless (unknown) ret_type, so EVERY inner loop with an in-body
     global store was rejected as "non-f32 memory access" - reduction
     loops (no in-body store) vectorized, elementwise loops never did.
     Gate G2 (lands with the fix commit): the elementwise forms must be
     bitwise AND actually vectorized (float4 accesses in the dumped
     SPIR-V, via QD_DUMP_IR=1).

One qd.init per process (S-guard): the parent re-executes this file as
subprocess arms scalar / vec / dump and compares sha256 hashes + stderr.

Run: python probe_c2_emitter_fixes.py [metal]   (exit 0 = all arms pass)
"""
import hashlib
import os
import subprocess
import sys
import time

import numpy as np

ARCH = sys.argv[1] if len(sys.argv) > 1 else "metal"
ARM = os.environ.get("QD_C2FIX_ARM", "")

if ARCH != "metal":
    print(f"c2-emitter-fixes gate arch={ARCH}: SKIP (transform is SPIR-V-only) -> exit 0")
    sys.exit(0)

# ------------------------------------------------------------------ kernels ----


def _define_kernels(qd):
    # defect 3 form (k_elementwise shape): scalar kernel arg in an
    # elementwise inner loop with an in-body global store.
    @qd.kernel
    def k_ew_arg(x: qd.types.ndarray(ndim=2), y: qd.types.ndarray(ndim=2), a: qd.f32, nv: qd.i32):
        for w in range(y.shape[0]):
            for j in range(nv):
                y[w, j] = a * x[w, j] + 3.0

    # defect 1 form: loop-invariant arithmetic over a kernel arg, hoisted
    # out of the inner loop, feeding an elementwise store loop.
    @qd.kernel
    def k_ew_extemp(x: qd.types.ndarray(ndim=2), y: qd.types.ndarray(ndim=2), a: qd.f32, nv: qd.i32):
        for w in range(y.shape[0]):
            t = a * 2.0
            for j in range(nv):
                y[w, j] = x[w, j] * t

    # defect 3 + 1 form: coeff-0 (single-site) ndarray load, hoisted out of
    # the inner loop, + in-body global store.
    @qd.kernel
    def k_ew_c0(x: qd.types.ndarray(ndim=2), y: qd.types.ndarray(ndim=2), c: qd.types.ndarray(ndim=1),
                nv: qd.i32):
        for w in range(y.shape[0]):
            for j in range(nv):
                y[w, j] = x[w, j] * c[w]

    # defect 2 form: coeff-0 read on the SAME array as the coeff-1 read
    # (two access sites -> cache-loop-invariant-global-vars declines to
    # hoist -> the coeff-0 load stays IN the body as a ScalarMemLoad)
    # + in-body global store.
    @qd.kernel
    def k_ew_c0same(x: qd.types.ndarray(ndim=2), y: qd.types.ndarray(ndim=2), nv: qd.i32):
        for w in range(y.shape[0]):
            for j in range(nv):
                y[w, j] = x[w, j] + x[w, 0]

    # defect 1 form: scalar kernel arg inside a reduction accumulator DAG.
    @qd.kernel
    def k_red_arg(x: qd.types.ndarray(ndim=2), y: qd.types.ndarray(ndim=2), a: qd.f32, nv: qd.i32):
        for w in range(y.shape[0]):
            s = 0.0
            for j in range(nv):
                s += x[w, j] * a
            y[w, 0] = s

    # defect 1 form: hoisted invariant temp inside a reduction accumulator DAG.
    @qd.kernel
    def k_red_extemp(x: qd.types.ndarray(ndim=2), y: qd.types.ndarray(ndim=2), a: qd.f32, nv: qd.i32):
        for w in range(y.shape[0]):
            t = a * 2.0
            s = 0.0
            for j in range(nv):
                s += x[w, j] * t
            y[w, 0] = s

    # defect 1 form: hoisted coeff-0 ndarray load inside a reduction DAG
    # (the load is loop-unique -> hoisted -> a naive fast-path lane DAG
    # would let Apple's MSL fast-math factor it across the four lanes).
    @qd.kernel
    def k_red_c0hoisted(x: qd.types.ndarray(ndim=2), y: qd.types.ndarray(ndim=2),
                        c: qd.types.ndarray(ndim=1), nv: qd.i32):
        for w in range(y.shape[0]):
            s = 0.0
            for j in range(nv):
                s += x[w, j] * c[w]
            y[w, 0] = s

    return (k_ew_arg, k_ew_extemp, k_ew_c0, k_ew_c0same, k_red_arg, k_red_extemp, k_red_c0hoisted)


# --------------------------------------------------------------- child arms ----
if ARM:
    import quadrants as qd

    (k_ew_arg, k_ew_extemp, k_ew_c0, k_ew_c0same, k_red_arg, k_red_extemp, k_red_c0hoisted) = _define_kernels(qd)
    vec = ARM in ("vec", "dump")
    dump = ARM == "dump"
    if dump:
        os.environ["QD_DUMP_IR"] = "1"
        dump_dir = os.environ.get("QD_C2FIX_DUMP_DIR", "/tmp/qd_c2fix_dump")
        qd.init(arch=qd.metal, spirv_vectorize=vec, offline_cache=False, debug_dump_path=dump_dir)
    else:
        qd.init(arch=qd.metal, spirv_vectorize=vec, offline_cache=False)

    def dev(a):
        t = qd.ndarray(dtype=qd.f32, shape=a.shape)
        t.from_numpy(np.ascontiguousarray(a))
        return t

    def hash_of(arr):
        return hashlib.sha256(np.ascontiguousarray(arr.to_numpy()).tobytes()).hexdigest()

    rng = np.random.default_rng(11)
    nw, nv = 3, 16
    x2 = dev(rng.standard_normal((nw, nv)).astype(np.float32))
    c1 = dev(rng.standard_normal(nw).astype(np.float32))
    a32 = np.float32(0.5)

    # Compile every form once (covers both the analyze verdicts and, on the
    # dump arm, the SPIR-V dumps of all tasks on disk).
    for kern, args in ((k_ew_arg, (a32, nv)), (k_ew_extemp, (a32, nv)), (k_ew_c0, (c1, nv)),
                       (k_ew_c0same, (nv,)), (k_red_arg, (a32, nv)), (k_red_extemp, (a32, nv)),
                       (k_red_c0hoisted, (c1, nv))):
        y = dev(np.zeros((nw, nv), dtype=np.float32))
        kern(x2, y, *args)
        if not dump:
            print(f"HASH {kern.__name__} {hash_of(y)}")
    qd.sync()

    if dump:
        print("DUMP DONE")
    sys.exit(0)

# ------------------------------------------------------------------ parent ----
QPU = [sys.executable, os.path.abspath(__file__), ARCH]


def run_arm(arm, timeout=600):
    env = dict(os.environ)
    env["QD_C2FIX_ARM"] = arm
    env.pop("QD_C2FIX_DUMP_DIR", None)
    return subprocess.run(QPU, env=env, capture_output=True, text=True, timeout=timeout)


fails = []


def gate(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL':4}] {name} {detail}")
    if not ok:
        fails.append(name)


print(f"== GATE c2-emitter-fixes (Metal, {time.strftime('%F %T')}) ==")

hashes = {}
errs = {}
for arm in ("scalar", "vec"):
    r = run_arm(arm)
    got = {}
    for line in r.stdout.splitlines():
        if line.startswith("HASH "):
            _, name, h = line.split()
            got[name] = h
    hashes[arm] = got
    errs[arm] = (r.returncode, r.stderr)
    if r.returncode != 0 or not got:
        gate(f"arm {arm} runs", False, f"exit {r.returncode}")
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        sys.exit(1)

names = sorted(hashes["scalar"])
if not names or names != sorted(hashes["vec"]):
    gate("arms cover the same outputs", False)
    sys.exit(1)

# IR-builder error signatures that must never appear on either path
ir_err = {arm: [ln for ln in errs[arm][1].splitlines() if "already exists" in ln or "does not yet exist" in ln]
          for arm in ("scalar", "vec")}

# G1 (defect 1: arg-in-DAG env miss) --------------------------------------------
g1_forms = ["k_red_arg", "k_red_extemp", "k_red_c0hoisted"]
g1_bad = [n for n in g1_forms if hashes["scalar"].get(n) != hashes["vec"].get(n)]
gate("G1 arg-in-DAG reductions: bitwise scalar==vec, no crash", not g1_bad,
     "" if not g1_bad else f"mismatch: {g1_bad}")
gate("G1 no IR-builder errors on any form", not any(ir_err.values()),
     "" if not any(ir_err.values()) else f"{ir_err}")

print(f"\nC2 EMITTER-FIXES GATE arch={ARCH}: "
      f"{'ALL HARD GATES PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
