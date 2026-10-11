"""Gate probe (mujocoqd/reassociate-flag): opt-in FP reassociation inside the C2
vector main loop.

The `reassociate` compile_config flag (default OFF, serialized into the
offline-cache key, implies spirv_vectorize) rewrites qualifying loop-carried
scalar reduction accumulators of the C2 float4 vector main loop into k=4
partial sums (stride-4, per-lane phis) with a tree merge after the scalar
remainder; the order-barrier inner lane loop is dropped for them. Contract:
the ON path is NON-bitwise by definition (accuracy stays inside the 1e-5
physics gate); the OFF path must stay bitwise identical to the scalar path.

Three subprocess arms for the codegen/hash gates (one qd.init per process is
the S-guard contract):
  scalar : qd.init(arch=metal)                       (everything off)
  vec    : qd.init(arch=metal, spirv_vectorize=True) (C2, bitwise)
  rea    : qd.init(arch=metal, reassociate=True)     (flag alone; implication
                                                      must turn C2 on)

Gates:
  G1 (HARD) OFF path bitwise unchanged: scalar == vec on every output hash
        (re-validates the C2 contract under this branch).
  G2 (HARD) reassociation actually happens: rea != vec bitwise on reduction
        outputs where the vector main loop provably runs (nv in {24,23}; the
        payoff shapes at nw=1).
  G3 (HARD) flag-inert anchor: rea == vec == scalar on k_elementwise (its
        inner loop stays on the plain scalar path -- arg-in-DAG -- so this
        pins the flag leaves untouched loops untouched; hash-only).
  G4 (HARD) implication semantics: rea != scalar on the reduction outputs --
        reassociation only exists inside the C2 vector path, so differing
        => C2 was on with the flag alone.
  G5 (HARD) accuracy: rea arm within 1e-5 max rel-err of a numpy float64
        reference on the payoff shapes (probe_reassociate_payoff measured
        worst ~1e-6 for reassociated chains). chol_dot uses the payoff
        probe's positive PSD recipe: cancelling data explodes the rel metric
        on near-zero dots (measured 1.4e-4 @ ulp 2031 with tril(normal) data
        vs 1.5e-7 @ ulp 2 with PSD data -- data geometry, not codegen).
  G6 (WARN-ONLY) payoff: in-process interleaved speedup of the reassociated
        codegen (plain source, reassociate=ON) over the sequential-chain
        scalar baseline (identical source + loop_config(serialize=True) on
        the inner loop, which vec_analyze rejects -> plain scalar emission
        in the SAME process) >= 1.4x on matvec m=48 and symmatvec m=64.
        chol m=64 informational. Clock warmup, then 8 pairs of deep ~30ms
        A/S bursts in ALTERNATING order, per-arm MINIMUM (machine modes only
        add time, so min converges to the true cost; median reported
        alongside). Machine bimodality lesson (probe_reassociate_payoff /
        C2 gate-4): relative in-process ratios only, never absolute wall
        clock. Two earlier estimators were tried and rejected on this
        machine: cross-process ABA child rounds (mode-sticky per process,
        ratios flipped sign between runs) and short-burst anchor-dropping
        sandwiches (acceptance-set selection bias, same flip problem).

Baseline note: a literal same-process flag-ON vs flag-OFF pair is impossible
(one qd.init per process, S-guard), and the two constructs that could fake a
flag-OFF emission inside an ON process (runtime-scalar factor / coeff-0 load
in the reduction DAG) both trip PRE-EXISTING C2 emitter limitations (argload
outside the loop body has no env entry; coeff-0 loads re-register in the
remainder walk) -- see the ledger entry. The serialized-loop baseline is the
same sequential-accumulation semantics the flag-OFF path computes (bitwise
per G1), measured in-process. The cross-process vec-arm microseconds are
reported for context only (mode-sticky across processes, never compared to
the ON child).

Non-metal archs: SKIP, exit 0 (the transform only runs on SPIR-V backends).

Run: python probe_reassociate_flag.py [metal] [--quick]   (exit 0 = all hard
gates pass; perf warnings do not affect the exit code)
"""
import argparse
import hashlib
import os
import statistics
import subprocess
import sys
import time

import numpy as np

import quadrants as qd
from quadrants.lang.misc import loop_config

ARCH = sys.argv[1] if len(sys.argv) > 1 else "metal"
ARM = os.environ.get("QD_REASSOC_ARM", "")
PERF_ARM = os.environ.get("QD_REASSOC_PERF_ARM", "")

if ARCH != "metal":
    print(f"reassociate-flag gate arch={ARCH}: SKIP (transform is SPIR-V-only) -> exit 0")
    sys.exit(0)

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true", help="fewer perf rounds, skip chol perf shape")
args, _ = p.parse_known_args()

# ---------------------------------------------------------------- kernels ----
# C2 gate kernel family (bitwise anchors; k_cholrow carries the `s -= X`
# frontend neg-form reduction and a division).

@qd.kernel
def k_dot(m: qd.types.ndarray(ndim=3), v: qd.types.ndarray(ndim=2), o: qd.types.ndarray(ndim=2),
          nv: qd.i32, niter: qd.i32):
    for w in range(o.shape[0]):
        for i in range(nv):
            s = 0.0
            for _ in range(niter):
                for j in range(nv):
                    s += m[w, i, j] * v[w, j]
            o[w, i] = s


@qd.kernel
def k_cholrow(m: qd.types.ndarray(ndim=3), v: qd.types.ndarray(ndim=2), o: qd.types.ndarray(ndim=2), nv: qd.i32):
    for w in range(o.shape[0]):
        for i in range(nv):
            s = m[w, i, i]
            for j in range(nv):
                s -= m[w, i, j] * v[w, j]
            o[w, i] = s / (v[w, i] + 2.0)


@qd.kernel
def k_elementwise(x: qd.types.ndarray(ndim=2), y: qd.types.ndarray(ndim=2), a: qd.f32, nv: qd.i32):
    for w in range(y.shape[0]):
        for j in range(nv):
            y[w, j] = a * x[w, j] + 3.0


# Payoff shapes (probe_reassociate_payoff): under-occupied per-world threads +
# long serial chains -- the shapes the payoff probe measured at 1.5-2.8x.
# Each shape has a plain form (reassociated by the flag under ON) and a
# serialized-inner-loop form (vec_analyze rejects -> plain scalar emission in
# ANY arm: the sequential-chain baseline, runnable in the ON process).

@qd.kernel
def k_matvec(A: qd.types.ndarray(ndim=3), x: qd.types.ndarray(ndim=2), y: qd.types.ndarray(ndim=2),
             m: qd.i32, niter: qd.i32):
    for w in range(A.shape[0]):
        for _ in range(niter):
            for i in range(m):
                s = 0.0
                for j in range(m):
                    s = s + A[w, i, j] * x[w, j]
                y[w, i] = s


@qd.kernel
def k_matvec_s(A: qd.types.ndarray(ndim=3), x: qd.types.ndarray(ndim=2), y: qd.types.ndarray(ndim=2),
               m: qd.i32, niter: qd.i32):
    for w in range(A.shape[0]):
        for _ in range(niter):
            for i in range(m):
                s = 0.0
                loop_config(serialize=True)
                for j in range(m):
                    s = s + A[w, i, j] * x[w, j]
                y[w, i] = s


@qd.kernel
def k_symmatvec(L: qd.types.ndarray(ndim=3), x: qd.types.ndarray(ndim=2), y: qd.types.ndarray(ndim=2),
                m: qd.i32, niter: qd.i32):
    for w in range(L.shape[0]):
        for _ in range(niter):
            for i in range(m):
                s = L[w, i, i] * x[w, i]
                for j in range(i):
                    s = s + L[w, i, j] * x[w, j]
                y[w, i] = s


@qd.kernel
def k_symmatvec_s(L: qd.types.ndarray(ndim=3), x: qd.types.ndarray(ndim=2), y: qd.types.ndarray(ndim=2),
                  m: qd.i32, niter: qd.i32):
    for w in range(L.shape[0]):
        for _ in range(niter):
            for i in range(m):
                s = L[w, i, i] * x[w, i]
                loop_config(serialize=True)
                for j in range(i):
                    s = s + L[w, i, j] * x[w, j]
                y[w, i] = s


@qd.kernel
def k_chol_dot(L: qd.types.ndarray(ndim=3), o: qd.types.ndarray(ndim=3), m: qd.i32, niter: qd.i32):
    for w in range(L.shape[0]):
        for _ in range(niter):
            for i in range(m):
                for j in range(i):
                    s = 0.0
                    for k in range(j):
                        s = s + L[w, i, k] * L[w, j, k]
                    o[w, i, j] = s


@qd.kernel
def k_chol_dot_s(L: qd.types.ndarray(ndim=3), o: qd.types.ndarray(ndim=3), m: qd.i32, niter: qd.i32):
    for w in range(L.shape[0]):
        for _ in range(niter):
            for i in range(m):
                for j in range(i):
                    s = 0.0
                    loop_config(serialize=True)
                    for k in range(j):
                        s = s + L[w, i, k] * L[w, j, k]
                    o[w, i, j] = s


def dev(a):
    t = qd.ndarray(dtype=qd.f32, shape=a.shape)
    t.from_numpy(np.ascontiguousarray(a))
    return t


def dev_zeros(shape):
    return dev(np.zeros(shape, dtype=np.float32))


def max_rel_err(res_f32, ref_f64):
    rel = np.abs(res_f32.astype(np.float64) - ref_f64) / np.maximum(np.abs(ref_f64), 1e-30)
    return float(rel.max())


# --------------------------------------------------------------- child arm ----
if ARM:
    qd.init(arch=qd.metal, spirv_vectorize=(ARM == "vec"), reassociate=(ARM == "rea"),
            offline_cache=False)

    outputs = {}
    errs = {}
    for nv in (24, 23, 5, 1):
        nw = 9
        rng = np.random.default_rng(7 + nv)
        A = rng.standard_normal((nw, nv, nv)).astype(np.float32) * 0.1
        V = rng.standard_normal((nw, nv)).astype(np.float32)
        m3, v2 = dev(A), dev(V)
        o1 = dev_zeros((nw, nv))
        o2 = dev_zeros((nw, nv))
        y = dev_zeros((nw, nv))
        k_dot(m3, v2, o1, nv, 4)
        k_cholrow(m3, v2, o2, nv)
        k_elementwise(v2, y, 0.5, nv)
        qd.sync()
        for name, arr in ((f"k_dot_nv{nv}", o1), (f"k_cholrow_nv{nv}", o2), (f"k_ew_nv{nv}", y)):
            outputs[name] = hashlib.sha256(np.ascontiguousarray(arr.to_numpy()).tobytes()).hexdigest()

    # payoff shapes at nw=1: hash (bitwise contrast) + accuracy vs numpy f64.
    # chol_dot uses the payoff probe's positive PSD recipe (no cancellation ->
    # rel err = pure accumulation rounding, see G5 in the docstring).
    m48, m64 = 48, 64
    rng48 = np.random.default_rng(200 + m48)
    A48 = (rng48.standard_normal((1, m48, m48)) * 0.1).astype(np.float32)
    x48 = rng48.standard_normal((1, m48)).astype(np.float32)
    A64 = np.random.default_rng(4242).uniform(0.5, 1.0, (1, m64, m64)).astype(np.float32)
    L64 = np.ascontiguousarray(np.tril(A64) + np.diag(np.full(m64, float(m64), np.float32)))
    rng64 = np.random.default_rng(300 + m64)
    x64 = rng64.standard_normal((1, m64)).astype(np.float32)
    S64 = np.ascontiguousarray(np.tril((rng64.standard_normal((1, m64, m64)) * 0.1).astype(np.float32)))
    A48d, x48d = dev(A48), dev(x48)
    L64d, x64d = dev(L64), dev(x64)
    S64d = dev(S64)
    y48 = dev_zeros((1, m48))
    y64 = dev_zeros((1, m64))
    o64 = dev_zeros((1, m64, m64))
    k_matvec(A48d, x48d, y48, m48, 1)
    k_symmatvec(S64d, x64d, y64, m64, 1)
    k_chol_dot(L64d, o64, m64, 1)
    qd.sync()
    ref48 = np.einsum("wij,wj->wi", A48.astype(np.float64), x48.astype(np.float64))
    ref64 = np.einsum("wij,wj->wi", S64.astype(np.float64), x64.astype(np.float64))
    Lb = L64.astype(np.float64)
    ref_chol = np.zeros((1, m64, m64))
    for i in range(m64):
        for j in range(i):
            ref_chol[0, i, j] = np.dot(Lb[0, i, :j], Lb[0, j, :j])
    for name, arr, ref in (("matvec_m48", y48, ref48), ("sym_m64", y64, ref64), ("chol_m64", o64, ref_chol)):
        r = arr.to_numpy()
        outputs[name] = hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest()
        errs[name] = max_rel_err(r, ref)

    for name in sorted(outputs):
        print(f"HASH {name} {outputs[name]}")
    for name in sorted(errs):
        print(f"ERR {name} {errs[name]:.6e}")
    sys.exit(0)

# -------------------------------------------------------------- perf child ----
if PERF_ARM:
    # arms:
    #   inproc : reassociate=True -- plain kernels get the reassociated
    #            codegen (A arm); the _s variants are vec_analyze-rejected and
    #            emit plain scalar code in the SAME process (S baseline).
    #            Interleaved A/S with ABA sandwich -> the pinned ratio.
    #   vec    : spirv_vectorize=True -- plain kernels with the flag-OFF (C2
    #            bitwise) emission; microseconds reported for context only
    #            (cross-process, mode-sticky, never compared to `inproc`).
    qd.init(arch=qd.metal, spirv_vectorize=(PERF_ARM == "vec"), reassociate=(PERF_ARM == "inproc"),
            offline_cache=True)
    m48, m64 = 48, 64
    SHAPES = [
        ("matvec_m48", m48, 1024, 8),
        ("sym_m64", m64, 512, 8),
    ]
    if not args.quick:
        SHAPES.append(("chol_m64", m64, 1024, 2))

    rng = np.random.default_rng(999)
    for name, m, nw, niter in SHAPES:
        A = (rng.standard_normal((nw, m, m)) * 0.1).astype(np.float32)
        x = rng.standard_normal((nw, m)).astype(np.float32)
        L = np.ascontiguousarray(np.tril(A))
        if name == "matvec_m48":
            ka, ks = k_matvec, k_matvec_s
            arga = (dev(A), dev(x), dev_zeros((nw, m)), m, niter)
        elif name == "sym_m64":
            ka, ks = k_symmatvec, k_symmatvec_s
            arga = (dev(L), dev(x), dev_zeros((nw, m)), m, niter)
        else:
            ka, ks = k_chol_dot, k_chol_dot_s
            arga = (dev(L), dev_zeros((nw, m, m)), m, niter)
        ka(*arga)
        ks(*arga)
        qd.sync()

        def bench(kd, la, reps):
            t0 = time.perf_counter()
            for _ in range(reps):
                kd(*la)
            qd.sync()
            return (time.perf_counter() - t0) / reps

        t0 = time.perf_counter()
        ka(*arga)
        qd.sync()
        one = time.perf_counter() - t0
        reps = max(1, min(32, int(0.03 / max(one, 1e-6))))
        # GPU clock warmup (~1.5s of alternating bursts) so neither arm measures
        # the ramp; then interleaved A/S burst PAIRS WITH ALTERNATING ORDER
        # (kills first-order drift/ramp bias), no dropping. Deep ~30ms bursts
        # (same estimator as the vec child's per-arm min): machine modes only
        # add time, so the per-arm minimum converges to the true cost and the
        # min/min ratio is robust to the documented bimodal timing (median
        # shown alongside).
        w = time.monotonic()
        while time.monotonic() - w < (0.5 if args.quick else 1.5):
            ka(*arga)
            ks(*arga)
        qd.sync()
        pairs = 6 if args.quick else 8
        ta, ts = [], []
        for i in range(pairs):
            if i % 2 == 0:
                ta.append(bench(ka, arga, reps))
                ts.append(bench(ks, arga, reps))
            else:
                ts.append(bench(ks, arga, reps))
                ta.append(bench(ka, arga, reps))
        r_min = min(ts) / min(ta)
        r_med = statistics.median(ts) / statistics.median(ta)
        print(f"PERF {name} {r_min:.3f} {r_med:.3f} {pairs} {reps} "
              f"{min(ta) * 1e6:.1f} {min(ts) * 1e6:.1f}")
    sys.exit(0)

# ------------------------------------------------------------------ parent ----
QPU = [sys.executable, os.path.abspath(__file__), ARCH]


def run_arm(arm, timeout=900):
    env = dict(os.environ)
    env["QD_REASSOC_ARM"] = arm
    env.pop("QD_REASSOC_PERF_ARM", None)
    r = subprocess.run(QPU, env=env, capture_output=True, text=True, timeout=timeout)
    hashes, errs = {}, {}
    for line in r.stdout.splitlines():
        if line.startswith("HASH "):
            _, name, h = line.split()
            hashes[name] = h
        elif line.startswith("ERR "):
            _, name, e = line.split()
            errs[name] = float(e)
    return hashes, errs, r


def run_perf_child(arm, timeout=900):
    env = dict(os.environ)
    env["QD_REASSOC_PERF_ARM"] = arm
    env.pop("QD_REASSOC_ARM", None)
    extra = ["--quick"] if args.quick else []
    r = subprocess.run(QPU + extra, env=env, capture_output=True, text=True, timeout=timeout)
    out = {}
    for line in r.stdout.splitlines():
        if line.startswith("PERF "):
            _, name, rmin, rmed, pairs, reps, us_a, us_s = line.split()
            out[name] = (float(rmin), float(rmed), int(pairs), int(reps), float(us_a), float(us_s))
    if r.returncode != 0 or not out:
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        return None
    return out


fails = []
warns = []


def gate(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL':4}] {name} {detail}")
    if not ok:
        fails.append(name)


print(f"== GATE reassociate flag (Metal, {time.strftime('%F %T')}) ==")

arms = {}
for arm in ("scalar", "vec", "rea"):
    hashes, errs, r = run_arm(arm)
    arms[arm] = hashes
    if r.returncode != 0 or not hashes:
        gate(f"arm {arm} runs", False, f"exit {r.returncode}")
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        sys.exit(1)
    if arm == "rea":
        rea_errs = errs

if not arms["scalar"].keys() == arms["vec"].keys() == arms["rea"].keys():
    gate("arms cover the same outputs", False)
    sys.exit(1)

# G1: OFF path bitwise unchanged (hard)
g1_bad = [n for n in sorted(arms["scalar"]) if arms["scalar"][n] != arms["vec"][n]]
gate("G1 OFF path bitwise (scalar==vec, all outputs)", not g1_bad,
     "" if not g1_bad else f"mismatch: {g1_bad}")

# G2: reassociation actually happens on qualifying outputs (hard)
rea_outputs = [n for n in sorted(arms["rea"])
               if n.startswith(("k_dot_nv24", "k_dot_nv23", "k_cholrow_nv24", "k_cholrow_nv23",
                                "matvec_m48", "sym_m64", "chol_m64"))]
g2_bad = [n for n in rea_outputs if arms["rea"][n] == arms["vec"][n]]
gate("G2 rea is non-bitwise vs vec on qualifying reductions", not g2_bad,
     "" if not g2_bad else f"unexpectedly identical: {g2_bad}")

# G3: flag-inert anchor (hard)
ew = [n for n in sorted(arms["rea"]) if n.startswith("k_ew_")]
g3_bad = [n for n in ew if not (arms["rea"][n] == arms["vec"][n] == arms["scalar"][n])]
gate("G3 k_elementwise identical in all arms (untouched loop anchor)", not g3_bad,
     "" if not g3_bad else f"flag changed a scalar-path loop: {g3_bad}")

# G4: implication semantics (hard)
g4_bad = [n for n in rea_outputs if arms["rea"][n] == arms["scalar"][n]]
gate("G4 rea!=scalar on qualifying reductions (flag alone implies C2)", not g4_bad,
     "" if not g4_bad else f"vectorization inactive with flag alone: {g4_bad}")

# G5: accuracy vs f64 (hard)
g5_bad = [f"{n}={e:.3e}" for n, e in sorted(rea_errs.items()) if not (e < 1e-5)]
gate("G5 rea arm accuracy < 1e-5 rel (payoff shapes, nw=1)", not g5_bad,
     "; ".join(f"{n} {e:.2e}" for n, e in sorted(rea_errs.items()))
     if not g5_bad else f"violations: {g5_bad}")

# ------------------------------------------------------------------ perf ----
perf_names = ["matvec_m48", "sym_m64"] + ([] if args.quick else ["chol_m64"])


def gpu_idle_or_skip():
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_gpu_busy.sh")
    deadline = time.monotonic() + 300
    while True:
        if subprocess.run(["/bin/zsh", str(script)], capture_output=True).returncode == 0:
            return True
        if time.monotonic() > deadline:
            return False
        print("  gpu busy; perf phase waits (retry 15s, up to 5 min)")
        time.sleep(15)


t_perf0 = time.monotonic()
if not gpu_idle_or_skip():
    warns.append("perf skipped: gpu busy >5 min")
    print("[WARN] perf phase skipped: GPU busy")
else:
    print("\n=== PERF (in-process interleaved ON-codegen[A] vs sequential-scalar[S]; min-based) ===")
    print(f"{'shape':<14}{'A us':>9}{'S us':>9}{'S/A min':>9}{'S/A med':>9}{'vec A us*':>11}")
    print("    (* vec arm = flag-OFF C2-bitwise emission, cross-process: context only)")
    inproc = run_perf_child("inproc")
    vec = run_perf_child("vec")
    if inproc is None:
        warns.append("perf inproc child failed")
        print("[WARN] perf child failed; skipping perf verdict")
    else:
        for n in perf_names:
            if n not in inproc:
                warns.append(f"perf {n}: child did not report")
                print(f"[WARN] {n}: missing from child output")
                continue
            r_min, r_med, pairs, reps, us_a, us_s = inproc[n]
            us_v = vec.get(n, (0, 0, 0, float("nan"), float("nan"), float("nan")))[4] if vec else float("nan")
            print(f"{n:<14}{us_a:>8.0f}{us_s:>9.0f}{r_min:>8.2f}x{r_med:>8.2f}x{us_v:>11.0f}")
            if n in ("matvec_m48", "sym_m64") and r_min < 1.4:
                warns.append(f"perf {n}: S/A min {r_min:.2f}x < 1.4x WARN-only")
                print(f"[WARN] {n} S/A {r_min:.2f}x below the 1.4x payoff pin (WARN-only)")
    print(f"    (perf phase {time.monotonic() - t_perf0:.0f}s)")

print(f"\nREASSOC FLAG GATE: {'ALL HARD GATES PASS' if not fails else 'FAIL: ' + ', '.join(fails)}"
      + ("" if not warns else f" | WARNINGS: {'; '.join(warns)}"))
sys.exit(1 if fails else 0)
