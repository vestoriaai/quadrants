"""Gate probe (C2): SPIR-V inner-loop vectorization — bitwise scalar-vs-vector parity.

The C2 spirv-vectorize feature rewrites serial innermost range loops with
provably stride-1 ndarray accesses into [align-up scalar prologue] + [float4
vector main loop] + [scalar remainder] on Metal/Vulkan (compile_config
`spirv_vectorize`, default OFF). The acceptance contract is bitwise-identical
results vs the scalar path: loop-carried reductions keep a sequential scalar
accumulator (order-barrier inner lane loop for shapes the Apple MSL compiler
reassociates), and every element is covered by exactly one of the three loops
in the original order.

One-init-per-process forbids both arms in one process, so this probe
re-executes ITSELF twice (env QD_C2_ARM=scalar / vec), each arm printing
sha256 of every output buffer; the parent compares. Covers:
  * fast-path reduction  (pure add/mul over loads: `s += a*b`)
  * barrier reduction    (frontend `s -= X` -> `s + neg(X)` shape, division)
  * elementwise loop     (y[w, j] = a*x[w, j] + b)
  * odd / tiny bounds    (nv in {1, 5, 23} -> prologue + remainder paths)
  * multi-world rows misaligned mod 16 (nv=23: row stride 92B)

Non-metal archs: SKIP, exit 0 (the transform only runs on SPIR-V backends).

Run: python probe_spirv_vectorize.py [metal]   (exit 0 = all bitwise)
"""
import hashlib
import os
import subprocess
import sys

import numpy as np

import quadrants as qd

ARCH = sys.argv[1] if len(sys.argv) > 1 else "metal"
ARM = os.environ.get("QD_C2_ARM", "")

if ARCH != "metal":
    print(f"C2 spirv-vectorize gate arch={ARCH}: SKIP (transform is SPIR-V-only) -> exit 0")
    sys.exit(0)

if not ARM:
    # parent: run both arms, compare
    env = dict(os.environ)
    fails = []
    hashes = {}
    for arm in ("scalar", "vec"):
        e = dict(env)
        e["QD_C2_ARM"] = arm
        r = subprocess.run([sys.executable, os.path.abspath(__file__), ARCH], env=e,
                           capture_output=True, text=True, timeout=600)
        for line in r.stdout.splitlines():
            if line.startswith("HASH "):
                _, name, h = line.split()
                hashes.setdefault(name, {})[arm] = h
        if r.returncode != 0:
            fails.append(f"arm {arm} exited {r.returncode}")
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
    if fails:
        print("C2 SPIRV-VECTORIZE GATE: FAIL --", "; ".join(fails))
        sys.exit(1)
    for name in sorted(hashes):
        hs = hashes[name]
        ok = hs.get("scalar") == hs.get("vec")
        print(f"[{'PASS' if ok else 'FAIL':4}] {name} scalar={hs.get('scalar')} vec={hs.get('vec')}")
        if not ok:
            fails.append(name)
    print(f"\nC2 SPIRV-VECTORIZE GATE arch={ARCH}: "
          f"{'ALL PASS (bitwise scalar==vec)' if not fails else 'FAIL: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)

# ---------------- child arm ----------------
qd.init(arch=qd.metal, spirv_vectorize=(ARM == "vec"), offline_cache=False)


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
    # `s -= a*b` + division: frontend lowers the sub to `s + neg(mul)` and the
    # neg/const shapes take the order-barrier reduction path.
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


def dev(a):
    t = qd.ndarray(dtype=qd.f32, shape=a.shape)
    t.from_numpy(np.ascontiguousarray(a))
    return t


rng = np.random.default_rng(7)
outputs = {}
for nv in (24, 23, 5, 1):
    nw = 9
    A = rng.standard_normal((nw, nv, nv)).astype(np.float32) * 0.1
    V = rng.standard_normal((nw, nv)).astype(np.float32)
    m3, v2 = dev(A), dev(V)
    o1 = dev(np.zeros((nw, nv), dtype=np.float32))
    o2 = dev(np.zeros((nw, nv), dtype=np.float32))
    y = dev(np.zeros((nw, nv), dtype=np.float32))
    k_dot(m3, v2, o1, nv, 4)
    k_cholrow(m3, v2, o2, nv)
    k_elementwise(v2, y, 0.5, nv)
    qd.sync()
    for name, arr in (("k_dot_nv%d" % nv, o1), ("k_cholrow_nv%d" % nv, o2), ("k_ew_nv%d" % nv, y)):
        outputs[name] = hashlib.sha256(np.ascontiguousarray(arr.to_numpy()).tobytes()).hexdigest()

for name in sorted(outputs):
    print(f"HASH {name} {outputs[name]}")
