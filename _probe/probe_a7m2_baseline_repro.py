"""A7/M2 root-fix evidence: minimal repros for the two early_return=True defects.

Defect 1 (compound slots): slot init `qd.cast(0, T)` with T=vector/matrix produces a
scalar-operand cast of tensor-declared slot -> malformed IR; CPU dies in LLVM codegen
("Cannot cast between tensor type and non-tensor type"), Metal emits invalid SPIR-V.

Defect 2 (sibling guards): every post-return statement gets its OWN `if __er_done == 0:`
block; a name first-bound inside one runtime-if block is retired at VariableScopeGuard
exit and the sibling block's read hits the actionable branch-scope NameError (A20 family).

Run with any pre-fix wheel:  <venv>/bin/python probe_a7m2_baseline_repro.py [cpu|metal]
Exit 0 iff BOTH defects reproduce with the expected error signatures (pre-fix wheel),
or BOTH compile+run clean (post-fix wheel). Anything else = unexplained, exit 1.
"""
import sys

import numpy as np

import quadrants as qd
from quadrants.lang.exception import QuadrantsSyntaxError

ARCH = sys.argv[1] if len(sys.argv) > 1 else "cpu"
qd.init(arch=qd.metal if ARCH == "metal" else qd.cpu)

results = []


def expect_error(name, fn, needles):
    """Compile+run fn; record ('error', matched-needle) or ('ok', None)."""
    try:
        fn()
        qd.sync()
        results.append((name, "ok", None))
    except QuadrantsSyntaxError as e:
        msg = str(e)
        hit = next((n for n in needles if n in msg), None)
        results.append((name, "error", hit if hit else f"WRONG-ERR {msg[:120]}"))
    except Exception as e:  # noqa: BLE001
        results.append((name, "error", f"WRONG-TYPE {type(e).__name__}: {str(e)[:120]}"))


# ---- defect 1: compound slot init ------------------------------------------
vec3 = qd.types.vector(3, qd.f32)


@qd.func(early_return=True)
def f_compound_tup(x: qd.f32) -> tuple[vec3, qd.f32]:
    if x < 0.0:
        return vec3([1.0, 0.0, 0.0]), -1.0
    return vec3([0.0, x, 0.0]), x * 2.0


@qd.kernel
def k_d1(xs: qd.types.ndarray(ndim=1), ys: qd.types.ndarray(ndim=1), zs: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        v, s = f_compound_tup(xs[i])
        for j in range(3):
            ys[i * 3 + j] = v[j]
        zs[i] = s


def run_d1():
    xs = qd.ndarray(dtype=qd.f32, shape=(2,))
    ys = qd.ndarray(dtype=qd.f32, shape=(6,))
    zs = qd.ndarray(dtype=qd.f32, shape=(2,))
    xs.from_numpy(np.array([-1.0, 2.0], dtype=np.float32))
    ys.from_numpy(np.zeros(6, dtype=np.float32))
    zs.from_numpy(np.zeros(2, dtype=np.float32))
    k_d1(xs, ys, zs)
    qd.sync()


expect_error(
    "d1_compound_slot_init",
    run_d1,
    ["Cannot cast between tensor type and non-tensor type", "cast", "tensor"],
)

# ---- defect 2: sibling-guard first-binding ---------------------------------
@qd.func(early_return=True)
def f_sibling(x: qd.f32) -> qd.f32:
    if x < 0.0:
        return -1.0
    t = x * 0.5  # first binding of t lands in the merged/sibling guard region
    return t + 1.0


@qd.kernel
def k_d2(xs: qd.types.ndarray(ndim=1), ys: qd.types.ndarray(ndim=1)):
    for i in range(xs.shape[0]):
        ys[i] = f_sibling(xs[i])


def run_d2():
    xs = qd.ndarray(dtype=qd.f32, shape=(3,))
    ys = qd.ndarray(dtype=qd.f32, shape=(3,))
    xs.from_numpy(np.array([-1.0, 0.5, 4.0], dtype=np.float32))
    ys.from_numpy(np.zeros(3, dtype=np.float32))
    k_d2(xs, ys)
    qd.sync()
    got = ys.to_numpy()
    want = [-1.0, 0.5 * 0.5 + 1.0, 4.0 * 0.5 + 1.0]
    assert np.allclose(got, want, atol=1e-6), f"numeric mismatch {got} vs {want}"


expect_error("d2_sibling_guard_binding", run_d2, ["retired", "is not defined", "NameError", "not defined"])

# ---- report -----------------------------------------------------------------
print(f"\n=== A7/M2 baseline repro arch={ARCH} ===")
for name, kind, detail in results:
    print(f"  {name:28s} -> {kind:5s} {detail or ''}")

kinds = {name: kind for name, kind, _ in results}
if kinds == {"d1_compound_slot_init": "error", "d2_sibling_guard_binding": "error"}:
    details_ok = all(d is not None and not d.startswith(("WRONG", "WRONG")) for _, _, d in results)
    print("PRE-FIX WHEEL: both defects reproduce" + ("" if details_ok else " (but with unexpected error text)"))
    sys.exit(0 if details_ok else 1)
if kinds == {"d1_compound_slot_init": "ok", "d2_sibling_guard_binding": "ok"}:
    print("POST-FIX WHEEL: both repro forms compile and pass numerics")
    sys.exit(0)
print("UNEXPLAINED state: mixed pass/error or wrong errors")
sys.exit(1)
