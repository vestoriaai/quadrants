"""A34 gate probe: bool(<float>) compiles to C truthiness on both backends.

Pre-fix behavior this gate pins against (ledger A34):
  - cpu:  cast f32->u1 lowered to ICmpNE(x, null) -> LLVM AssertOK abort (exit 134
    on assertions-on wheels) / illegal IR on release builds;
  - metal: loud rejection "do not support type cast from f32 to u1".
Post-fix contract (C semantics, explicit UNORDERED compare):
  - bool(f): NaN -> True; int/ptr path unchanged (CreateIsNotNull stays legal);
  - implicit truthiness 'if <float>:' stays loudly rejected (QuadrantsTypeError).

Usage: python probe_bool_cast.py [cpu|metal]   (default cpu)
Prints "BOOL CAST GATE [arch]: ALL PASS" on success; nonzero exit otherwise.
"""
import subprocess
import sys
import tempfile

import numpy as np

import quadrants as qd

ARCH = sys.argv[1] if len(sys.argv) > 1 else "cpu"
qd.init(arch=getattr(qd, ARCH), offline_cache=False, cfg_optimization=False)

fails = []


def check(name, ok, detail=""):
    print(f"  [{ARCH}] {name:44s} {'PASS' if ok else 'FAIL'} {detail}")
    if not ok:
        fails.append(name)


def run_case(name, dtype_in, x_np, want):
    arr = qd.ndarray(dtype_in, (1, 4))
    arr.from_numpy(np.expand_dims(x_np, 0))
    out = qd.ndarray(bool, (1, 4))

    @qd.kernel
    def k(x_in: qd.types.ndarray(dtype=dtype_in, ndim=2), o: qd.types.ndarray(dtype=bool, ndim=2), __n0: int):
        for __t in range(__n0):
            for j in range(4):
                o[__t, j] = bool(x_in[__t, j])

    k(arr, out, 1)
    qd.sync()
    got = out.to_numpy()[0]
    check(name, (got == np.array(want, dtype=bool)).all(), f"got {got.tolist()}")


X_F = np.array([0.0, 1.5, np.nan, -2.0], dtype=np.float32)

# 1. int->bool unchanged (CreateIsNotNull path stays legal)
run_case("int2bool", qd.i32, np.array([0, 3, -2, 0], dtype=np.int32), [False, True, True, False])

# 2. float->bool: the fixed path (was abort / loud reject)
run_case("flt2bool", qd.f32, X_F, [False, True, True, True])

# 3. NaN truthiness: C semantics NaN != 0 -> True (unordered compare, NOT ordered)
run_case("nan2bool", qd.f32, X_F, [False, True, True, True])

# 4. implicit truthiness stays loudly rejected (subprocess: the rejection is a
# compile-time raise in a fresh interpreter so a pre-fix abort cannot mask it)
CODE = r"""
import numpy as np
import quadrants as qd
qd.init(arch=qd.cpu, offline_cache=False, cfg_optimization=False)
arr = qd.ndarray(qd.f32, (1, 4))
arr.from_numpy(np.zeros((1, 4), dtype=np.float32))
out = qd.ndarray(qd.f32, (1, 4))
@qd.kernel
def k(x_in: qd.types.ndarray(dtype=qd.f32, ndim=2), o: qd.types.ndarray(dtype=qd.f32, ndim=2), __n0: int):
    for __t in range(__n0):
        v = 0.0
        if x_in[__t, 0]:
            v = 1.0
        o[__t, 0] = v
k(arr, out, 1)
"""
with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
    f.write(CODE)
    tmp = f.name
r = subprocess.run([sys.executable, tmp], capture_output=True, text=True)
rejected = r.returncode != 0 and "QuadrantsTypeError" in (r.stdout + r.stderr)
check("ifflt loud rejection", rejected, f"rc={r.returncode}")

if fails:
    print(f"BOOL CAST GATE [{ARCH}]: FAIL ({len(fails)} case(s))")
    sys.exit(1)
print(f"BOOL CAST GATE [{ARCH}]: ALL PASS")
