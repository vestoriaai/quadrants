#!/usr/bin/env python3
"""S2 probe: assignment to a runtime loop variable must fail LOUDLY and cleanly.

Defect (vendored quadrants 1.3.3, upstream taichi lineage): `for t in range(n):`
binds `t` to a bare id expression that lowers to LoopIndexStmt
(quadrants/transforms/lower_ast.cpp:315).  A body assignment `t = ...` lowered
into LocalStoreStmt(LoopIndexStmt, ...) and tripped the dest-type assert at
quadrants/ir/statements.h:837 (cryptic RuntimeError); the augmented form
`t += ...` died even earlier in frontend eval with an uncaught C++ exception
(quadrants/ir/type.h:97 `Cannot treat i32 as PointerType`) that ABORTED the
process (exit 134) -- the "segfaults the JIT" symptom class.

The S2 python-frontend fix (patches/s2_fix.patch) rejects these at AST time
with a clear QuadrantsSyntaxError.  This probe pins the whole family:

  broken family (must raise QuadrantsSyntaxError):
    A  range-for direct assign, in-body        B  same, inside a runtime if
    C  range-for augmented assign              D  struct-for target assign
  working family (must keep working):
    E  func-local shadow of a caller loop var  F  while-loop counter
    G  static-for loop var rebind              H  plain shadow rebinding
    I  ndrange target assign                   J  grouped-ndrange target assign

Usage: probe_loopvar_assign.py [arch]   (default cpu; GPU not required)
Verdict: RESULT: PASS / RESULT: FAIL
"""
import sys

import quadrants as qd

qd.init(arch=getattr(qd, sys.argv[1] if len(sys.argv) > 1 else "cpu"))

HEAD = """import numpy as np, quadrants as qd
qd.init(arch=qd.cpu)
n = 4
out = qd.field(dtype=qd.i32, shape=16)
partial = np.zeros((n, 1), np.float32)
"""
TAIL = """
Tiny().k(out, partial, n)
print("OK")
"""

CASES = {
    "A_rangefor_assign": """
@qd.data_oriented
class Tiny:
    @qd.kernel
    def k(self, out: qd.template(), partial: qd.types.ndarray(), n: int) -> None:
        for t in range(n):
            t = 0
""",
    "B_rangefor_assign_in_if": """
@qd.data_oriented
class Tiny:
    @qd.kernel
    def k(self, out: qd.template(), partial: qd.types.ndarray(), n: int) -> None:
        for t in range(n):
            if t > 1:
                t = 0
""",
    "C_rangefor_augassign": """
@qd.data_oriented
class Tiny:
    @qd.kernel
    def k(self, out: qd.template(), partial: qd.types.ndarray(), n: int) -> None:
        for t in range(n):
            t += 1
""",
    "D_structfor_assign": """
@qd.data_oriented
class Tiny:
    @qd.kernel
    def k(self, out: qd.template(), partial: qd.types.ndarray(), n: int) -> None:
        for i in out:
            i = 3
""",
    "E_func_shadow": """
@qd.func
def f(t: int) -> int:
    t = t + 1
    return t

@qd.data_oriented
class Tiny:
    @qd.kernel
    def k(self, out: qd.template(), partial: qd.types.ndarray(), n: int) -> None:
        for t in range(n):
            partial[t, 0] = f(t)
""",
    "F_while_counter": """
@qd.data_oriented
class Tiny:
    @qd.kernel
    def k(self, out: qd.template(), partial: qd.types.ndarray(), n: int) -> None:
        t = 0
        while t < n:
            t = t + 1
        partial[0, 0] = t
""",
    "G_staticfor_rebind": """
@qd.data_oriented
class Tiny:
    @qd.kernel
    def k(self, out: qd.template(), partial: qd.types.ndarray(), n: int) -> None:
        for t in qd.static(range(4)):
            t = t + 1
            out[t - 1] = t
""",
    "H_shadow_rebind": """
@qd.data_oriented
class Tiny:
    @qd.kernel
    def k(self, out: qd.template(), partial: qd.types.ndarray(), n: int) -> None:
        for t in range(n):
            tt = t
            tt = tt * 2
            out[t] = tt
""",
    "I_ndrange_target": """
@qd.data_oriented
class Tiny:
    @qd.kernel
    def k(self, out: qd.template(), partial: qd.types.ndarray(), n: int) -> None:
        for i in qd.ndrange(4):
            i = i + 1
            out[i - 1] = i
""",
    "J_grouped_ndrange_target": """
@qd.data_oriented
class Tiny:
    @qd.kernel
    def k(self, out: qd.template(), partial: qd.types.ndarray(), n: int) -> None:
        for I in qd.grouped(qd.ndrange(4)):
            I = 0
""",
}

# cases whose kernels must COMPILE and RUN (the language allows these today)
MUST_WORK = {"E_func_shadow", "F_while_counter", "H_shadow_rebind",
             "I_ndrange_target", "J_grouped_ndrange_target"}
# static-for rebind: already rejected pre-S2 (python int has no _assign);
# pin its pre-existing loud error so the fix does not silently change it.
PREEXISTING_REJECT = {"G_staticfor_rebind"}
MUST_REJECT = {"A_rangefor_assign", "B_rangefor_assign_in_if",
               "C_rangefor_augassign", "D_structfor_assign"}

import subprocess  # noqa: E402  (after qd.init banner, keep output tidy)

VENV = sys.executable
fails = []
for name in sorted(CASES):
    path = f"/tmp/_probe_loopvar_{name}.py"
    with open(path, "w") as fh:
        fh.write(HEAD + CASES[name] + TAIL)
    r = subprocess.run([VENV, "-u", path], capture_output=True, text=True)
    out = r.stdout + r.stderr
    clean_reject = "Cannot assign to loop variable" in out and "QuadrantsSyntaxError" in out
    if name in MUST_REJECT:
        ok = clean_reject
    elif name in PREEXISTING_REJECT:
        ok = ("cannot be assigned" in out) and "libc++abi" not in out
    else:
        ok = "\nOK" in out and "libc++abi" not in out and "Assertion failure" not in out
    print(f"  {name:28s} {'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(name)

print("RESULT:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(0 if not fails else 1)
