"""S3 probe: form matrix for the four documented frontend limitations.
Each case runs in a subprocess (one crash must not mask the others).

Case families:
  J1  static-if assigns name, name used AFTER join, inside runtime loop
  J2  static-if with return in func called from runtime loop (probe_static_return)
  J3  static-if assigns inside DOUBLE nested runtime loop
  B1  bool literal init then runtime reassign (issue 3: "acc=False -> NameError")
  B2  runtime comparison bool reused across branch join
  B3  bool literal branch on runtime value
  M1  qd.min/max in flat runtime loop
  M2  qd.min/max in nested runtime loops with loop-carried accumulator
  M3  qd.sqrt in nested runtime loop (signed distance form)
  R1  scalar reassign inside runtime-if (S2 guard expected: loud error)
"""
import subprocess
import sys

TEMPLATE = """import numpy as np
import quadrants as qd

qd.init(arch=qd.cpu)

{body}

x = np.array([0.25, 2.0, -1.0], np.float32)
k(x, 3)
print("CASE {name} OK:", x.tolist())
"""

CASES = {}

CASES["J1_staticif_join"] = """
@qd.data_oriented
class P:
    F = True

    @qd.kernel
    def k(self, x: qd.types.ndarray(), n: int) -> None:
        for i in range(n):
            if qd.static(self.F):
                y = x[i] + 1.0
            else:
                y = x[i] + 2.0
            x[i] = y

k = P().k
"""

CASES["J2_staticif_return"] = """
@qd.data_oriented
class P:
    @qd.func
    def pick(self, a, b):
        if qd.static(True):
            return a * 2.0
        else:
            return b * 3.0

    @qd.kernel
    def k(self, x: qd.types.ndarray(), n: int) -> None:
        for i in range(n):
            x[i] = self.pick(x[i], x[i])

k = P().k
"""

CASES["J3_staticif_nested"] = """
@qd.data_oriented
class P:
    F = True

    @qd.kernel
    def k(self, x: qd.types.ndarray(), n: int) -> None:
        for i in range(n):
            for j in range(2):
                if qd.static(self.F):
                    y = x[i] + 1.0
                x[i] = y

k = P().k
"""

CASES["B1_bool_reassign"] = """
@qd.kernel
def k(x: qd.types.ndarray(), n: int) -> None:
    for i in range(n):
        take = False
        if x[i] > 0.0:
            take = True
        if take:
            x[i] = 1.0
"""

CASES["B2_cmp_reuse"] = """
@qd.kernel
def k(x: qd.types.ndarray(), n: int) -> None:
    for i in range(n):
        take = x[i] > 0.0
        if take:
            x[i] = 1.0
        else:
            x[i] = 0.5
"""

CASES["B3_literal_branch"] = """
@qd.kernel
def k(x: qd.types.ndarray(), n: int) -> None:
    for i in range(n):
        if qd.static(True):
            x[i] = x[i] + 1.0
        else:
            x[i] = x[i] - 1.0
"""

CASES["M1_minmax_flat"] = """
@qd.kernel
def k(x: qd.types.ndarray(), n: int) -> None:
    for i in range(n):
        x[i] = qd.min(qd.max(x[i], 0.0), 1.5)
"""

CASES["M2_minmax_nested"] = """
@qd.kernel
def k(x: qd.types.ndarray(), n: int) -> None:
    acc = 0.0
    for i in range(n):
        for j in range(2):
            acc = qd.max(acc, x[i] + j)
        x[i] = qd.min(x[i], acc)
"""

CASES["M3_sqrt_nested"] = """
@qd.kernel
def k(x: qd.types.ndarray(), n: int) -> None:
    for i in range(n):
        for j in range(2):
            m = x[i] * x[i] - 1.0
            d = qd.sqrt(m * m)
            x[i] = d * 0.0 + x[i]
"""

CASES["R1_reassign_if"] = """
@qd.kernel
def k(x: qd.types.ndarray(), n: int) -> None:
    for i in range(n):
        s = x[i]
        if x[i] > 0.0:
            s = 0.0
        x[i] = s
"""

# R1 is a regular (non-loop-variable) scalar reassigned inside a runtime if:
# it must keep working with CORRECT values -- the "scalar reassignment hard-
# asserts" claim in self_collision_kernels' docstring was the loop-variable
# `t` collision (S2 root cause), not branch reassignment in general.
EXPECTED = {
    "J1_staticif_join":    [1.25, 3.0, 0.0],   # static branch y = x + 1.0
    "J2_staticif_return":  [0.5, 4.0, -2.0],   # pick -> a * 2.0
    "J3_staticif_nested":  [2.25, 4.0, 1.0],   # x += 1.0 applied twice
    "B1_bool_reassign":    [1.0, 1.0, -1.0],
    "B2_cmp_reuse":        [1.0, 1.0, 0.5],
    "B3_literal_branch":   [1.25, 3.0, 0.0],
    "M1_minmax_flat":      [0.25, 1.5, 0.0],
    "M2_minmax_nested":    [0.25, 2.0, -1.0],
    "M3_sqrt_nested":      [0.25, 2.0, -1.0],
    "R1_reassign_if":      [0.0, 0.0, -1.0],
}

only = sys.argv[1] if len(sys.argv) > 1 else None
n_pass = n_fail = 0
for name, body in CASES.items():
    if only and name != only:
        continue
    gen = f"/tmp/_s3_form_{name}.py"
    with open(gen, "w") as fh:
        fh.write(TEMPLATE.format(body=body, name=name))
    r = subprocess.run([sys.executable, "-u", gen], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    last = out[-1] if out else "(no output)"
    rc = r.returncode
    ok = rc == 0 and f"CASE {name} OK" in "\n".join(out)
    try:
        vals = eval(last.split("OK:", 1)[1].strip(), {}, {}) if "OK:" in last else None
    except Exception:
        vals = None
    want = EXPECTED[name]
    values_ok = vals is not None and vals[:3] == want
    verdict = "PASS" if (ok and values_ok) else f"FAIL (want {want})"
    n_pass += verdict == "PASS"
    n_fail += verdict != "PASS"
    print(f"{name:22s} {verdict:26s} :: {last[:120]}")
print(f"RESULT: {'PASS' if n_fail == 0 else 'FAIL'} ({n_pass}/{n_pass + n_fail} cases)")
