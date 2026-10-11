"""Gate 4 of the robostudio post-install regression: readback R2/R3 perf + correctness.

Extracted verbatim (as a standalone file) from the gate-4 heredoc of
robostudio worktree scratch/quadrants_c0s/post_install_gate.sh during the
2026-10-11 consolidation (docs/consolidation_robostudio_main.md).

Pins (from simtest qd-render-poc results/readback_r2r3_report.md):
  to_numpy   64MB  <= 7.0 ms   (report 5.561, overlay wheel 5.435)
  from_numpy 64MB  <= 5.0 ms   (report 3.239, overlay wheel 3.432)
plus a byte-correctness sweep (1KB/4MB/64MB i32 vs numpy reference) on
arch=metal.

Pin policy (2026-10-11, supersedes gate-failing absolute pins): the wall-clock
pins are WARN-only. Gate-4 failed spuriously across rebuilds at up to +-40%
machine variance while correctness stayed green, so the gate verdict now keys
on the byte-correctness sweep only; the perf lines print measured-vs-pin for
the ledger / human diffing. Real transfer-perf regressions are caught by the
per-kernel A/B benches of the line that caused them (e.g. spirv-vectorize),
not by this I/O gate.

Verdict: prints "RESULT: PASS" and exits 0, or "RESULT: FAIL [...]" / exit 1.
"""
import sys
import time

import numpy as np

import quadrants as qd

qd.init(arch=qd.metal)

TO_64MB_MAX_MS = 7.0
FROM_64MB_MAX_MS = 5.0

fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        fails.append(name)


# byte-correctness sweep: 1KB / 4MB / 64MB, i32, vs numpy reference
for label, nbytes in (("1KB", 1024), ("4MB", 4 * 1024 * 1024), ("64MB", 64 * 1024 * 1024)):
    n = nbytes // 4
    host = ((np.arange(n, dtype=np.int64) * 2654435761 + 12345) % (1 << 31)).astype(np.int32)
    arr = qd.ndarray(qd.i32, (n,))
    arr.from_numpy(host)
    back = arr.to_numpy()
    check(f"byte-identical {label} i32 from_numpy->to_numpy", np.array_equal(back, host),
          f"{n} elems")

# perf: 64MB, warmup + min-of-15 (same full-chain convention as the report)
def best_ms(fn, k=15):
    for _ in range(2):
        fn()
    qd.sync()
    best = float("inf")
    for _ in range(k):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


n = 64 * 1024 * 1024 // 4
host = np.zeros(n, np.int32)
host[::4096] = 7
arr = qd.ndarray(qd.i32, (n,))
arr.from_numpy(host)
qd.sync()
best_to_ms = best_ms(lambda: arr.to_numpy())
best_from_ms = best_ms(lambda: arr.from_numpy(host))
print(f"  to_numpy   64MB best-of-15 = {best_to_ms:.3f} ms (pin: report 5.561 / overlay 5.435)")
print(f"  from_numpy 64MB best-of-15 = {best_from_ms:.3f} ms (pin: report 3.239 / overlay 3.432)")
# WARN-only: machine variance up to +-40% was observed on identical code, so
# these never fail the gate (see docstring pin-policy note).
for name, got, pin in (("to_numpy 64MB", best_to_ms, TO_64MB_MAX_MS),
                       ("from_numpy 64MB", best_from_ms, FROM_64MB_MAX_MS)):
    mark = "ok" if got <= pin else f"WARN over pin x{got / pin:.2f}"
    print(f"  WARN-CHECK {name}: {mark}")

print("RESULT:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(0 if not fails else 1)
