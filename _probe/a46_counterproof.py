"""A46 guard counter-proof: runtime-control-flow recursion must still raise the
inline-cycle error (fast, not OOM); static-terminating recursion passes.

Usage: python a46_counterproof.py  (exit 0 = both behaviours correct)
"""

import sys

import quadrants as qd

qd.init(arch=qd.cpu)

# 1. static-terminating template recursion: allowed (A46 upgrade)
@qd.func
def fact_static(x: qd.template()) -> qd.i32:
    if qd.static(x > 1):
        return x * fact_static(x - 1)
    return 1


@qd.kernel
def k_static(x: qd.template()) -> qd.i32:
    return fact_static(x)


assert k_static(5) == 120, "static recursion broken"
print("static recursion ok: fact(5) = 120")

# 2. runtime recursion (early-return form): rejected at compile time by
# whichever rule fires first (early-return transform or the cycle guard) --
# the A46 contract is "fails fast, never OOMs", not a specific message.
@qd.func
def bad_runtime(x: qd.i32) -> qd.i32:
    if x > 1:
        return x * bad_runtime(x - 1)
    return 1


@qd.kernel
def k_runtime(x: qd.i32) -> qd.i32:
    return bad_runtime(x)


try:
    k_runtime(5)
except Exception as e:  # noqa: BLE001
    print(f"runtime recursion (early-return form) rejected: {type(e).__name__}")
else:
    print("runtime recursion NOT rejected -- guard regression!")
    sys.exit(3)

# 3. runtime recursion WITHOUT early return: this is the form only the cycle
# guard can catch (assignment body, single tail return) -- must raise the
# inline-cycle error.
@qd.func
def bad_runtime_assign(x: qd.i32) -> qd.i32:
    r = 1
    if x > 1:
        r = x * bad_runtime_assign(x - 1)
    return r


@qd.kernel
def k_runtime_assign(x: qd.i32) -> qd.i32:
    return bad_runtime_assign(x)


try:
    k_runtime_assign(5)
except Exception as e:  # noqa: BLE001
    if "Recursive" in str(e) or "cycle" in str(e):
        print("runtime recursion (assign form) rejected by cycle guard:", type(e).__name__)
        sys.exit(0)
    print(f"WRONG ERROR: {type(e).__name__}: {e}")
    sys.exit(2)
print("runtime recursion (assign form) NOT rejected -- guard regression!")
sys.exit(3)
