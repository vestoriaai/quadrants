"""Gate 8 of the robostudio union suite: R5 host-visible ndarray allocation.

qd.ndarray(..., host_visible=True) allocates the backing buffer host-mappable
(Metal MTLStorageModeShared, via the existing host_read=>Shared rule), so
to_numpy/from_numpy skip the bridge-kernel copy (the ~0.131ms/call launch tax
isolated in the R2/R3 report) and become drain + map + memcpy. Default
ndarrays keep the exact pre-R5 behavior (Private on Metal + bridge kernel).

Checks (see docs/robostudio_features.md "R5" entry):
  C1 byte-parity sweep: host_visible round-trip (from_numpy -> to_numpy) equals
     both the numpy reference and the default ndarray's output, i32/f32/u8 x
     1KB/1MB/64MB.
  C2 obs pattern: user kernel writes a host_visible ndarray -> to_numpy fast
     path observes them (sync-before-memcpy ordering).
  C3 upload pattern: from_numpy fast path -> subsequently launched kernel reads
     the host write (host-write-before-enqueue visibility).
  C4 matrix/vector shapes (always AOS) through the same fast path, plus a
     kernel write on a vector ndarray.
  C5 knob introspection: default = (_qd_host_visible False, C++ buffer not
     host-mappable); opt-in reports True on both. needs_grad grad buffer stays
     default (not host-visible).
  P  perf A/B (metal only): same-process interleaved min-of-k, default vs
     host_visible at 256B/64KB/1MB/64MB. Relative pins only (gate-4 lesson:
     absolute ms pins are machine-variance-fragile): fast path <= 0.5x default
     on small buffers (bridge-kernel launch tax removed), <= 1.2x at 64MB.

Verdict: prints "RESULT: PASS" and exits 0, or "RESULT: FAIL [...]" / exit 1.

Usage: python gate_r5_host_visible.py [--arch metal|cpu]   (default metal)
"""
import sys
import time

import numpy as np

import quadrants as qd
from quadrants.lang.impl import grouped  # kernel-scope name resolution (as _kernels.py does)

ARCH = qd.metal
args = sys.argv[1:]
if "--arch" in args:
    ARCH = {"metal": qd.metal, "cpu": qd.cpu}[args[args.index("--arch") + 1]]
qd.init(arch=ARCH)

IS_METAL = ARCH == qd.metal
fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        fails.append(name)


def host_u32(n, seed=12345):
    idx = np.arange(n, dtype=np.int64)
    return ((idx * 2654435761 + seed) % (1 << 31)).astype(np.int32)


def host_f32(n, seed=12345):
    # NaN-free deterministic floats in [0, 4)
    return ((np.arange(n, dtype=np.int64) * 2654435761 + seed) % (1 << 22)).astype(np.float32) / np.float32(1 << 20)


# ---------------------------------------------------------------- C1 sweep
for label, nbytes in (("1KB", 1024), ("1MB", 1024 * 1024), ("64MB", 64 * 1024 * 1024)):
    for dt, np_dt in ((qd.i32, np.int32), (qd.f32, np.float32), (qd.u8, np.uint8)):
        item = np.dtype(np_dt).itemsize
        n = max(1, nbytes // item)
        # bitwise comparison everywhere (int32 bit patterns cover fp NaN payloads)
        host = host_u32((n * item + 3) // 4).view(np.uint8)[: n * item].view(np_dt)
        arr_def = qd.ndarray(dt, n)
        arr_def.from_numpy(host)
        back_def = arr_def.to_numpy()
        arr_hv = qd.ndarray(dt, n, host_visible=True)
        arr_hv.from_numpy(host)
        back_hv = arr_hv.to_numpy()
        check(
            f"C1 byte-identical {label} {np_dt.__name__} hv-vs-default-vs-numpy",
            back_hv.tobytes() == host.tobytes() and back_hv.tobytes() == back_def.tobytes(),
            f"{n} elems",
        )

# ------------------------------------------------- C2 kernel writes obs buffer
n = 4096
host = host_f32(n)
arr = qd.ndarray(qd.f32, n, host_visible=True)
arr.from_numpy(host)


@qd.kernel
def k_scale(xs: qd.types.ndarray(), val: qd.f32):
    for I in grouped(xs):
        xs[I] = xs[I] * val


k_scale(arr, 2.0)
qd.sync()
got = arr.to_numpy()
check("C2 kernel-write visible to to_numpy fast path", np.allclose(got, host * 2.0, rtol=0, atol=1e-6), f"n={n}")

# ------------------------------------------- C3 from_numpy -> kernel reads it
src = qd.ndarray(qd.f32, n, host_visible=True)
src.from_numpy(host)
dst = qd.ndarray(qd.f32, n)  # default destination exercises the bridge readback


@qd.kernel
def k_copy(src: qd.types.ndarray(), dst: qd.types.ndarray()):
    for I in grouped(src):
        dst[I] = src[I]


k_copy(src, dst)
got = dst.to_numpy()
check("C3 from_numpy fast path visible to next kernel", np.allclose(got, host, rtol=0, atol=1e-6), f"n={n}")

# ------------------------------------------------------- C4 matrix / vector
host_v = host_f32(10 * 2 * 3, seed=777)
vec = qd.ndarray(qd.types.vector(3, qd.f32), (10, 2), host_visible=True)
vec.from_numpy(host_v.reshape(10, 2, 3))
check("C4 vector ndarray round trip", np.array_equal(vec.to_numpy(), host_v.reshape(10, 2, 3)))

k_scale(vec, 2.0)
qd.sync()
check("C4 vector ndarray kernel-write round trip", np.allclose(vec.to_numpy(), host_v.reshape(10, 2, 3) * 2.0, rtol=0, atol=1e-6))

host_m = host_f32(4 * 5 * 3 * 4, seed=888)
mat = qd.ndarray(qd.types.matrix(3, 4, qd.f32), (4, 5), host_visible=True)
mat.from_numpy(host_m.reshape(4, 5, 3, 4))
check("C4 matrix ndarray round trip", np.array_equal(mat.to_numpy(), host_m.reshape(4, 5, 3, 4)))

# --------------------------------------------------------- C5 introspection
plain = qd.ndarray(qd.f32, 16)
hv = qd.ndarray(qd.f32, 16, host_visible=True)
check(
    "C5 default ndarray untouched",
    plain._qd_host_visible is False and plain.arr.is_host_visible() is False,
)
check("C5 opt-in ndarray flagged", hv._qd_host_visible is True and hv.arr.is_host_visible() is True)

g = qd.ndarray(qd.f32, 16, needs_grad=True, host_visible=True)
check("C5 grad buffer stays default", g.grad is not None and g.grad._qd_host_visible is False)

# to_numpy(dtype=...) still applies the cast on the fast path
cast = hv.to_numpy(dtype=np.float64)
check("C5 to_numpy(dtype) cast on fast path", cast.dtype == np.float64)

# ------------------------------------------------------------- P perf A/B
if IS_METAL:
    print("-- perf A/B (metal, interleaved min-of-k; relative pins, absolute numbers for the ledger) --")

    def best_ms(fn, k):
        best = float("inf")
        for _ in range(k):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best * 1e3

    perf_fail = []
    for label, nbytes in (("256B", 256), ("64KB", 64 * 1024), ("1MB", 1024 * 1024), ("64MB", 64 * 1024 * 1024)):
        n = nbytes // 4
        host = np.zeros(n, np.float32)
        a_def = qd.ndarray(qd.f32, n)
        a_hv = qd.ndarray(qd.f32, n, host_visible=True)
        a_hv.from_numpy(host)
        qd.sync()
        k = 15 if nbytes <= 1024 * 1024 else 8
        acc = {("to", "def"): float("inf"), ("to", "hv"): float("inf"),
               ("fr", "def"): float("inf"), ("fr", "hv"): float("inf")}
        for _ in range(4):  # interleaved rounds: A,B,A,B per round, min across rounds
            for arm, arr in (("def", a_def), ("hv", a_hv)):
                acc[("to", arm)] = min(acc[("to", arm)], best_ms(lambda a=arr: a.to_numpy(), k))
                acc[("fr", arm)] = min(acc[("fr", arm)], best_ms(lambda a=arr, h=host: a.from_numpy(h), k))
        r_to = acc[("to", "hv")] / max(acc[("to", "def")], 1e-9)
        r_fr = acc[("fr", "hv")] / max(acc[("fr", "def")], 1e-9)
        pin = 0.5 if nbytes <= 1024 * 1024 else 1.2
        ok = r_to <= pin and r_fr <= pin
        if not ok:
            perf_fail.append(label)
        print(
            f"  {label:>5}: to def={acc[('to', 'def')]:7.3f}ms hv={acc[('to', 'hv')]:7.3f}ms (x{r_to:.2f})"
            f" | from def={acc[('fr', 'def')]:7.3f}ms hv={acc[('fr', 'hv')]:7.3f}ms (x{r_fr:.2f})"
            f"  pin<=x{pin:.1f} {'ok' if ok else 'FAIL'}"
        )
    check("P perf relative pins (small <=x0.5, 64MB <=x1.2)", not perf_fail, f"failing sizes: {perf_fail}")

print("RESULT:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(0 if not fails else 1)
