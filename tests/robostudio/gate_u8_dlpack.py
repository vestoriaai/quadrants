"""Phase 1 acceptance gates for qd u8 DLPack export (QD_UPSTREAM_PLAN.md R4 / §3.3).

Usage: python u8_dlpack_gate.py --arch metal|cpu
(qd.init is once-per-process and arch-irreversible, so each arch runs in its own process.)

Gates:
  G1  type matrix: u8/i8/u16/i16/u32/i32/i64/f32/f64/u64 to_dlpack -> torch.from_dlpack
      round-trip equality both directions (qd->torch write visibility and torch->qd
      write visibility), on metal (mps:0) and cpu (cpu:0) backends.
  G2  zero-copy: wrap time ~constant across 1KB..64MB (a hidden copy would scale
      with size); absolute wrap cost at the DLPack floor.
  G3  device reported faithfully: mps:0 from metal export (not silently copied to
      host), cpu:0 on cpu backend.
  G4  contract identical to i32/f32 (#12): the alias tensor views the qd buffer --
      buffer reuse through from_numpy silently changes alias contents.
  G5  u8 end-to-end mini sample: qd kernel writes u8 (N,3) -> to_dlpack ->
      from_dlpack -> data equality (qd u8 write is mod-256 wrap; values kept <=255).
"""
import argparse
import time

import numpy as np

import quadrants as qd

parser = argparse.ArgumentParser()
parser.add_argument("--arch", default="metal", choices=["metal", "cpu"])
args = parser.parse_args()

qd.init(arch=getattr(qd, args.arch))
ARCH = args.arch
EXPECT_DEVICE = "mps" if ARCH == "metal" else "cpu"

failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def consumer_read(t):
    """Read a torch alias to host, forcing MPS queue order first on metal."""
    import torch

    if ARCH == "metal" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    return t.cpu().numpy()


def wrap_time_ms(arr, k=30):
    import torch

    torch.from_dlpack(arr.to_dlpack())  # warmup
    best = 1e9
    for _ in range(k):
        t0 = time.perf_counter()
        torch.from_dlpack(arr.to_dlpack())
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def device_ok(t):
    if EXPECT_DEVICE == "mps":
        return str(t.device) == "mps:0"
    return t.device.type == "cpu"


def roundtrip_dtype(dt, np_dtype, shape):
    """G1: one dtype cell of the matrix, both directions of write visibility."""
    import torch

    n = int(np.prod(shape))
    mod = 256 if np_dtype == np.uint8 else 240
    host = ((np.arange(n, dtype=np.int64) * 7 + 3) % mod).astype(np_dtype).reshape(shape)
    arr = qd.ndarray(dt, shape)
    arr.from_numpy(host)

    t = torch.from_dlpack(arr.to_dlpack())
    check(f"G1 {dt} export device", device_ok(t), str(t.device))
    check(f"G1 {dt} dtype", t.dtype == torch.from_numpy(np.zeros(1, np_dtype)).dtype, str(t.dtype))
    check(f"G1 {dt} qd->torch data equality", np.array_equal(consumer_read(t), host))

    # torch-side write -> qd to_numpy sees it (same memory). Cross-queue order:
    # torch writes on its MPS queue, qd reads via its own -- flush torch first.
    t.view(-1)[0] = torch.tensor(42, dtype=t.dtype)
    if ARCH == "metal" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    back = arr.to_numpy()
    check(f"G1 {dt} torch->qd write visible", np.array_equal(back, consumer_read(t)))
    check(f"G1 {dt} torch write changed qd buffer", not np.array_equal(back, host))

    # qd-side write -> torch alias sees it (same memory)
    host2 = host.copy()
    host2.reshape(-1)[0] = (int(host2.reshape(-1)[0]) + 17) % mod
    arr.from_numpy(host2)
    check(f"G1 {dt} qd->torch write visible", np.array_equal(consumer_read(t), host2))


print(f"=== G1 type matrix ({ARCH}) ===")
for dt, np_dtype in [
    (qd.u8, np.uint8),
    (qd.i8, np.int8),
    (qd.u16, np.uint16),
    (qd.i16, np.int16),
    (qd.u32, np.uint32),
    (qd.i32, np.int32),
    (qd.i64, np.int64),
    (qd.f32, np.float32),
    (qd.f64, np.float64),
    (qd.u64, np.uint64),
]:
    try:
        roundtrip_dtype(dt, np_dtype, (97, 3))
    except Exception as e:  # noqa: BLE001 - report the cell and keep the matrix running
        check(f"G1 {dt} matrix cell", False, f"{type(e).__name__}: {e}")

print(f"=== G2/G3 zero-copy wrap scaling + device (u8, {ARCH}) ===")
import torch

timings = {}
for mb in (1 / 1024, 0.25, 1, 4, 16, 64):  # 1KB .. 64MB
    n = int(mb * 1024 * 1024)
    arr = qd.ndarray(qd.u8, (n,))
    t = torch.from_dlpack(arr.to_dlpack())
    check(f"G3 {mb:.4f}MB device faithful", device_ok(t), str(t.device))
    timings[mb] = wrap_time_ms(arr)
    print(f"  u8 {mb:8.4f} MB  wrap = {timings[mb]:.3f} ms")
delta = timings[64.0] - timings[1 / 1024]
check("G2 wrap size-independence (64MB - 1KB delta < 0.5ms)", abs(delta) < 0.5, f"delta={delta:.4f} ms")
check("G2 wrap absolute floor (64MB < 0.2ms)", timings[64.0] < 0.2, f"{timings[64.0]:.3f} ms")

print(f"=== G4 alias contract == i32/f32 (#12) ({ARCH}) ===")
for dt, np_dtype in [(qd.u8, np.uint8), (qd.i32, np.int32), (qd.f32, np.float32)]:
    n = 4096
    arr = qd.ndarray(dt, (n,))
    h1 = (np.arange(n) % 200).astype(np_dtype)
    arr.from_numpy(h1)
    alias = torch.from_dlpack(arr.to_dlpack())
    check(f"G4 {dt} alias equals buffer before reuse", np.array_equal(consumer_read(alias), h1))
    h2 = (np.arange(n) % 200 + 50).astype(np_dtype)
    arr.from_numpy(h2)  # buffer reuse through a kernel write
    check(
        f"G4 {dt} buffer reuse silently changes alias (#12 semantics)",
        np.array_equal(consumer_read(alias), h2),
    )

print(f"=== G5 u8 end-to-end kernel sample ({ARCH}) ===")


@qd.kernel
def write_rgb(img: qd.types.ndarray(dtype=qd.u8, ndim=2), count: qd.i32):
    for i in range(count):
        for j in range(3):
            img[i, j] = qd.cast((i * 3 + j * 7) % 256, qd.u8)


N = 512
img = qd.ndarray(qd.u8, (N, 3))
write_rgb(img, N)
qd.sync()
expect = ((np.arange(N)[:, None] * 3 + np.arange(3)[None, :] * 7) % 256).astype(np.uint8)
tt = torch.from_dlpack(img.to_dlpack())
check(
    "G5 kernel-written u8 (N,3) -> uint8 alias",
    tt.dtype == torch.uint8 and device_ok(tt),
    f"{tt.dtype} {tt.device}",
)
check("G5 kernel data equality", np.array_equal(consumer_read(tt), expect))
qd.sync()

print()
if failures:
    print(f"RESULT: {len(failures)} FAILURES: {failures}")
    raise SystemExit(1)
print("RESULT: ALL GATES PASS")
