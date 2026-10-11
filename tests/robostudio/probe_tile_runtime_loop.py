"""Gate probe (A2): Tile16x16 blocked Cholesky — runtime loop vs static unroll (Metal wrong values).

Pins the tile flavor of the Apple loop-carried miscompile family (cf. A5 /
probe_metal_loopdep.py): when tile registers must live across RUNTIME backedges
(block loop `for k in range(nb)` with runtime nb, plus the nested solve / rank-1
update loops), the dumped MSL carries a 16-tile-register phi chain at the loop
head and the m=48 factorization returns wrong values on Metal (~2e-2 vs numpy).
The fully static-unrolled variant of the IDENTICAL algorithm (straight-line,
no runtime backedge carries tiles) is correct (~1e-7). Same construction as
mujoqo _probe/p0_block_chol.py (runtime kernel) and /tmp/t12.py (static kernel).

cpu backend: Tile16x16 has no cpu lowering by design (kernel compile raises
QuadrantsSyntaxError) -> SKIP, exit 0.

Batched-tile layout convention: top-level loop emits nworld*16 iterations,
env = idx // 16, loop_config(block_dim=16); see misc/demos/cholesky_blocked.py.

Statuses on metal: static variant is ASSERTED (drives the exit code); runtime
variant is expected to FAIL — reported as PINNED. When Apple's toolchain is
fixed (or our mitigation lands) the runtime line flips to PASS and the pin in
docs/robostudio_features.md (A2) must be updated deliberately.

Run: python probe_tile_runtime_loop.py [metal]   (exit 0 = static gate passes)
"""
import sys

import numpy as np

import quadrants as qd

ARCH = sys.argv[1] if len(sys.argv) > 1 else "metal"
qd.init(arch=qd.metal if ARCH == "metal" else qd.cpu)

if ARCH != "metal":
    print(f"A2 tile runtime-loop gate arch={ARCH}: SKIP (Tile16x16 is metal-only by design; "
          f"cpu kernel compile raises QuadrantsSyntaxError) -> exit 0")
    sys.exit(0)

S = 16   # tile size
NB = 3   # 3x3 blocks
M = NB * S  # 48
NW = 4


# ---- A2 runtime-loop variant (known WRONG on metal, ~2e-2) ----
# Direct port of p0_block_chol.py chol_tile (factorization only): every block
# loop bound is a runtime value, so tile registers are carried across runtime
# backedges — the shape that miscompiles.
@qd.kernel
def chol_runtime(a: qd.types.ndarray(ndim=3), lout: qd.types.ndarray(ndim=3), m: qd.i32):
    qd.loop_config(block_dim=S)
    for idx in range(a.shape[0] * S):   # nworld*16 iterations; 16 lanes cooperate per world
        w = idx // S                    # env = idx // 16
        nb = m // S                     # runtime trip count (3 at m=48) — kept runtime on purpose
        for k in range(nb):
            k0 = k * S
            T = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            T[:] = a[w, k0:k0 + S, k0:k0 + S]
            T.cholesky_(1e-6)
            lout[w, k0:k0 + S, k0:k0 + S] = T
            for i in range(k + 1, nb):
                i0 = i * S
                B = qd.simt.Tile16x16.zeros(dtype=qd.f32)
                B[:] = a[w, i0:i0 + S, k0:k0 + S]
                T.solve_triangular_(B)
                lout[w, i0:i0 + S, k0:k0 + S] = B
                for j in range(k + 1, nb):
                    j0 = j * S
                    Wij = qd.simt.Tile16x16.zeros(dtype=qd.f32)
                    Wij[:] = a[w, j0:j0 + S, i0:i0 + S]
                    for c in range(S):
                        v1 = lout[w, j0:j0 + S, k0 + c]
                        v2 = lout[w, i0:i0 + S, k0 + c]
                        Wij -= qd.outer(v1, v2)
                    a[w, j0:j0 + S, i0:i0 + S] = Wij


# ---- A2 static-unrolled variant (known CORRECT, ~1e-7) ----
# Identical algorithm, block loops fully unrolled (t12.py): straight-line SSA,
# no tile register is carried across a runtime backedge.
@qd.kernel
def chol_static(a: qd.types.ndarray(ndim=3), lout: qd.types.ndarray(ndim=3)):
    qd.loop_config(block_dim=S)
    for idx in range(a.shape[0] * S):
        w = idx // S
        # k=0
        T0 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
        T0[:] = a[w, 0:16, 0:16]
        T0.cholesky_(1e-6)
        lout[w, 0:16, 0:16] = T0
        B1 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
        B1[:] = a[w, 16:32, 0:16]
        T0.solve_triangular_(B1)
        lout[w, 16:32, 0:16] = B1
        B2 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
        B2[:] = a[w, 32:48, 0:16]
        T0.solve_triangular_(B2)
        lout[w, 32:48, 0:16] = B2
        # trailing updates: (1,1),(2,2),(2,1)
        W11 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
        W11[:] = a[w, 16:32, 16:32]
        for c in range(S):
            W11 -= qd.outer(lout[w, 16:32, c], lout[w, 16:32, c])
        a[w, 16:32, 16:32] = W11
        W22 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
        W22[:] = a[w, 32:48, 32:48]
        for c in range(S):
            W22 -= qd.outer(lout[w, 32:48, c], lout[w, 32:48, c])
        a[w, 32:48, 32:48] = W22
        W21 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
        W21[:] = a[w, 32:48, 16:32]
        for c in range(S):
            W21 -= qd.outer(lout[w, 32:48, c], lout[w, 16:32, c])
        a[w, 32:48, 16:32] = W21
        # k=1
        T1 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
        T1[:] = a[w, 16:32, 16:32]
        T1.cholesky_(1e-6)
        lout[w, 16:32, 16:32] = T1
        B21 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
        B21[:] = a[w, 32:48, 16:32]
        T1.solve_triangular_(B21)
        lout[w, 32:48, 16:32] = B21
        W22b = qd.simt.Tile16x16.zeros(dtype=qd.f32)
        W22b[:] = a[w, 32:48, 32:48]
        for c in range(S):
            W22b -= qd.outer(lout[w, 32:48, 16 + c], lout[w, 32:48, 16 + c])
        a[w, 32:48, 32:48] = W22b
        # k=2
        T2 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
        T2[:] = a[w, 32:48, 32:48]
        T2.cholesky_(1e-6)
        lout[w, 32:48, 32:48] = T2


def run_variant(kernel, a_in):
    w = qd.ndarray(dtype=qd.f32, shape=(NW, M, M))
    w.from_numpy(a_in.copy())
    l = qd.ndarray(dtype=qd.f32, shape=(NW, M, M))
    l.from_numpy(np.zeros((NW, M, M), np.float32))
    kernel(w, l) if kernel is chol_static else kernel(w, l, M)
    qd.sync()
    # cholesky_ leaves the upper triangle untouched; L starts at zeros
    return np.abs(np.tril(l.to_numpy()) - REF).max()


rng = np.random.default_rng(0)
Bm = rng.standard_normal((NW, M, M)).astype(np.float32) * 0.3
A = np.matmul(Bm, Bm.transpose(0, 2, 1)) + M * np.eye(M, dtype=np.float32)
REF = np.linalg.cholesky(A)

STATIC_TOL = 1e-5    # static variant measures ~1e-6 (documented 1e-7..1e-6 band)
RUNTIME_TOL = 1e-4   # runtime variant measures ~2e-2 when the defect is present

err_static = run_variant(chol_static, A)
err_runtime = run_variant(chol_runtime, A)

static_ok = err_static < STATIC_TOL
runtime_ok = err_runtime < RUNTIME_TOL

print(f"A2 tile runtime-loop gate arch={ARCH}  m={M} ({NB}x{NB} blocks of {S}x{S}, {NW} worlds, vs numpy)")
print(f"  static  (block loops fully unrolled): err={err_static:.3e}  [{'PASS' if static_ok else 'FAIL'}] (asserted, tol {STATIC_TOL:.0e})")
print(f"  runtime (runtime block loops):        err={err_runtime:.3e}  [{'PASS' if runtime_ok else 'FAIL'}] (pinned, tol {RUNTIME_TOL:.0e})")
if runtime_ok:
    print("  A2 PINNED status: runtime variant now CORRECT on metal — defect fixed/mitigated; "
          "update the pin (docs/robostudio_features.md A2) and flip this gate green.")
else:
    print("  A2 PINNED status: runtime variant loses updates on metal as expected — defect pinned "
          "(Apple loop-carried tile miscompile family, cf. A5).")
print(f"A2 tile runtime-loop gate arch={ARCH}: exit {0 if static_ok else 1} (decided by the static assertion)")
sys.exit(0 if static_ok else 1)
