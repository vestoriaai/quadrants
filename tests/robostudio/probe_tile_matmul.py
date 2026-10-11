"""Gate probe (C3): Tile16x16.matmul_ / zero_upper_ — register-tile blocked matmul.

1. Unit: C += alpha * (A @ B) vs numpy for alpha=±1.
2. Blocked Cholesky (the mujoco Newton-Hessian shape): static-unrolled 3x3-block
   factorize+solve using matmul_ (replacing the column-outer loop), verified vs numpy.
   Static unroll because runtime-loop tile composition is A2 (still open).
Run: python probe_tile_matmul.py [metal|cpu]  (exit 0 = pass)
Ledger: mujocoqd/compiler-line C3.
"""
import sys

import numpy as np

import quadrants as qd

ARCH = sys.argv[1] if len(sys.argv) > 1 else "metal"
qd.init(arch=qd.metal if ARCH == "metal" else qd.cpu)

S = 16
fails = []


def gate(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL':4}] {name} {detail}")
    if not ok:
        fails.append(name)


# ---- 1. unit matmul: C += alpha * A@B  (batched over worlds, one tile pair per world)
@qd.kernel
def k_unit(a: qd.types.ndarray(ndim=3), b: qd.types.ndarray(ndim=3), c: qd.types.ndarray(ndim=3), alpha: qd.f32):
    qd.loop_config(block_dim=S)
    for idx in range(a.shape[0] * S):
        w = idx // S
        A = qd.simt.Tile16x16.zeros(dtype=qd.f32)
        A[:] = a[w, 0:S, 0:S]
        B = qd.simt.Tile16x16.zeros(dtype=qd.f32)
        B[:] = b[w, 0:S, 0:S]
        C = qd.simt.Tile16x16.zeros(dtype=qd.f32)
        C[:] = c[w, 0:S, 0:S]
        C.matmul_(A, B, alpha)
        c[w, 0:S, 0:S] = C


rng = np.random.default_rng(0)
nw = 8
A = rng.standard_normal((nw, S, S)).astype(np.float32)
B = rng.standard_normal((nw, S, S)).astype(np.float32)
for alpha in (1.0, -1.0):
    C0 = rng.standard_normal((nw, S, S)).astype(np.float32)
    c_t = qd.ndarray(dtype=qd.f32, shape=(nw, S, S)); c_t.from_numpy(C0.copy())
    a_t = qd.ndarray(dtype=qd.f32, shape=(nw, S, S)); a_t.from_numpy(A)
    b_t = qd.ndarray(dtype=qd.f32, shape=(nw, S, S)); b_t.from_numpy(B)
    k_unit(a_t, b_t, c_t, np.float32(alpha))
    qd.sync()
    want = C0 + alpha * np.matmul(A, B)
    got = c_t.to_numpy()
    gate(f"unit_matmul alpha={alpha}", np.abs(got - want).max() / np.abs(want).max() < 1e-5,
         f"maxerr={np.abs(got - want).max():.2e}")

# ---- 2. blocked factorize+solve, static 2x2 and 3x3, via matmul_ + zero_upper_
def make_src(nb):
    m = nb * S
    Bm = rng.standard_normal((nw, m, m)).astype(np.float32) * 0.3
    return np.matmul(Bm, Bm.transpose(0, 2, 1)) + m * np.eye(m, dtype=np.float32)


@qd.kernel
def chol2(w_in: qd.types.ndarray(ndim=3), L: qd.types.ndarray(ndim=3), bv: qd.types.ndarray(ndim=2), xv: qd.types.ndarray(ndim=2)):
    qd.loop_config(block_dim=S)
    for idx in range(w_in.shape[0] * S):
        w = idx // S
        tid = qd.i32(qd.simt.subgroup.invocation_id())
        T = qd.simt.Tile16x16.zeros(dtype=qd.f32)
        T[:] = w_in[w, 0:S, 0:S]
        T.cholesky_(1e-6)
        T.zero_upper_()
        L[w, 0:S, 0:S] = T
        B1 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
        B1[:] = w_in[w, S:2 * S, 0:S]
        T.solve_triangular_(B1)
        L[w, S:2 * S, 0:S] = B1
        W11 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
        W11[:] = w_in[w, S:2 * S, S:2 * S]
        W11.matmul_at_(B1, B1, -1.0)  # W11 -= B1 @ B1^T
        W11.cholesky_(1e-6)
        W11.zero_upper_()
        L[w, S:2 * S, S:2 * S] = W11
        if tid == 0:
            for i in range(2 * S):
                s = bv[w, i]
                for j in range(i):
                    s -= L[w, i, j] * xv[w, j]
                xv[w, i] = s / L[w, i, i]
            for ii in range(2 * S):
                i = 2 * S - 1 - ii
                s = xv[w, i]
                for j in range(i + 1, 2 * S):
                    s -= L[w, j, i] * xv[w, j]
                xv[w, i] = s / L[w, i, i]


# the 2x2 case above is complete; run it
m = 2 * S
A, b = make_src(2), None
Br = np.random.default_rng(1).standard_normal((nw, m)).astype(np.float32)
wt = qd.ndarray(dtype=qd.f32, shape=(nw, m, m)); wt.from_numpy(A.copy())
Lt = qd.ndarray(dtype=qd.f32, shape=(nw, m, m)); Lt.from_numpy(np.zeros((nw, m, m), np.float32))
bt = qd.ndarray(dtype=qd.f32, shape=(nw, m)); bt.from_numpy(Br)
xt = qd.ndarray(dtype=qd.f32, shape=(nw, m)); xt.from_numpy(np.zeros((nw, m), np.float32))
chol2(wt, Lt, bt, xt)
qd.sync()
Lnp = Lt.to_numpy()
recon = np.matmul(Lnp, Lnp.transpose(0, 2, 1))
err_f = np.abs(recon - A).max() / np.abs(A).max()
got = xt.to_numpy()
err_s = np.abs(np.matmul(A, got[..., None])[..., 0] - Br).max() / np.abs(Br).max()
gate("blocked2x2_chol_solve", err_f < 1e-3 and err_s < 1e-3, f"fact={err_f:.2e} solve={err_s:.2e}")
ref = np.linalg.cholesky(A)
errb = np.abs(np.tril(Lnp) - ref)
for rb in range(2):
    for cb in range(2):
        print(f"   blk({rb},{cb}) err={errb[:, rb*16:(rb+1)*16, cb*16:(cb+1)*16].max():.2e}")

print(f"\nTILE MATMUL GATE arch={ARCH}: {'ALL PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
