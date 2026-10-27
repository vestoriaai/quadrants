"""A86 因子化探针: 隔离「if 门控包装」与「2D+动态界」两个结构差异.

Variant A: 2D 展平 + 动态界 range(start,end), 无 if 包装 → 若 CLEAN 则 if 包装是致死因子
Variant B: 1D + 静态界 nnz, res 存储包在 if not doneqd 里 → 若 REPRODUCED 则确认 if 包装
(共享 FD 机制, 各自 kernel 各自验证)
"""
import os
import numpy as np
import quadrants as qd

qd.init(arch=qd.cpu, cfg_optimization=False, advanced_optimization=False, opt_level=0,
        ad_stack_experimental_enabled=True, ad_stack_size=256, num_compile_threads=1,
        cpu_max_num_threads=1)

nw = 1
nv = 4
nnz = 3

# ---- Variant A 数据 (2D + 动态界, 无 if) ----
resA = qd.ndarray(qd.f32, (nw, nv), needs_grad=True)
M_A = qd.ndarray(qd.f32, (nw, nnz * nv), needs_grad=True)
rowadr = qd.ndarray(qd.i32, (nv + 1))
col = qd.ndarray(qd.i32, (nnz * nv))
madr = qd.ndarray(qd.i32, (nnz * nv))
vecA = qd.ndarray(qd.f32, (nw, nv), needs_grad=True)

rowadr_np = (np.arange(nv) * nnz).astype(np.int32)
rowadr.from_numpy(np.append(rowadr_np, nv * nnz))
M_np = (0.5 + 0.25 * np.arange(nnz * nv)).astype(np.float32)
M_A.from_numpy(M_np[None, :])
col_np = np.stack([np.arange(nv)] * nnz).T.reshape(-1).astype(np.int32)
col.from_numpy(col_np)
madr_np = np.arange(nnz * nv).astype(np.int32)
madr.from_numpy(madr_np)
vec_np = np.linspace(1.0, 2.0, nv).astype(np.float32)
vecA.from_numpy(vec_np[None, :])
w_np = np.linspace(0.1, 0.4, nv).astype(np.float32)


@qd.kernel
def varA_kernel(M_flat: qd.types.ndarray(dtype=qd.f32, ndim=2),
                m_rowadr: qd.types.ndarray(dtype=qd.i32, ndim=1),
                m_col: qd.types.ndarray(dtype=qd.i32, ndim=1),
                m_madr: qd.types.ndarray(dtype=qd.i32, ndim=1),
                vv: qd.types.ndarray(dtype=qd.f32, ndim=2),
                rr: qd.types.ndarray(dtype=qd.f32, ndim=2),
                n0: int, n1: int):
    for t in range(n0 * n1):
        acc = 0.0
        worldid = t // n1
        dofid = t % n1
        start = m_rowadr[dofid]
        end = m_rowadr[dofid + 1]
        for k in range(start, end):
            cc = m_col[k]
            mm = m_madr[k]
            acc += M_flat[worldid, mm] * vv[worldid, cc]
        rr[worldid, dofid] = acc


# ---- Variant B 数据 (1D + 静态界 + if 包装) ----
resB = qd.ndarray(qd.f32, (nv), needs_grad=True)
M_B = qd.ndarray(qd.f32, (nnz * nv), needs_grad=True)
vecB = qd.ndarray(qd.f32, (nv), needs_grad=True)
M_B.from_numpy(M_np)
vecB.from_numpy(vec_np)


@qd.kernel
def varB_kernel(M_flat: qd.types.ndarray(dtype=qd.f32, ndim=1),
                m_rowadr: qd.types.ndarray(dtype=qd.i32, ndim=1),
                m_col: qd.types.ndarray(dtype=qd.i32, ndim=1),
                m_madr: qd.types.ndarray(dtype=qd.i32, ndim=1),
                vv: qd.types.ndarray(dtype=qd.f32, ndim=1),
                rr: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    for d in range(nv):
        doneqd = False
        acc = 0.0
        start = 0
        end = 0
        if not doneqd:
            start = m_rowadr[d]
        if not doneqd:
            end = m_rowadr[d + 1]
        if not doneqd:
            for k in range(start, end):
                cc = m_col[k]
                mm = m_madr[k]
                acc += M_flat[mm] * vv[cc]
        if not doneqd:
            rr[d] = acc


def run_case(tag, kern, res, M, vec, two_d):
    kern(M, rowadr, col, madr, vec, res) if not two_d else kern(M, rowadr, col, madr, vec, res, nw, nv)
    res_np = res.to_numpy()
    ref = np.zeros(nv)
    for dof in range(nv):
        for k in range(nnz):
            idx = rowadr_np[dof] + k
            ref[dof] += M_np[madr_np[idx]] * vec_np[col_np[idx]]
    fwd_ok = np.allclose(res_np.reshape(-1), ref, rtol=1e-5)
    print(f"[{tag}] fwd:", np.array2string(res_np.reshape(-1), precision=3), " 一致:", fwd_ok)

    def fwd_loss(vv_n=None, mm_n=None):
        if vv_n is not None:
            vec.from_numpy(vv_n[None, :] if two_d else vv_n)
        if mm_n is not None:
            M.from_numpy(mm_n[None, :] if two_d else mm_n)
        if two_d:
            kern(M, rowadr, col, madr, vec, res, nw, nv)
        else:
            kern(M, rowadr, col, madr, vec, res)
        r = res.to_numpy().reshape(-1).astype(np.float64)
        return float(np.sum(0.5 * (r - w_np) ** 2))

    g_fd_vec = np.zeros(nv)
    g_fd_M = np.zeros(nnz * nv)
    e = 1e-3
    for i in range(nv):
        vp = vec_np.copy(); vp[i] += e
        vm = vec_np.copy(); vm[i] -= e
        g_fd_vec[i] = (fwd_loss(vv_n=vp) - fwd_loss(vv_n=vm)) / (2 * e)
    fwd_loss()
    for i in range(nnz * nv):
        mp = M_np.copy(); mp[i] += e
        mn = M_np.copy(); mn[i] -= e
        g_fd_M[i] = (fwd_loss(mm_n=mp) - fwd_loss(mm_n=mn)) / (2 * e)
    fwd_loss()

    res.grad.from_numpy(((res_np.reshape(-1) - w_np).astype(np.float32))[None, :] if two_d
                        else (res_np - w_np).astype(np.float32))
    vec.grad.fill(0)
    M.grad.fill(0)
    if two_d:
        kern.grad(M, rowadr, col, madr, vec, res, nw, nv)
    else:
        kern.grad(M, rowadr, col, madr, vec, res)
    g_ad_vec = vec.grad.to_numpy().reshape(-1)
    g_ad_M = M.grad.to_numpy().reshape(-1)
    ok_vec = np.allclose(g_ad_vec, g_fd_vec, rtol=1e-3, atol=1e-4)
    ok_M = np.allclose(g_ad_M, g_fd_M, rtol=1e-3, atol=1e-4)
    print(f"[{tag}] AD vec:", np.array2string(g_ad_vec, precision=4))
    print(f"[{tag}] FD vec:", np.array2string(g_fd_vec, precision=4))
    print(f"[{tag}] AD M[:4]:", np.array2string(g_ad_M[:4], precision=4),
          " FD M[:4]:", np.array2string(g_fd_M[:4], precision=4))
    print(f"[{tag}] VERDICT:", "REPRODUCED" if not (ok_vec and ok_M) else "CLEAN")


run_case("A: 2D+动态界 无if", varA_kernel, resA, M_A, vecA, two_d=True)
run_case("B: 1D+静态界 if包装", varB_kernel, resB, M_B, vecB, two_d=False)
