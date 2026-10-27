"""A86 追踪四十九 P0-b: 真 _mul_m 形状(2D CSR + doneqd 门 + 动态界)的最小复现.

与 1D CLEAN 版(repro_store_after_loop.py)的差异:
- 外层 for(__n0*__n1) 展平 + worldid/dofid div-mod;
- doneqd 门控(check_skip=False 结构保留);
- 内层 for k in range(start, end) 动态界, start/end 来自两次独立全局读;
- col/madr 中间局部变量。

检验 M_in.grad 与 vec.grad 的伴随传播(对照 FD)。
"""
import os
import numpy as np
import quadrants as qd

qd.init(arch=qd.cpu, cfg_optimization=False, advanced_optimization=False, opt_level=0,
        ad_stack_experimental_enabled=True, ad_stack_size=256, num_compile_threads=1,
        cpu_max_num_threads=1,
        **({"debug_dump_path": os.environ["QD_DUMP_DIR"]} if os.environ.get("QD_DUMP_DIR") else {}))
if os.environ.get("QD_DUMP_DIR"):
    print("dump ->", os.environ["QD_DUMP_DIR"])

nw = 1
nv = 4
nnz = 3  # 每行 3 元素

res = qd.ndarray(qd.f32, (nw, nv), needs_grad=True)
M_in = qd.ndarray(qd.f32, (nw, nnz * nv), needs_grad=True)  # 扁平 madr 空间
rowadr = qd.ndarray(qd.i32, (nv + 1))
col = qd.ndarray(qd.i32, (nnz * nv))
madr = qd.ndarray(qd.i32, (nnz * nv))
vec = qd.ndarray(qd.f32, (nw, nv), needs_grad=True)

rowadr_np = (np.arange(nv) * nnz).astype(np.int32)
rowadr.from_numpy(np.append(rowadr_np, nv * nnz))
M_np = (0.5 + 0.25 * np.arange(nnz * nv)).astype(np.float32)
M_in.from_numpy(M_np[None, :])
col_np = np.stack([np.arange(nv)] * nnz).T.reshape(-1).astype(np.int32)  # 对角窗口
col.from_numpy(col_np)
madr_np = np.arange(nnz * nv).astype(np.int32)  # 1:1 映射, 简化
madr.from_numpy(madr_np)
vec_np = np.linspace(1.0, 2.0, nv).astype(np.float32)
vec.from_numpy(vec_np[None, :])
w_np = np.linspace(0.1, 0.4, nv).astype(np.float32)


@qd.kernel
def mul_like_2d(M_flat: qd.types.ndarray(dtype=qd.f32, ndim=2),
                m_rowadr: qd.types.ndarray(dtype=qd.i32, ndim=1),
                m_col: qd.types.ndarray(dtype=qd.i32, ndim=1),
                m_madr: qd.types.ndarray(dtype=qd.i32, ndim=1),
                vv: qd.types.ndarray(dtype=qd.f32, ndim=2),
                rr: qd.types.ndarray(dtype=qd.f32, ndim=2),
                n0: int, n1: int):
    for t in range(n0 * n1):
        doneqd = False
        acc = 0.0
        end = 0
        start = 0
        worldid = t // n1
        dofid = t % n1
        if not doneqd:
            start = m_rowadr[dofid]
        if not doneqd:
            end = m_rowadr[dofid + 1]
        if not doneqd:
            for k in range(start, end):
                cc = m_col[k]
                mm = m_madr[k]
                acc += M_flat[worldid, mm] * vv[worldid, cc]
        if not doneqd:
            rr[worldid, dofid] = acc


res.fill(0)
mul_like_2d(M_in, rowadr, col, madr, vec, res, nw, nv)
res_np = res.to_numpy()
print("fwd res:", np.array2string(res_np[0], precision=4))

ref = np.zeros(nv)
for dof in range(nv):
    for k in range(nnz):
        idx = rowadr_np[dof] + k
        ref[dof] += M_np[madr_np[idx]] * vec_np[col_np[idx]]
print("fwd 解析:", np.array2string(ref, precision=4), " 一致:", np.allclose(res_np[0], ref, rtol=1e-5))


def fwd_loss(vv_np=None, mm_np=None):
    if vv_np is not None:
        vec.from_numpy(vv_np[None, :])
    if mm_np is not None:
        M_in.from_numpy(mm_np[None, :])
    res.fill(0)
    mul_like_2d(M_in, rowadr, col, madr, vec, res, nw, nv)
    r = res.to_numpy()[0].astype(np.float64)
    return float(np.sum(0.5 * (r - w_np) ** 2))


g_fd_vec = np.zeros(nv)
g_fd_M = np.zeros(nnz * nv)
e = 1e-3
for i in range(nv):
    vp = vec_np.copy(); vp[i] += e
    vm = vec_np.copy(); vm[i] -= e
    g_fd_vec[i] = (fwd_loss(vv_np=vp) - fwd_loss(vv_np=vm)) / (2 * e)
fwd_loss()  # 恢复原始 vec
for i in range(nnz * nv):
    mp = M_np.copy(); mp[i] += e
    mn = M_np.copy(); mn[i] -= e
    g_fd_M[i] = (fwd_loss(mm_np=mp) - fwd_loss(mm_np=mn)) / (2 * e)
fwd_loss()  # 恢复原始 M

# AD: seed dL/dres = res - w
res.grad.from_numpy((res_np[0] - w_np).astype(np.float32)[None, :])
vec.grad.fill(0)
M_in.grad.fill(0)
mul_like_2d.grad(M_in, rowadr, col, madr, vec, res, nw, nv)
g_ad_vec = vec.grad.to_numpy()[0]
g_ad_M = M_in.grad.to_numpy()[0]
print("AD vec.grad:", np.array2string(g_ad_vec, precision=5))
print("FD vec.grad:", np.array2string(g_fd_vec, precision=5))
print("AD M.grad[:6]:", np.array2string(g_ad_M[:6], precision=5))
print("FD M.grad[:6]:", np.array2string(g_fd_M[:6], precision=5))
ok_vec = np.allclose(g_ad_vec, g_fd_vec, rtol=1e-3, atol=1e-4)
ok_M = np.allclose(g_ad_M, g_fd_M, rtol=1e-3, atol=1e-4)
print("vec 传播:", "PASS" if ok_vec else "FAIL")
print("M   传播:", "PASS" if ok_M else "FAIL")
print("VERDICT:", "REPRODUCED" if (not ok_vec) or (not ok_M) else "CLEAN")
