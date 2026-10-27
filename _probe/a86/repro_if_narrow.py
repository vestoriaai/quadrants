"""A86 收窄探针: if 包装致死的最小层级.

B1: 只 if 包装 store(循环无包装)
B2: 只 if 包装循环(store 无包装)
B3: 运行时真条件(flag=True 传入)包装 store —— 检验 const-true 折叠是否是必要条件
"""
import numpy as np
import quadrants as qd

qd.init(arch=qd.cpu, cfg_optimization=False, advanced_optimization=False, opt_level=0,
        ad_stack_experimental_enabled=True, ad_stack_size=256, num_compile_threads=1,
        cpu_max_num_threads=1)

nv = 4
nnz = 3

res = qd.ndarray(qd.f32, (nv), needs_grad=True)
M = qd.ndarray(qd.f32, (nnz * nv), needs_grad=True)
rowadr = qd.ndarray(qd.i32, (nv + 1))
col = qd.ndarray(qd.i32, (nnz * nv))
madr = qd.ndarray(qd.i32, (nnz * nv))
vec = qd.ndarray(qd.f32, (nv), needs_grad=True)

rowadr_np = (np.arange(nv) * nnz).astype(np.int32)
rowadr.from_numpy(np.append(rowadr_np, nv * nnz))
M_np = (0.5 + 0.25 * np.arange(nnz * nv)).astype(np.float32)
M.from_numpy(M_np)
col_np = np.stack([np.arange(nv)] * nnz).T.reshape(-1).astype(np.int32)
col.from_numpy(col_np)
madr_np = np.arange(nnz * nv).astype(np.int32)
madr.from_numpy(madr_np)
vec_np = np.linspace(1.0, 2.0, nv).astype(np.float32)
vec.from_numpy(vec_np)
w_np = np.linspace(0.1, 0.4, nv).astype(np.float32)
flag = qd.ndarray(qd.i32, (1))
flag.from_numpy(np.array([1], dtype=np.int32))


@qd.kernel
def b1_kernel(M_flat: qd.types.ndarray(dtype=qd.f32, ndim=1),
              m_rowadr: qd.types.ndarray(dtype=qd.i32, ndim=1),
              m_col: qd.types.ndarray(dtype=qd.i32, ndim=1),
              m_madr: qd.types.ndarray(dtype=qd.i32, ndim=1),
              vv: qd.types.ndarray(dtype=qd.f32, ndim=1),
              rr: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    for d in range(nv):
        doneqd = False
        acc = 0.0
        start = m_rowadr[d]
        end = m_rowadr[d + 1]
        for k in range(start, end):
            cc = m_col[k]
            mm = m_madr[k]
            acc += M_flat[mm] * vv[cc]
        if not doneqd:
            rr[d] = acc


@qd.kernel
def b2_kernel(M_flat: qd.types.ndarray(dtype=qd.f32, ndim=1),
              m_rowadr: qd.types.ndarray(dtype=qd.i32, ndim=1),
              m_col: qd.types.ndarray(dtype=qd.i32, ndim=1),
              m_madr: qd.types.ndarray(dtype=qd.i32, ndim=1),
              vv: qd.types.ndarray(dtype=qd.f32, ndim=1),
              rr: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    for d in range(nv):
        doneqd = False
        acc = 0.0
        start = m_rowadr[d]
        end = m_rowadr[d + 1]
        if not doneqd:
            for k in range(start, end):
                cc = m_col[k]
                mm = m_madr[k]
                acc += M_flat[mm] * vv[cc]
        rr[d] = acc


@qd.kernel
def b3_kernel(M_flat: qd.types.ndarray(dtype=qd.f32, ndim=1),
              m_rowadr: qd.types.ndarray(dtype=qd.i32, ndim=1),
              m_col: qd.types.ndarray(dtype=qd.i32, ndim=1),
              m_madr: qd.types.ndarray(dtype=qd.i32, ndim=1),
              vv: qd.types.ndarray(dtype=qd.f32, ndim=1),
              rr: qd.types.ndarray(dtype=qd.f32, ndim=1),
              ff: qd.types.ndarray(dtype=qd.i32, ndim=1)):
    for d in range(nv):
        acc = 0.0
        start = m_rowadr[d]
        end = m_rowadr[d + 1]
        for k in range(start, end):
            cc = m_col[k]
            mm = m_madr[k]
            acc += M_flat[mm] * vv[cc]
        if ff[0] == 1:
            rr[d] = acc


def run_case(tag, kern, has_flag):
    if has_flag:
        kern(M, rowadr, col, madr, vec, res, flag)
    else:
        kern(M, rowadr, col, madr, vec, res)
    res_np = res.to_numpy()
    ref = np.zeros(nv)
    for dof in range(nv):
        for k in range(nnz):
            idx = rowadr_np[dof] + k
            ref[dof] += M_np[madr_np[idx]] * vec_np[col_np[idx]]
    print(f"[{tag}] fwd:", np.array2string(res_np, precision=3), " 一致:", np.allclose(res_np, ref, rtol=1e-5))

    def fwd_loss(vv_n=None):
        if vv_n is not None:
            vec.from_numpy(vv_n)
        if has_flag:
            kern(M, rowadr, col, madr, vec, res, flag)
        else:
            kern(M, rowadr, col, madr, vec, res)
        r = res.to_numpy().astype(np.float64)
        return float(np.sum(0.5 * (r - w_np) ** 2))

    g_fd_vec = np.zeros(nv)
    e = 1e-3
    for i in range(nv):
        vp = vec_np.copy(); vp[i] += e
        vm = vec_np.copy(); vm[i] -= e
        g_fd_vec[i] = (fwd_loss(vv_n=vp) - fwd_loss(vv_n=vm)) / (2 * e)
    fwd_loss()

    res.grad.from_numpy((res_np - w_np).astype(np.float32))
    vec.grad.fill(0)
    if has_flag:
        kern.grad(M, rowadr, col, madr, vec, res, flag)
    else:
        kern.grad(M, rowadr, col, madr, vec, res)
    g_ad_vec = vec.grad.to_numpy()
    ok_vec = np.allclose(g_ad_vec, g_fd_vec, rtol=1e-3, atol=1e-4)
    print(f"[{tag}] AD vec:", np.array2string(g_ad_vec, precision=4),
          " FD vec:", np.array2string(g_fd_vec, precision=4))
    print(f"[{tag}] VERDICT:", "REPRODUCED" if not ok_vec else "CLEAN")


run_case("B1: 只 if 包 store", b1_kernel, False)
run_case("B2: 只 if 包循环", b2_kernel, False)
run_case("B3: 运行时条件包 store", b3_kernel, True)
