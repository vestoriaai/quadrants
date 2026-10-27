"""A86 最小致死形状 dump: B2 (if 包装循环) 的 IR dump 入口."""
import os
import numpy as np
import quadrants as qd

qd.init(arch=qd.cpu, cfg_optimization=False, advanced_optimization=False, opt_level=0,
        ad_stack_experimental_enabled=True, ad_stack_size=256, num_compile_threads=1,
        cpu_max_num_threads=1,
        **({"debug_dump_path": os.environ["QD_DUMP_DIR"]} if os.environ.get("QD_DUMP_DIR") else {}))

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


res.fill(0)
b2_kernel(M, rowadr, col, madr, vec, res)
res.grad.from_numpy((res.to_numpy() - 1.0).astype(np.float32))
b2_kernel.grad(M, rowadr, col, madr, vec, res)
print("M.grad:", M.grad.to_numpy())
print("vec.grad:", vec.grad.to_numpy())
