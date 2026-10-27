import os
os.environ['QD_DUMP_IR'] = '1'
"""A86 追踪四十九 P0: store-after-loop 形状的最小复现.

复刻 mul_m_kernel 形状: 外层展平循环 + 内层 for(nnz 动态界)累积 acc + 循环后
写 res。检验自动反向: res.grad → acc.grad → val.grad/vec.grad 的传播是否
完整(对照手工 FD 与解析解)。

预期(若 bug 复现): M_val.grad 正常写出, vec.grad 丢失(被 load-top+pop 丢弃)。
"""
import os, sys
import numpy as np
import quadrants as qd

qd.init(arch=qd.cpu, cfg_optimization=False, advanced_optimization=False, opt_level=0,
        ad_stack_experimental_enabled=True, ad_stack_size=256, num_compile_threads=1,
        cpu_max_num_threads=1, debug_dump_path='/tmp/repro_ir')

n = 4
nnz = 3

res = qd.ndarray(qd.f32, (n), needs_grad=True)
rowadr = qd.ndarray(qd.i32, (n))
val = qd.ndarray(qd.f32, (nnz * n), needs_grad=True)
col = qd.ndarray(qd.i32, (nnz * n))
vec = qd.ndarray(qd.f32, (n), needs_grad=True)

# 每行一个 3-元稠密窗口(动态内层界, 仿 mujoqo CSR)
rowadr.from_numpy((np.arange(n) * nnz).astype(np.int32))
val_np = (0.5 + 0.25 * np.arange(nnz * n)).astype(np.float32)
col_np = np.stack([np.arange(n)] * nnz).T.reshape(-1).astype(np.int32)  # 对角 + 邻元
val.from_numpy(val_np)
col.from_numpy(col_np)
vec_np = np.linspace(1.0, 2.0, n).astype(np.float32)
vec.from_numpy(vec_np)
w_np = np.linspace(0.1, 0.4, n).astype(np.float32)  # loss 权重


@qd.kernel
def mul_like(res: qd.types.ndarray(dtype=qd.f32, ndim=1),
             rowadr: qd.types.ndarray(dtype=qd.i32, ndim=1),
             vval: qd.types.ndarray(dtype=qd.f32, ndim=1),
             ccol: qd.types.ndarray(dtype=qd.i32, ndim=1),
             vvec: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    for d in range(n):
        acc = 0.0
        start = rowadr[d]
        for k in range(nnz):
            acc = acc + vval[start + k] * vvec[ccol[start + k]]
        res[d] = acc


@qd.kernel
def loss_k(res: qd.types.ndarray(dtype=qd.f32, ndim=1), w: qd.types.ndarray(dtype=qd.f32, ndim=1),
           Lt: qd.types.template()):
    for d in range(n):
        Lt[None] += 0.5 * (res[d] - w[d]) * (res[d] - w[d])


res.fill(0)
L = qd.field(dtype=qd.f32, shape=(), needs_grad=True)
Lx = qd.ndarray(qd.f32, (n))
Lx.from_numpy(w_np)

# forward
mul_like(res, rowadr, val, col, vec)
res_np = res.to_numpy()
print("fwd res:", np.array2string(res_np, precision=4))

# 解析参考: res[d] = Σ_k val[rowadr+k]·vec[col]
ref = np.zeros(n)
for d in range(n):
    for k in range(nnz):
        ref[d] += val_np[rowadr[d] + k] * vec_np[col_np[rowadr[d] + k]]
print("fwd 解析:", np.array2string(ref, precision=4), " fwd 一致:", np.allclose(res_np, ref, rtol=1e-5))

# FD 参考(vec.grad 与 val.grad)
def fwd_loss(vv=None, vv2=None):
    x = res.to_numpy().copy()
    mul_like(res, rowadr, val if vv2 is None else vv2, col, vec if vv is None else vv)
    r = res.to_numpy().copy()
    l = float(np.sum(0.5 * (r - w_np) ** 2))
    mul_like(res, rowadr, val, col, vec)
    return l

g_fd_vec = np.zeros(n)
g_fd_val = np.zeros(nnz * n)
e = 1e-3
for i in range(n):
    vp = vec_np.copy(); vp[i] += e
    vm = vec_np.copy(); vm[i] -= e
    g_fd_vec[i] = (fwd_loss(vp) - fwd_loss(vm)) / (2 * e)
for i in range(nnz * n):
    vp = val_np.copy(); vp[i] += e
    vm = val_np.copy(); vm[i] -= e
    g_fd_val[i] = (fwd_loss(vp) - fwd_loss(vm)) / (2 * e)

# AD
vec.grad.fill(0)
val.grad.fill(0)
L.fill(0)
mul_like.grad(res, rowadr, val, col, vec)
g_ad_vec = vec.grad.to_numpy()
g_ad_val = val.grad.to_numpy()
print("AD  vec.grad:", np.array2string(g_ad_vec, precision=5))
print("FD  vec.grad:", np.array2string(g_fd_vec, precision=5))
print("AD  val.grad[:8]:", np.array2string(g_ad_val[:8], precision=5))
print("FD  val.grad[:8]:", np.array2string(g_fd_val[:8], precision=5))
ok_vec = np.allclose(g_ad_vec, g_fd_vec, rtol=1e-3, atol=1e-5)
ok_val = np.allclose(g_ad_val, g_fd_val, rtol=1e-3, atol=1e-5)
print("vec 传播:", "PASS" if ok_vec else "FAIL")
print("val 传播:", "PASS" if ok_val else "FAIL")
verdict = "REPRODUCED" if (not ok_vec) or (not ok_val) else "CLEAN"
print("VERDICT:", verdict)
