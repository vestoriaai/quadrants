"""两核 tape 二分: quadrants 纯 ndarray kernel 的 tape{A;B} REV 是否生效.

A: y = x * 2 (y 独立 needs_grad 数组)
B: L += 0.5*Σ(y - t)^2
期望 x.grad = 2*(y-t)*2 = 4*(2x-t)
"""
import numpy as np
import quadrants as qd

qd.init(arch=qd.cpu, cfg_optimization=False, advanced_optimization=False, opt_level=0,
        ad_stack_experimental_enabled=True, ad_stack_size=256, num_compile_threads=1)

n = 8
x = qd.ndarray(qd.f32, (n), needs_grad=True)
y = qd.ndarray(qd.f32, (n), needs_grad=True)
t = qd.ndarray(qd.f32, (n))
x_np = np.linspace(1.0, 2.0, n).astype(np.float32)
x.from_numpy(x_np)
t_np = np.linspace(0.5, 1.5, n).astype(np.float32)
t.from_numpy(t_np)

L = qd.field(dtype=qd.f32, shape=(), needs_grad=True)


@qd.kernel
def scale2(xx: qd.types.ndarray(dtype=qd.f32, ndim=1), yy: qd.types.ndarray(dtype=qd.f32, ndim=1)):
    for i in range(n):
        yy[i] = xx[i] * 2.0


@qd.kernel
def loss_k(yy: qd.types.ndarray(dtype=qd.f32, ndim=1), tt: qd.types.ndarray(dtype=qd.f32, ndim=1),
           Lt: qd.types.template()):
    for i in range(n):
        diff = yy[i] - tt[i]
        Lt[None] += 0.5 * diff * diff


# 基线: 单核 tape{loss}(y 已由 FWD 填好)
scale2(x, y)
for _warm in range(3):
    L.fill(0)
    with qd.ad.Tape(loss=L):
        loss_k(y, t, L)
    qd.ad.clear_all_gradients()

scale2(x, y)
L.fill(0)
with qd.ad.Tape(loss=L):
    loss_k(y, t, L)
qd.sync()
g1 = x.grad.to_numpy().copy()
exp1 = 4.0 * (2.0 * x_np - t_np)
print("单核 tape x.grad:", np.array2string(g1, precision=3), " PASS" if np.allclose(g1, exp1, atol=1e-4) else " FAIL")

# 两核 tape{scale2; loss}
for _warm in range(3):
    x.from_numpy(x_np)
    L.fill(0)
    with qd.ad.Tape(loss=L):
        scale2(x, y)
        loss_k(y, t, L)
    qd.ad.clear_all_gradients()

x.from_numpy(x_np)
L.fill(0)
with qd.ad.Tape(loss=L):
    scale2(x, y)
    loss_k(y, t, L)
qd.sync()
g2 = x.grad.to_numpy()
exp2 = 4.0 * (2.0 * x_np - t_np)
print("两核 tape x.grad:", np.array2string(g2, precision=3), " PASS" if np.allclose(g2, exp2, atol=1e-4) else " FAIL")
