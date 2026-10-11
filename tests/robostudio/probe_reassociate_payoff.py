"""Probe (M5 / C2-followup ①): source-level FP reassociation payoff on Metal.

探针目的
--------
C2(spirv-vectorize,已合入)证明:物理流式 kernel 被 DRAM 字带宽墙(~63 GB/s ≈ copy)
限制,bitwise 保序向量化在该墙 bound 的形状上只有 ~1.0-1.2x;但算力密集的 solver kernel
(行式 Cholesky 内积链,flop:byte = O(m):O(1))不撞墙 —— 它的瓶颈是**串行 FP 加法链**
(每拍只有 1 个加法在飞,Apple GPU fma/add 延迟 ≈ 4 cycle)。本探针在纯源码层做 A/B
(不改编译器):同一 kernel 的两个源码版 —— 顺序累加(s = s + a*b 链)vs 手写重结合
(4 个独立部分和 s0..s3 步距 4 累加 + 树形合并 + 标量尾),量化 `reassociate` opt-in
旗标在每个形状上能买到的收益与数值代价,为旗标语义(codegen 决策)提供数字。

预期结果(2026-10-11 五窗口实测后的校准版)
------------------------------------------
* 欠占用 + 长链(m≥48~64)形状:稳定收益 —— symmatvec m=64 ~2.1x、matvec m=48
  (per-world)~2.2-2.8x、chol_dot m=64 ~1.5-1.6x;输出与顺序版逐位不同,双方 vs f64
  的误差类不变(worst 1.8e-6,1e-5 门下余量 6x)。
* 短链(m=32/48 chol)≈无收益(0.84-1.34x);大 nworld 流式 dot 对照组 = 1.00x ± 0.01
  —— 复证 C2 墙理论(重结合只在墙外有收益)。
* 满占用 (w,i) 并行消融:跨窗口双峰且两臂互换(0.26x~3.91x),无方向性信号 ——
  计时伪迹(同源码跨运行 0.52ms↔2.0ms 双态),不作为旗标依据;方向上 TLP 已隐藏
  延迟时 ILP 无空间、多累加器寄存器压力反而有害。
* k=8 部分和一致差于 k=4(m=48)→ 旗标实现取 k=4。

如何解读
--------
两张表:perf 表(shape × GF/s × speedup,多轮交错取中位)+ numerics 表(vs f64 参考
max rel-err / f32-ulp / 与顺序版的逐位差异 %)。verdict 块输出旗标语义建议的量化输入:
(a) 哪些形状值得开 (b) 期望收益 (c) 数值代价 vs 1e-5 门。

写法地雷(全部规避)
--------------------
* A5(Apple MSL 编译器条件 RMW miscompile):所有内层循环都是**无条件**累加
  (s = s + x 赋值形式),循环体内无 if/continue 穿越累加语句。
* qd 的 range 只收 1-2 个参数(无 step):步距 4 写成 `for k4 in range(n4 // 4): k = k4*4`。
* `loop_config(serialize=True)` 不需要:无 SharedArray / qd.algorithms 调用,
  最外层 world 循环是唯一并行层,内层循环天然 per-thread 串行。
* Metal 无 f64:kernel 全 f32,参考解用 numpy float64。
* 无静态展开失控:m/n 以 i32 runtime 参数传入,所有循环 runtime 边界。
* 池指针 4B 对齐:标量访存,无向量载入假设。

Run: python probe_reassociate_payoff.py [metal] [--quick]   (GPU 空闲 + load 门;exit 0 = 跑通。
计时 = ABA 锚定轮:每轮 seq×2 夹 rea,两针差 >25% 的轮弃掉,收满 5 个有效轮取中位;
perf 尺度输出做两臂一致性门(防「快是因为算错」),绝对精度门在 nw=1 numerics 表)
Ledger: docs/robostudio_features.md "探针线 ①"。
"""
import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import quadrants as qd

ARCH = sys.argv[1] if len(sys.argv) > 1 else "metal"

if ARCH != "metal":
    print(f"reassoc payoff probe arch={ARCH}: SKIP (A/B is a Metal codegen question) -> exit 0")
    sys.exit(0)

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true", help="3 rounds instead of 5")
args, _ = p.parse_known_args()
ROUNDS = 3 if args.quick else 7

qd.init(arch=qd.metal, offline_cache=False)


def load1():
    out = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout
    return float(out.split()[1])


def gpu_idle_or_die():
    """GPU 与 R5 线共享 —— busy 则等(15s 重试,上限 5min),超时即失败退出。

    追加负载门:统一内存 Mac 上,重 CPU 负载(R5 native 构建)挤占 DRAM/SLC 带宽,
    实测把 rea 臂(每迭代 8 载入,MLP 更高)压得更狠 → 数字系统性偏向低估重结合。
    计时前等 load1 < ncpu/2;最多等 15 min,超时则带 LOADED 警告继续(数字=下界)。
    """
    script = Path(__file__).resolve().parent / "_gpu_busy.sh"
    ncpu = os.cpu_count() or 10
    thresh = max(2.0, ncpu / 2)
    deadline = time.monotonic() + 300
    while True:
        r = subprocess.run(["/bin/zsh", str(script)], capture_output=True, text=True)
        if r.returncode != 0:
            if time.monotonic() > deadline:
                print("FATAL: GPU busy >5min -> timing meaningless, exit 2")
                sys.exit(2)
            print("  gpu busy, retry in 15s ...")
            time.sleep(15)
            continue
        ld = load1()
        if ld < thresh:
            print(f"gpu idle, load1 {ld:.1f} < {thresh:.0f} -> clean timing window")
            return
        if time.monotonic() > deadline + 600:  # 15 min total
            global LOADED
            LOADED = True
            print(f"WARNING: load1 still {ld:.1f} >= {thresh:.0f} after 15 min -> continuing; "
                  f"timings are a LOWER BOUND on rea gain (MLP-sensitive arm punished by DRAM contention)")
            return
        print(f"  load1 {ld:.1f} >= {thresh:.0f} (native build contention), retry in 20s ...")
        time.sleep(20)


LOADED = False
gpu_idle_or_die()

fails = []


def gate(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL':4}] {name} {detail}")
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- kernels ----
# 每个 shape 两个源码版: _seq = 顺序累加链; _rea = 4 独立部分和 + 树形合并 + 标量尾。

@qd.kernel
def chol_dot_seq(L: qd.types.ndarray(ndim=3), o: qd.types.ndarray(ndim=3), m: qd.i32, niter: qd.i32):
    for w in range(L.shape[0]):
        for _ in range(niter):
            for i in range(m):
                for j in range(i):
                    s = 0.0
                    for k in range(j):
                        s = s + L[w, i, k] * L[w, j, k]
                    o[w, i, j] = s


@qd.kernel
def chol_dot_rea(L: qd.types.ndarray(ndim=3), o: qd.types.ndarray(ndim=3), m: qd.i32, niter: qd.i32):
    for w in range(L.shape[0]):
        for _ in range(niter):
            for i in range(m):
                for j in range(i):
                    n4 = j - j % 4
                    s0 = 0.0
                    s1 = 0.0
                    s2 = 0.0
                    s3 = 0.0
                    for k4 in range(n4 // 4):
                        k = k4 * 4
                        s0 = s0 + L[w, i, k] * L[w, j, k]
                        s1 = s1 + L[w, i, k + 1] * L[w, j, k + 1]
                        s2 = s2 + L[w, i, k + 2] * L[w, j, k + 2]
                        s3 = s3 + L[w, i, k + 3] * L[w, j, k + 3]
                    for k in range(n4, j):
                        s0 = s0 + L[w, i, k] * L[w, j, k]
                    o[w, i, j] = (s0 + s1) + (s2 + s3)


@qd.kernel
def matvec_seq(A: qd.types.ndarray(ndim=3), x: qd.types.ndarray(ndim=2), y: qd.types.ndarray(ndim=2),
               m: qd.i32, niter: qd.i32):
    for w in range(A.shape[0]):
        for _ in range(niter):
            for i in range(m):
                s = 0.0
                for j in range(m):
                    s = s + A[w, i, j] * x[w, j]
                y[w, i] = s


@qd.kernel
def matvec_rea(A: qd.types.ndarray(ndim=3), x: qd.types.ndarray(ndim=2), y: qd.types.ndarray(ndim=2),
               m: qd.i32, niter: qd.i32):
    for w in range(A.shape[0]):
        for _ in range(niter):
            for i in range(m):
                n4 = m - m % 4
                s0 = 0.0
                s1 = 0.0
                s2 = 0.0
                s3 = 0.0
                for k4 in range(n4 // 4):
                    k = k4 * 4
                    s0 = s0 + A[w, i, k] * x[w, k]
                    s1 = s1 + A[w, i, k + 1] * x[w, k + 1]
                    s2 = s2 + A[w, i, k + 2] * x[w, k + 2]
                    s3 = s3 + A[w, i, k + 3] * x[w, k + 3]
                for k in range(n4, m):
                    s0 = s0 + A[w, i, k] * x[w, k]
                y[w, i] = (s0 + s1) + (s2 + s3)


@qd.kernel
def matvec_rea8(A: qd.types.ndarray(ndim=3), x: qd.types.ndarray(ndim=2), y: qd.types.ndarray(ndim=2),
                m: qd.i32, niter: qd.i32):
    for w in range(A.shape[0]):
        for _ in range(niter):
            for i in range(m):
                n8 = m - m % 8
                s0 = 0.0
                s1 = 0.0
                s2 = 0.0
                s3 = 0.0
                s4 = 0.0
                s5 = 0.0
                s6 = 0.0
                s7 = 0.0
                for k8 in range(n8 // 8):
                    k = k8 * 8
                    s0 = s0 + A[w, i, k] * x[w, k]
                    s1 = s1 + A[w, i, k + 1] * x[w, k + 1]
                    s2 = s2 + A[w, i, k + 2] * x[w, k + 2]
                    s3 = s3 + A[w, i, k + 3] * x[w, k + 3]
                    s4 = s4 + A[w, i, k + 4] * x[w, k + 4]
                    s5 = s5 + A[w, i, k + 5] * x[w, k + 5]
                    s6 = s6 + A[w, i, k + 6] * x[w, k + 6]
                    s7 = s7 + A[w, i, k + 7] * x[w, k + 7]
                for k in range(n8, m):
                    s0 = s0 + A[w, i, k] * x[w, k]
                y[w, i] = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))


# matvec 的 (world,row) 并行布局:物理实现里 matvec 可以按行并行,每线程链更长可见 ——
# 消融「重结合收益是否在满占用下幸存」;对照 chol(行间依赖,不能并行)。
@qd.kernel
def matvec_rowpar_seq(A: qd.types.ndarray(ndim=3), x: qd.types.ndarray(ndim=2), y: qd.types.ndarray(ndim=2),
                      m: qd.i32, niter: qd.i32):
    total = A.shape[0] * m
    for idx in range(total):
        w = idx // m
        i = idx - w * m
        for _ in range(niter):
            s = 0.0
            for j in range(m):
                s = s + A[w, i, j] * x[w, j]
            y[w, i] = s


@qd.kernel
def matvec_rowpar_rea(A: qd.types.ndarray(ndim=3), x: qd.types.ndarray(ndim=2), y: qd.types.ndarray(ndim=2),
                      m: qd.i32, niter: qd.i32):
    total = A.shape[0] * m
    for idx in range(total):
        w = idx // m
        i = idx - w * m
        for _ in range(niter):
            n4 = m - m % 4
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            for k4 in range(n4 // 4):
                k = k4 * 4
                s0 = s0 + A[w, i, k] * x[w, k]
                s1 = s1 + A[w, i, k + 1] * x[w, k + 1]
                s2 = s2 + A[w, i, k + 2] * x[w, k + 2]
                s3 = s3 + A[w, i, k + 3] * x[w, k + 3]
            for k in range(n4, m):
                s0 = s0 + A[w, i, k] * x[w, k]
            y[w, i] = (s0 + s1) + (s2 + s3)


# 对称(Gram 形)matvec:只读下三角 + 对角 —— 物理 solver 的 J^T M J / A^T A 邻居形状。
@qd.kernel
def symmatvec_seq(L: qd.types.ndarray(ndim=3), x: qd.types.ndarray(ndim=2), y: qd.types.ndarray(ndim=2),
                  m: qd.i32, niter: qd.i32):
    for w in range(L.shape[0]):
        for _ in range(niter):
            for i in range(m):
                s = L[w, i, i] * x[w, i]
                for j in range(i):
                    s = s + L[w, i, j] * x[w, j]
                y[w, i] = s


@qd.kernel
def symmatvec_rea(L: qd.types.ndarray(ndim=3), x: qd.types.ndarray(ndim=2), y: qd.types.ndarray(ndim=2),
                  m: qd.i32, niter: qd.i32):
    for w in range(L.shape[0]):
        for _ in range(niter):
            for i in range(m):
                n4 = i - i % 4
                diag = L[w, i, i] * x[w, i]
                s0 = 0.0
                s1 = 0.0
                s2 = 0.0
                s3 = 0.0
                for k4 in range(n4 // 4):
                    k = k4 * 4
                    s0 = s0 + L[w, i, k] * x[w, k]
                    s1 = s1 + L[w, i, k + 1] * x[w, k + 1]
                    s2 = s2 + L[w, i, k + 2] * x[w, k + 2]
                    s3 = s3 + L[w, i, k + 3] * x[w, k + 3]
                for k in range(n4, i):
                    s0 = s0 + L[w, i, k] * x[w, k]
                y[w, i] = diag + ((s0 + s1) + (s2 + s3))


# 流式 dot(对照组):大 nworld,数据量超 SLC -> DRAM 字带宽墙(C2 定案 ~63 GB/s)。
@qd.kernel
def stream_dot_seq(a: qd.types.ndarray(ndim=2), b: qd.types.ndarray(ndim=2), o: qd.types.ndarray(ndim=1),
                   n: qd.i32, niter: qd.i32):
    for w in range(a.shape[0]):
        for _ in range(niter):
            s = 0.0
            for j in range(n):
                s = s + a[w, j] * b[w, j]
            o[w] = s


@qd.kernel
def stream_dot_rea(a: qd.types.ndarray(ndim=2), b: qd.types.ndarray(ndim=2), o: qd.types.ndarray(ndim=1),
                   n: qd.i32, niter: qd.i32):
    for w in range(a.shape[0]):
        for _ in range(niter):
            n4 = n - n % 4
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            for k4 in range(n4 // 4):
                k = k4 * 4
                s0 = s0 + a[w, k] * b[w, k]
                s1 = s1 + a[w, k + 1] * b[w, k + 1]
                s2 = s2 + a[w, k + 2] * b[w, k + 2]
                s3 = s3 + a[w, k + 3] * b[w, k + 3]
            for k in range(n4, n):
                s0 = s0 + a[w, k] * b[w, k]
            o[w] = (s0 + s1) + (s2 + s3)


# ---------------------------------------------------------------- helpers ----

def dev(a):
    t = qd.ndarray(dtype=qd.f32, shape=a.shape)
    t.from_numpy(np.ascontiguousarray(a))
    return t


def dev_zeros(shape):
    return dev(np.zeros(shape, dtype=np.float32))


def bits_to_ordered(u32):
    """IEEE bits (uint32) -> monotone int64 ordering (f32-ulp distances well-defined)."""
    sign = (u32.astype(np.uint32) >> np.uint32(31)).astype(np.int64)
    mag = (u32.astype(np.uint32) & np.uint32(0x7FFFFFFF)).astype(np.int64)
    return np.where(sign == 1, -mag, mag)


def numerics(res_f32, ref_f64):
    rel = np.abs(res_f32.astype(np.float64) - ref_f64) / np.maximum(np.abs(ref_f64), 1e-30)
    ulp = np.abs(bits_to_ordered(np.ascontiguousarray(res_f32).view(np.uint32))
                 - bits_to_ordered(np.ascontiguousarray(ref_f64.astype(np.float32)).view(np.uint32)))
    return float(rel.max()), int(ulp.max())


def bench_arm(kfn, launch_args, reps):
    kfn(*launch_args)  # warm
    qd.sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        kfn(*launch_args)
    qd.sync()
    return (time.perf_counter() - t0) / reps


def run_shape(sh):
    """Numerics (nw=1, niter=1) vs f64 reference, then interleaved seq/rea A/B timing."""
    b1 = sh["build"](1, 1)
    sh["seq"](*b1["args_s"])
    sh["rea"](*b1["args_r"])
    qd.sync()
    ref = sh["ref_fn"](b1["inputs"])
    r_seq, r_rea = b1["out_s"].to_numpy(), b1["out_r"].to_numpy()
    e_seq, u_seq = numerics(r_seq, ref)
    e_rea, u_rea = numerics(r_rea, ref)
    bits = int((r_seq.view(np.uint32) != r_rea.view(np.uint32)).sum())
    frac = 100.0 * bits / r_seq.size
    gate(f"{sh['name']} seq accuracy", e_seq < 1e-5, f"max rel-err {e_seq:.3e}, max ulp {u_seq}")
    gate(f"{sh['name']} rea accuracy", e_rea < 1e-5, f"max rel-err {e_rea:.3e}, max ulp {u_rea}")
    gate(f"{sh['name']} reassociation changes bits", bits > 0,
         f"{bits}/{r_seq.size} elements ({frac:.1f}%) differ vs seq")

    nw, niter, reps = sh["perf_nw"], sh["perf_niter"], sh["reps"]
    b2 = sh["build"](nw, niter)
    # ABA 锚定:每轮 seq 计时两次夹住 rea;两针差 >25% = 本轮撞上机器快/慢模式切换
    # (实测同 kernel 双峰 ~3.8x),整轮弃掉。收满 ACCEPT_ROUNDS 个有效轮为止。
    ACCEPT_ROUNDS = max(ROUNDS - 2, 3)
    ts, tr, dropped = [], [], 0
    for attempt in range(ACCEPT_ROUNDS * 3):
        t_s1 = bench_arm(sh["seq"], b2["args_s"], reps)
        t_r = bench_arm(sh["rea"], b2["args_r"], reps)
        t_s2 = bench_arm(sh["seq"], b2["args_s"], reps)
        if abs(t_s1 - t_s2) > 0.25 * min(t_s1, t_s2):
            dropped += 1
            continue
        ts.append((t_s1 + t_s2) / 2)
        tr.append(t_r)
        if len(ts) >= ACCEPT_ROUNDS:
            break
    t_seq, t_rea = statistics.median(ts), statistics.median(tr)

    # perf 尺度防「快是因为算错」:两臂输出必须一致(同数学不同求和序)。
    # vs-f64 的绝对精度门放 nw=1 numerics 表( perf 尺度多 world 取 max 会吃进分布尾巴)。
    o_seq32 = b2["out_s"].to_numpy()[:32]
    o_rea32 = b2["out_r"].to_numpy()[:32]
    denom = np.maximum(np.abs(np.asarray(sh["ref_fn"](b2["inputs"]))[:32]).max(), 1e-30)
    arm_gap = float(np.abs(o_rea32.astype(np.float64) - o_seq32.astype(np.float64)).max()) / denom
    gate(f"{sh['name']} perf-scale arms agree", arm_gap < 1e-5,
         f"max |rea-seq|/|ref|max {arm_gap:.3e}")
    if dropped:
        print(f"    ({dropped} rounds dropped by ABA anchor: machine mode switch mid-round)")
    fl = sh["flops"](nw, niter)
    g_seq, g_rea = fl / t_seq / 1e9, fl / t_rea / 1e9
    print(f"  {sh['name']:<18} seq {t_seq*1e3:8.3f} ms {g_seq:8.2f} GF/s | "
          f"rea {t_rea*1e3:8.3f} ms {g_rea:8.2f} GF/s | speedup {g_rea/g_seq:5.2f}x")
    return {"name": sh["name"], "m": sh["m"], "n": sh["n"], "nw": nw, "niter": niter,
            "g_seq": g_seq, "g_rea": g_rea, "t_seq": t_seq * 1e3, "t_rea": t_rea * 1e3,
            "speedup": g_rea / g_seq, "e_seq": e_seq, "e_rea": e_rea,
            "u_seq": u_seq, "u_rea": u_rea, "bits_pct": frac}


# ---------------------------------------------------------------- shapes ----

def mk_chol(m, perf_nw, perf_niter, reps=3):
    """行式 Cholesky 内积链(solver 代表):每个 (w,i) 线程对每个 j<i 重算
    dot(L[i,:j], L[j,:j]) —— 与逐行分解的内积更新完全同构;L 只读、o 不相交 → 无竞争。
    数据取正(PSD/Gram 形,物理 solver 的实际符号结构):无对消,f32 误差 = 纯累加舍入。"""

    def build(nw, niter):
        rng = np.random.default_rng(100 + m)
        A = rng.uniform(0.5, 1.0, (nw, m, m)).astype(np.float32)
        Lnp = np.ascontiguousarray(np.tril(A) + np.diag(np.full(m, float(m), np.float32)))
        Ld = dev(Lnp)
        o_s, o_r = dev_zeros(Lnp.shape), dev_zeros(Lnp.shape)
        return {"args_s": (Ld, o_s, m, niter), "args_r": (Ld, o_r, m, niter),
                "out_s": o_s, "out_r": o_r, "inputs": (Lnp,)}

    def ref_fn(inputs):
        L64 = inputs[0].astype(np.float64)
        ref = np.zeros_like(L64)
        for w in range(L64.shape[0]):
            for i in range(m):
                for j in range(i):
                    ref[w, i, j] = np.dot(L64[w, i, :j], L64[w, j, :j])
        return ref

    flops = lambda nw, niter: nw * niter * (m - 1) * m * (m - 2) / 3  # 2 flops × Σ_i Σ_{j<i} j
    return dict(name=f"chol_dot m={m}", seq=chol_dot_seq, rea=chol_dot_rea, build=build,
                ref_fn=ref_fn, flops=flops, reps=reps, perf_nw=perf_nw, perf_niter=perf_niter,
                m=m, n=0)


def _mk_matvec_base(name, seq_k, rea_k, m, perf_nw, perf_niter, reps=4):
    def build(nw, niter):
        rng = np.random.default_rng(200 + m)
        A = (rng.standard_normal((nw, m, m)) * 0.1).astype(np.float32)
        x = rng.standard_normal((nw, m)).astype(np.float32)
        Ad, xd = dev(A), dev(x)
        y_s, y_r = dev_zeros((nw, m)), dev_zeros((nw, m))
        return {"args_s": (Ad, xd, y_s, m, niter), "args_r": (Ad, xd, y_r, m, niter),
                "out_s": y_s, "out_r": y_r, "inputs": (A, x)}

    def ref_fn(inputs):
        return np.einsum("wij,wj->wi", inputs[0].astype(np.float64), inputs[1].astype(np.float64))

    flops = lambda nw, niter: nw * niter * m * m * 2
    return dict(name=name, seq=seq_k, rea=rea_k, build=build,
                ref_fn=ref_fn, flops=flops, reps=reps, perf_nw=perf_nw, perf_niter=perf_niter,
                m=m, n=0)


def mk_matvec(m, perf_nw, perf_niter, reps=4, k=4):
    if k == 8:
        return _mk_matvec_base(f"matvec m={m} k=8", matvec_seq, matvec_rea8, m, perf_nw, perf_niter, reps)
    return _mk_matvec_base(f"matvec m={m}", matvec_seq, matvec_rea, m, perf_nw, perf_niter, reps)


def mk_matvec_rowpar(m, perf_nw, perf_niter, reps=4):
    return _mk_matvec_base(f"matvec(w,i) m={m}", matvec_rowpar_seq, matvec_rowpar_rea, m, perf_nw, perf_niter, reps)


def mk_sym(m, perf_nw, perf_niter, reps=4):
    def build(nw, niter):
        rng = np.random.default_rng(300 + m)
        A = (rng.standard_normal((nw, m, m)) * 0.1).astype(np.float32)
        Lnp = np.ascontiguousarray(np.tril(A))
        x = rng.standard_normal((nw, m)).astype(np.float32)
        Ld, xd = dev(Lnp), dev(x)
        y_s, y_r = dev_zeros((nw, m)), dev_zeros((nw, m))
        return {"args_s": (Ld, xd, y_s, m, niter), "args_r": (Ld, xd, y_r, m, niter),
                "out_s": y_s, "out_r": y_r, "inputs": (Lnp, x)}

    def ref_fn(inputs):
        L64 = np.tril(inputs[0].astype(np.float64))  # kernel 只读下三角+对角
        return np.einsum("wij,wj->wi", L64, inputs[1].astype(np.float64))

    flops = lambda nw, niter: nw * niter * m * (m + 1)
    return dict(name=f"symmatvec m={m}", seq=symmatvec_seq, rea=symmatvec_rea, build=build,
                ref_fn=ref_fn, flops=flops, reps=reps, perf_nw=perf_nw, perf_niter=perf_niter,
                m=m, n=0)


def mk_stream(n, perf_nw, perf_niter, reps=2):
    def build(nw, niter):
        rng = np.random.default_rng(400 + n)
        a = (rng.standard_normal((nw, n)) * 0.1).astype(np.float32)
        b = (rng.standard_normal((nw, n)) * 0.1).astype(np.float32)
        ad, bd = dev(a), dev(b)
        o_s, o_r = dev_zeros((nw,)), dev_zeros((nw,))
        return {"args_s": (ad, bd, o_s, n, niter), "args_r": (ad, bd, o_r, n, niter),
                "out_s": o_s, "out_r": o_r, "inputs": (a, b)}

    def ref_fn(inputs):
        return np.einsum("wj,wj->w", inputs[0].astype(np.float64), inputs[1].astype(np.float64))

    flops = lambda nw, niter: nw * niter * n * 2
    return dict(name=f"stream_dot n={n}", seq=stream_dot_seq, rea=stream_dot_rea, build=build,
                ref_fn=ref_fn, flops=flops, reps=reps, perf_nw=perf_nw, perf_niter=perf_niter,
                m=0, n=n)


SHAPES = [
    mk_chol(32, perf_nw=4096, perf_niter=2),
    mk_chol(48, perf_nw=2048, perf_niter=2),
    mk_chol(64, perf_nw=1024, perf_niter=2),
    mk_matvec(48, perf_nw=1024, perf_niter=8),
    mk_matvec(48, perf_nw=1024, perf_niter=8, k=8),
    mk_matvec_rowpar(48, perf_nw=1024, perf_niter=8),
    mk_sym(64, perf_nw=512, perf_niter=8),
    mk_stream(1024, perf_nw=16384, perf_niter=1),
]

print(f"== PROBE ① reassociation payoff (Metal, {time.strftime('%F %T')}) ==")
print("dev55 native @68bd7965, spirv_vectorize=OFF, offline_cache=False, source-level A/B")
print(f"rounds={ROUNDS} interleaved seq/rea, per-arm median; GPU-idle + load gate passed" + (" [LOADED: lower-bound timings]" if LOADED else "") + "\n")

results = [run_shape(sh) for sh in SHAPES]

print("\n=== PERF TABLE (median; single-round machine variance up to ±4x, hence interleaved rounds) ===")
hdr = (f"{'shape':<18}{'m':>4}{'n':>6}{'nw':>7}{'niter':>6} | {'seq GF/s':>9} {'rea GF/s':>9} "
       f"{'speedup':>8} | {'seq ms':>9} {'rea ms':>9}")
print(hdr)
print("-" * len(hdr))
for r in results:
    print(f"{r['name']:<18}{r['m']:>4}{r['n']:>6}{r['nw']:>7}{r['niter']:>6} | "
          f"{r['g_seq']:>9.2f} {r['g_rea']:>9.2f} {r['speedup']:>7.2f}x | {r['t_seq']:>9.3f} {r['t_rea']:>9.3f}")

print("\n=== NUMERICS TABLE (nw=1 vs numpy float64 reference; physics gate 1e-5 rel) ===")
hdr2 = f"{'shape':<18}{'seq rel-err':>12}{'rea rel-err':>12}{'seq ulp':>9}{'rea ulp':>9}{'bits-diff %':>12}"
print(hdr2)
print("-" * len(hdr2))
for r in results:
    print(f"{r['name']:<18}{r['e_seq']:>12.3e}{r['e_rea']:>12.3e}{r['u_seq']:>9}{r['u_rea']:>9}{r['bits_pct']:>11.1f}")

print("\n=== VERDICT INPUTS (auto-derived) ===")
gains = [r for r in results if r["speedup"] >= 1.3]
ctrl = [r for r in results if r["speedup"] < 1.3]
max_err = max(max(r["e_seq"], r["e_rea"]) for r in results)
print(f"shapes with >=1.3x gain : {', '.join(r['name'] for r in gains) or 'none'}")
print(f"control shapes (<1.3x)  : {', '.join(r['name'] for r in ctrl) or 'none'}")
print(f"worst numeric error any arm/shape: {max_err:.3e} -> margin to 1e-5 physics gate: {1e-5/max_err:.0f}x")

print(f"\nREASSOC PAYOFF PROBE: {'ALL GATES PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
