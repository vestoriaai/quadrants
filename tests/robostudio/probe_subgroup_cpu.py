"""Probe (M5 item 4, CPU arm): qd.simt.subgroup + qd.simt.block on the CPU backend.

探针目的
--------
subgroup 线的 CPU 臂。Metal 臂(probe_subgroup_metal.py)钉的是 32-lane 全栈可用;CPU 侧
走的是**两套并存语义**(新契约,docs/robostudio_features.md "mujocoqd/subgroup-cpu-support"
条目):

1. **subgroup 层 = 虚拟组语义(group_size=1)**:每 CPU 线程 = 自己的 subgroup。所有
   subgroup 操作退化为恒等式(shuffle/broadcast = identity、reduce = 自身、scan = 平凡、
   ballot = bit0、bitonic_sort_kv = 恒等网络),参考解按该语义重算,验收**逐位一致**。
   reductions.py / sorting.py 的 shuffle-butterfly 在 log2_group_size()==0 下静态展开为空,
   与 GPU 共用同一代码路径 —— 本探针同时验证该退化展开的正确性。
2. **block 层 = 真并发虚拟 block**:`loop_config(block_dim=N)` + SharedArray/block.sync/
   block 集体操作 → 每 block 的 N 个虚拟线程真实并发(一线程一虚拟线程),barrier 是真
   屏障(收敛 + happens-before),SharedArray 是 block 窗口。参考解按 GPU block 语义
   (write→sync→置换读、block.reduce 全折叠):整数逐位,f32 用 1e-7 容差桶 + 同源
   顺序折叠的逐位确定性断言。

为何单独成文件而非扩 probe_subgroup_metal.py:Metal 臂的参考解全部以 lane=w%32 /
32-lane 群基址为前提(group_size=32),CPU 臂的参考解以 group_size=1 恒等式为前提,
逐 API 的期望值两臂不同;分开文件让 Metal 臂的 ledger 钉死输出(24/24)保持字节稳定,
且两后端一进程一 init(C0 纪律)。

Run: python probe_subgroup_cpu.py [cpu]   (exit 0 = pass)
Ledger: docs/robostudio_features.md "mujocoqd/subgroup-cpu-support"。
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

import quadrants as qd
from quadrants.lang.misc import loop_config
from quadrants.lang.simt import block as sg_block
from quadrants.lang.simt import subgroup as sg

ARCH = sys.argv[1] if len(sys.argv) > 1 else "cpu"
if ARCH != "cpu":
    print(f"subgroup cpu probe arch={ARCH}: SKIP (probe targets the CPU stack) -> exit 0")
    sys.exit(0)

qd.init(arch=qd.cpu, offline_cache=False)

fails = []
NW = 256  # top-level linear dispatch width; block-tier kernels use block_dim=256 (1 block) or 32 (8 blocks)


def gate(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL':4}] {name} {detail}")
    if not ok:
        fails.append(name)


def run_and_fetch(kernel, out, *ins):
    kernel(out, *ins)
    qd.sync()
    return out.to_numpy()


def i1d(np_arr):
    t = qd.ndarray(dtype=qd.i32, shape=(len(np_arr),))
    t.from_numpy(np.ascontiguousarray(np_arr.astype(np.int32)))
    return t


o_i = qd.ndarray(dtype=qd.i32, shape=(NW,))
o_u = qd.ndarray(dtype=qd.u32, shape=(NW,))
o_u64 = qd.ndarray(dtype=qd.u64, shape=(NW,))
o_f = qd.ndarray(dtype=qd.f32, shape=(NW,))

# ------------------------------------------------------------- host-side group size ----
gs, lg = sg.group_size(), sg.log2_group_size()
gate("host group_size==1", gs == 1, f"got {gs}")
gate("host log2_group_size==0", lg == 0, f"got {lg}")

# ------------------------------------------------------------- 1 identity / vote ----
# 虚拟组语义:invocation_id 恒 0,elect 恒 1。


@qd.kernel
def k_invocation_id(o: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.invocation_id()


@qd.kernel
def k_elect(o: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.elect()


gate("invocation_id == 0 (bitwise)", np.array_equal(run_and_fetch(k_invocation_id, o_i), np.zeros(NW, np.int32)))
gate("elect == 1 (bitwise)", np.array_equal(run_and_fetch(k_elect, o_i), np.ones(NW, np.int32)))

# ------------------------------------------------------------- 2 data movement ----
# shuffle/broadcast 全系 identity(group_size=1 下唯一可读 lane = 自己);index/offset 实参
# 仍按 u32 契约显式转换(与 Metal 臂同一人机工学)。
v_np = ((np.arange(NW) * 7 + 3) % 1000).astype(np.int32)
v_i = i1d(v_np)


@qd.kernel
def k_shuffle(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.shuffle(v[w], qd.u32(0))


@qd.kernel
def k_shuffle_xor(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.shuffle_xor(v[w], 3)


@qd.kernel
def k_shuffle_up(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.shuffle_up(v[w], qd.u32(1))


@qd.kernel
def k_shuffle_down(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.shuffle_down(v[w], qd.u32(3))


@qd.kernel
def k_broadcast(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.broadcast(v[w], qd.u32(0))


@qd.kernel
def k_broadcast_first(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.broadcast_first(v[w])


for name, k in (("shuffle(v,0)", k_shuffle), ("shuffle_xor(v,3)", k_shuffle_xor), ("shuffle_up(v,1)", k_shuffle_up),
                ("shuffle_down(v,3)", k_shuffle_down), ("broadcast(v,0)", k_broadcast),
                ("broadcast_first(v)", k_broadcast_first)):
    gate(f"{name} == identity (bitwise)", np.array_equal(run_and_fetch(k, o_i, v_i), v_np))

# ------------------------------------------------------------- 3 ballot / votes ----
pred_np = (np.arange(NW) % 3 == 0).astype(np.int32)
pred_i = i1d(pred_np)


@qd.kernel
def k_ballot_first_n(o: qd.types.ndarray(ndim=1), pred: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.ballot_first_n(pred[w], 32)


@qd.kernel
def k_ballot_u64(o: qd.types.ndarray(ndim=1), pred: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.ballot(pred[w])


@qd.kernel
def k_all_true(o: qd.types.ndarray(ndim=1), pred: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.all_true(pred[w])


@qd.kernel
def k_any_true(o: qd.types.ndarray(ndim=1), pred: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.any_true(pred[w])


@qd.kernel
def k_all_equal(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.all_equal(v[w])


want_bit = pred_np.astype(np.int64)  # bit0 = 自身谓词
gate("ballot_first_n(p,32) == bit0 (bitwise)", np.array_equal(run_and_fetch(k_ballot_first_n, o_u, pred_i), want_bit))
gate("ballot(p) u64 == bit0 (bitwise)", np.array_equal(run_and_fetch(k_ballot_u64, o_u64, pred_i), want_bit))
gate("all_true(p) == p (bitwise)", np.array_equal(run_and_fetch(k_all_true, o_i, pred_i), pred_np))
gate("any_true(p) == p (bitwise)", np.array_equal(run_and_fetch(k_any_true, o_i, pred_i), pred_np))
gate("all_equal(v) == 1 (bitwise)", np.array_equal(run_and_fetch(k_all_equal, o_i, v_i), np.ones(NW, np.int32)))

# ------------------------------------------------------------- 4 reduce / scan / segmented ----
# 全部退化为恒等:reduce=自身、inclusive=自身、exclusive=加法单位元、segmented=自身。
# 输入非平凡(1..32 循环),证明没有被特化成 0。
lane1 = (np.arange(NW) % 32 + 1).astype(np.int32)
v_r = i1d(lane1)


@qd.kernel
def k_reduce_add(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.reduce_add(v[w])


@qd.kernel
def k_reduce_all_add(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.reduce_all_add(v[w])


@qd.kernel
def k_reduce_min(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.reduce_min(v[w])


@qd.kernel
def k_reduce_max(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.reduce_max(v[w])


@qd.kernel
def k_inclusive_add(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.inclusive_add(v[w])


@qd.kernel
def k_exclusive_add(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.exclusive_add(v[w])


@qd.kernel
def k_segmented(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1), head: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.segmented_reduce_add(v[w], head[w])


gate("reduce_add == self (bitwise)", np.array_equal(run_and_fetch(k_reduce_add, o_i, v_r), lane1))
gate("reduce_all_add == self (bitwise)", np.array_equal(run_and_fetch(k_reduce_all_add, o_i, v_r), lane1))
gate("reduce_min == self (bitwise)", np.array_equal(run_and_fetch(k_reduce_min, o_i, v_r), lane1))
gate("reduce_max == self (bitwise)", np.array_equal(run_and_fetch(k_reduce_max, o_i, v_r), lane1))
gate("inclusive_add == self (bitwise)", np.array_equal(run_and_fetch(k_inclusive_add, o_i, v_r), lane1))
gate("exclusive_add == 0 (bitwise)", np.array_equal(run_and_fetch(k_exclusive_add, o_i, v_r), np.zeros(NW, np.int32)))
gate("segmented_reduce_add == self (bitwise)",
     np.array_equal(run_and_fetch(k_segmented, o_i, v_r, i1d((np.arange(NW) % 8 == 0).astype(np.int32))), lane1))

# ------------------------------------------------------------- 5 lanemask / sort ----
# lanemask 是纯算术 wrapper(全后端同源);bitonic 在 log2=0 是文档化的恒等网络。


@qd.kernel
def k_lanemask_lt(o: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.lanemask_lt(sg.invocation_id())


@qd.kernel
def k_sort_kv(ok: qd.types.ndarray(ndim=1), ov: qd.types.ndarray(ndim=1),
              key: qd.types.ndarray(ndim=1), val: qd.types.ndarray(ndim=1)):
    for w in range(ok.shape[0]):
        sk, sv = sg.bitonic_sort_kv(key[w], val[w])
        ok[w] = sk
        ov[w] = sv


gate("lanemask_lt(0) == 0 (bitwise)", np.array_equal(run_and_fetch(k_lanemask_lt, o_i), np.zeros(NW, np.int32)))
key_np = ((np.arange(NW) * 13) % 97).astype(np.int32)
val_np = ((np.arange(NW) * 5) % 89).astype(np.int32)
ok_arr = qd.ndarray(dtype=qd.i32, shape=(NW,))
ov_arr = qd.ndarray(dtype=qd.i32, shape=(NW,))
k_sort_kv(ok_arr, ov_arr, i1d(key_np), i1d(val_np))
qd.sync()
gate("bitonic_sort_kv == identity (bitwise)",
     np.array_equal(ok_arr.to_numpy(), key_np) and np.array_equal(ov_arr.to_numpy(), val_np))

# ------------------------------------------------------------- 6 block-tier: sync + SharedArray ----
# GPU block 语义逐字兑现:block_dim=256 虚拟线程并发,write→sync→置换读。
# 参考解:sh[tid]=255-tid; sync; out=sh[(tid*7)%256] -> out[i] = 255-(i*7)%256(单 block)。


@qd.kernel
def k_block_exchange(o: qd.types.ndarray(ndim=1)):
    sh = sg_block.SharedArray((256,), qd.i32)
    loop_config(block_dim=256)
    for w in range(o.shape[0]):
        tid = sg_block.thread_idx()
        sh[tid] = 255 - tid
        sg_block.sync()
        o[w] = sh[(tid * 7) % 256]


got = run_and_fetch(k_block_exchange, o_i)
want = (255 - (np.arange(NW) * 7) % 256).astype(np.int32)
gate("block.sync + SharedArray permuted exchange (bitwise)", np.array_equal(got, want))


# 跨 block 窗口隔离:block_dim=32,NW=256 -> 8 个 block,各窗口独立;读槽由「前一个」虚拟
# 线程写入(写读之间隔着屏障),没有真屏障的实现会读到 0 或上一 block 残留。
@qd.kernel
def k_block_window_isolation(o: qd.types.ndarray(ndim=1)):
    sh = sg_block.SharedArray((32,), qd.i32)
    loop_config(block_dim=32)
    for w in range(o.shape[0]):
        tid = sg_block.thread_idx()
        sh[(tid + 1) % 32] = tid * 2
        sg_block.sync()
        o[w] = sh[tid]


got = run_and_fetch(k_block_window_isolation, o_i)
want = (((np.arange(NW) % 32 + 31) % 32) * 2).astype(np.int32)
gate("block_dim=32 x 8 blocks, window isolation (bitwise)", np.array_equal(got, want))


# 默认 block_dim(CPU=32,与 GPU 缺省 128 同策略)下不显式 loop_config 也成立:同形置换读。
@qd.kernel
def k_block_default_dim(o: qd.types.ndarray(ndim=1)):
    sh = sg_block.SharedArray((32,), qd.i32)
    for w in range(o.shape[0]):
        tid = sg_block.thread_idx()
        sh[(tid + 3) % 32] = tid * 5
        sg_block.sync()
        o[w] = sh[tid]


got = run_and_fetch(k_block_default_dim, o_i)
want = (((np.arange(NW) % 32 + 29) % 32) * 5).astype(np.int32)
gate("default block_dim=32 (no loop_config) exchange (bitwise)", np.array_equal(got, want))

# ------------------------------------------------------------- 7 block collectives ----
# block.py 骨架在 SUBGROUP_SIZE=1 下自动正确:每线程 publish 自槽 + 屏障 + thread-0 顺序折叠。
# 整数逐位;f32 为同源顺序折叠(确定性),对照 numpy f64 参考 1e-7 容差桶。


@qd.kernel
def k_block_reduce_all_add_i32(o: qd.types.ndarray(ndim=1)):
    loop_config(block_dim=256)
    for w in range(o.shape[0]):
        o[w] = sg_block.reduce_all_add(qd.i32(1), 256, qd.i32)


got = run_and_fetch(k_block_reduce_all_add_i32, o_i)
gate("block.reduce_all_add(1, 256) == 256 (bitwise)", np.array_equal(got, np.full(NW, 256, np.int32)))


@qd.kernel
def k_block_inclusive_add_i32(o: qd.types.ndarray(ndim=1)):
    loop_config(block_dim=256)
    for w in range(o.shape[0]):
        tid = sg_block.thread_idx()
        o[w] = sg_block.inclusive_add(tid + 1, 256, qd.i32)


got = run_and_fetch(k_block_inclusive_add_i32, o_i)
want = ((np.arange(NW) + 1) * (np.arange(NW) + 2) // 2).astype(np.int32)
gate("block.inclusive_add(tid+1, 256) prefix (bitwise)", np.array_equal(got, want))


@qd.kernel
def k_block_exclusive_min_i32(o: qd.types.ndarray(ndim=1)):
    loop_config(block_dim=256)
    for w in range(o.shape[0]):
        tid = sg_block.thread_idx()
        o[w] = sg_block.exclusive_min(tid + 7, 256, qd.i32)


got = run_and_fetch(k_block_exclusive_min_i32, o_i)
want = np.full(NW, 7, np.int32)
want[0] = np.iinfo(np.int32).max  # thread 0 holds the dtype-derived identity
gate("block.exclusive_min(tid+7, 256) (bitwise)", np.array_equal(got, want))


@qd.kernel
def k_block_exclusive_max_i32(o: qd.types.ndarray(ndim=1)):
    loop_config(block_dim=256)
    for w in range(o.shape[0]):
        tid = sg_block.thread_idx()
        o[w] = sg_block.exclusive_max(tid, 256, qd.i32)


got = run_and_fetch(k_block_exclusive_max_i32, o_i)
want = np.concatenate([[np.iinfo(np.int32).min], np.arange(NW - 1)]).astype(np.int32)
gate("block.exclusive_max(tid, 256) (bitwise)", np.array_equal(got, want))


@qd.kernel
def k_block_sync_count_nonzero(o: qd.types.ndarray(ndim=1)):
    loop_config(block_dim=256)
    for w in range(o.shape[0]):
        tid = sg_block.thread_idx()
        o[w] = sg_block.sync_count_nonzero(qd.select(tid % 4 == 0, 1, 0))


got = run_and_fetch(k_block_sync_count_nonzero, o_i)
gate("block.sync_count_nonzero == 64 (bitwise)", np.array_equal(got, np.full(NW, 64, np.int32)))

# f32 block reduce:CPU 臂 = thread-0 顺序折叠(确定),GPU 臂(Metal)为树形归约(重结合桶)。
# 对照 numpy f64 参考;另断言同 kernel 两次 launch 输出逐位一致(CPU 顺序折叠的确定性)。


@qd.kernel
def k_block_reduce_all_add_f32(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    loop_config(block_dim=256)
    for w in range(o.shape[0]):
        o[w] = sg_block.reduce_all_add(v[w], 256, qd.f32)


vf_np = (np.arange(NW, dtype=np.float32) % 256) * np.float32(0.37) + np.float32(0.11)
vf = qd.ndarray(dtype=qd.f32, shape=(NW,))
vf.from_numpy(vf_np)
run_and_fetch(k_block_reduce_all_add_f32, o_f, vf)
first = o_f.to_numpy().copy()
run_and_fetch(k_block_reduce_all_add_f32, o_f, vf)
second = o_f.to_numpy()
ref = float((np.arange(256, dtype=np.float64) * 0.37 + 0.11).sum())
rel = abs(float(first[0]) - ref) / abs(ref)
gate("block.reduce_all_add f32 determinism (launch-twice bitwise)", np.array_equal(first, second))
gate("block.reduce_all_add f32 vs f64 ref (1e-7 bucket)", rel <= 1e-7, f"got {first[0]:.7e} ref {ref:.7e} rel {rel:.2e}")

# ------------------------------------------------------------- 8 subgroup.sync no-op inside block-tier ----
# 虚拟组语义下 subgroup.sync 是 no-op;与 block-tier 混用时不得破坏屏障链。


@qd.kernel
def k_subgroup_sync_in_block(o: qd.types.ndarray(ndim=1)):
    sh = sg_block.SharedArray((32,), qd.i32)
    loop_config(block_dim=32)
    for w in range(o.shape[0]):
        tid = sg_block.thread_idx()
        sh[(tid + 1) % 32] = tid * 3
        sg.sync()  # no-op on CPU (subgroup == 1 thread)
        sg_block.sync()
        o[w] = sh[tid]


got = run_and_fetch(k_subgroup_sync_in_block, o_i)
want = (((np.arange(NW) % 32 + 31) % 32) * 3).astype(np.int32)
gate("subgroup.sync no-op + block.sync exchange (bitwise)", np.array_equal(got, want))

# ------------------------------------------------------------- 9 loud rejections (fresh subprocess) ----
# 每个拒绝路径一个子进程(一进程一 init);错误必须响亮且可枚举,静默错值 = FAIL。
# 注:无屏障且被 LLVM 消除/平凡化的 SharedArray kernel 不进 block-tier 派发(纯本地访问被
# 优化掉是安全的——没有屏障的 kernel 本就没有跨线程可见性契约);拒绝探针因此都带 block.sync,
# 保证真正走到 block-tier 路径再验拒绝。
_SUBGROUP_PROBES = {
    "reject_block_tier_on_serial_task": (
        """
import quadrants as qd
from quadrants.lang.misc import loop_config
from quadrants.lang.simt import block as _block
qd.init(arch=qd.cpu)
i32_1d = qd.types.ndarray(qd.i32, ndim=1)

@qd.kernel
def k(out: i32_1d):
    sh = _block.SharedArray((16,), qd.i32)
    loop_config(serialize=True)
    for i in range(16):
        sh[i] = i
        _block.sync()
        out[i] = sh[15 - i]

t = qd.ndarray(qd.i32, shape=16)
try:
    k(t)
    print("MARKER_NO_ERROR")
except BaseException as e:
    print("MARKER_REJECTED" if "block-tier" in str(e) else "MARKER_OTHER")
""",
        "MARKER_REJECTED",
    ),
    "reject_rand_in_block_tier": (
        """
import quadrants as qd
from quadrants.lang.misc import loop_config
from quadrants.lang.simt import block as _block
qd.init(arch=qd.cpu)
f32_1d = qd.types.ndarray(qd.f32, ndim=1)

@qd.kernel
def k(out: f32_1d):
    sh = _block.SharedArray((8,), qd.i32)
    loop_config(block_dim=8)
    for i in range(8):
        sh[i] = 0
        _block.sync()
        out[i] = qd.random()

t = qd.ndarray(qd.f32, shape=8)
try:
    k(t)
    print("MARKER_NO_ERROR")
except BaseException as e:
    print("MARKER_REJECTED" if "rand" in str(e).lower() else "MARKER_OTHER")
""",
        "MARKER_REJECTED",
    ),
    "reject_block_dim_over_1024": (
        """
import quadrants as qd
from quadrants.lang.misc import loop_config
from quadrants.lang.simt import block as _block
qd.init(arch=qd.cpu)
i32_1d = qd.types.ndarray(qd.i32, ndim=1)

@qd.kernel
def k(out: i32_1d):
    sh = _block.SharedArray((2048,), qd.i32)
    loop_config(block_dim=2048)
    for i in range(2048):
        sh[i] = i
        _block.sync()
        out[i] = sh[i]

t = qd.ndarray(qd.i32, shape=2048)
try:
    k(t)
    print("MARKER_NO_ERROR")
except BaseException as e:
    print("MARKER_REJECTED" if "block_dim" in str(e) else "MARKER_OTHER")
""",
        "MARKER_REJECTED",
    ),
    "reject_global_thread_idx": (
        """
import quadrants as qd
from quadrants.lang.simt import block as _block
qd.init(arch=qd.cpu)

@qd.kernel
def k(n: qd.i32):
    for i in range(n):
        _block.global_thread_idx()

try:
    k(4)
    print("MARKER_NO_ERROR")
except BaseException as e:
    print("MARKER_REJECTED" if "global_thread_idx is not supported" in str(e) else "MARKER_OTHER")
""",
        "MARKER_REJECTED",
    ),
}

print()
for probe_name, (src, want_marker) in _SUBGROUP_PROBES.items():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=300,
                           env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")})
        got = want_marker in r.stdout
        if got:
            gate(f"expected_error {probe_name}", True)
        else:
            out = r.stdout.strip().splitlines()
            err = r.stderr.strip().splitlines()
            gate(f"expected_error {probe_name}", False,
                 f"stdout tail: {out[-1] if out else (err[-1] if err else 'no output')}")
    finally:
        os.unlink(path)

# ---------------------------------------------------------------- verdict ----
print(f"\nSUBGROUP CPU PROBE: {'ALL PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
