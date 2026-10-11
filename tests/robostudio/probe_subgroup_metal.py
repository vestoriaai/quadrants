"""Probe (M5 item 4): qd.simt.subgroup availability on Metal (SPIR-V -> SPIRV-Cross -> MSL).

探针目的
--------
M5 第 4 项(subgroup 归约)的前置问题:`simt.subgroup` 的 API 面(SPIR-V 侧发射
OpGroupNonUniform*,见 spirv_codegen.cpp L1499-1620 与 spirv_ir_builder.cpp L95-135;
Metal 侧 cap 映射 metal_device.mm L1106-1125:Apple6+/Mac2+ 开 vote/ballot、Apple7+/Mac2+
开 arithmetic,subgroup_size 钉 32)在 **Metal 全栈**(SPIR-V -> MSL 2.1/2.3 -> Apple GPU)
上到底可不可以用、数值对不对、值不值得用来替换 atomic_add 链。CPU 后端明确不可用
(block.sync ValueError / subgroup size 0 assert —— 那是 CPU,本探针只验 Metal)。

预期结果
--------
* 预期全部可用(block/radix-sort 层在 Metal 已在产线使用,而那层就是用 subgroup 搭的);
  唯一存疑点是 `ballot()` 的 u64 形(发射 u64 打包,Metal i64 走 MSL 2.3;若 SPIRV-Cross
  或驱动拒绝则输出 handback 规格,`ballot_first_n` u32 形为后备)。
* perf pin:atomic_add 链(每元素一次原子)vs subgroup.reduce_add + elect 单原子,
  预期 10x+ 级差(Metal 原子是 L2 round-trip)。

如何解读
--------
每行 = 一个 API:[PASS] 数值 vs 串行/宿主参考精确一致;[FAIL] 可用但数值错(附最大
偏差,exit 1);[RAISED] 不可用,附最小复现错误原文 —— 结尾按 C5 block-tier 三层证据链
格式输出 handback 规格(哪个 API、SPIR-V 侧发射什么、错误原文)。数值语义单列一节:
subgroup.reduce 本质 = 并行归约换序(重结合同桶),与 probe_reassociate_payoff.py 的
旗标语义结论绑定;本探针不落任何生产路径。

Run: python probe_subgroup_metal.py [metal]   (exit 0 = 无数值 FAIL;RAISED 项即 handback,
本来就是可用性交付物)
Ledger: docs/robostudio_features.md "探针线 ②"。
"""
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import quadrants as qd
from quadrants.lang.misc import loop_config
from quadrants.lang.simt import block as sg_block
from quadrants.lang.simt import subgroup as sg

ARCH = sys.argv[1] if len(sys.argv) > 1 else "metal"
if ARCH != "metal":
    print(f"subgroup metal probe arch={ARCH}: SKIP (probe targets Metal stack) -> exit 0")
    sys.exit(0)

qd.init(arch=qd.metal, offline_cache=False)


def gpu_idle_or_die():
    script = Path(__file__).resolve().parent / "_gpu_busy.sh"
    deadline = time.monotonic() + 300
    while True:
        r = subprocess.run(["/bin/zsh", str(script)], capture_output=True, text=True)
        if r.returncode == 0:
            return
        if time.monotonic() > deadline:
            print("FATAL: GPU busy >5min -> exit 2")
            sys.exit(2)
        print("  gpu busy, retry in 15s ...")
        time.sleep(15)


gpu_idle_or_die()

fails, raised = [], []
NP_D2QD = {np.dtype(np.int32): qd.i32, np.dtype(np.uint32): qd.u32,
           np.dtype(np.float32): qd.f32, np.dtype(np.int64): qd.i64, np.dtype(np.uint64): qd.u64}

# nw 个线程(线性派发),nw % 32 == 0 -> 32-lane 子组,参考解 lane = w % 32。
NW = 1024
W = np.arange(NW, dtype=np.int64)
LANE = W % 32
GBASE = W - LANE


def dev(a, dtype=None):
    dt = np.dtype(dtype) if dtype is not None else a.dtype  # infer: no silent i32 truncation
    t = qd.ndarray(dtype=NP_D2QD[dt], shape=a.shape)
    t.from_numpy(np.ascontiguousarray(a.astype(dt)))
    return t


def check(name, kfn, o, ins, expected, cmp_mask=None, note="", info_only=False):
    """Launch kfn(o, *ins); compare vs expected (exact) over cmp_mask.
    RAISED = 不可用(错误原文进 handback);FAIL = 可用但数值错;info_only = 只报告不进 gate。"""
    try:
        kfn(o, *ins)
        qd.sync()
        got = o.to_numpy()
    except Exception as e:  # noqa: BLE001 - probe must capture the verbatim failure
        lines_ = [l.strip() for l in str(e).splitlines() if l.strip()]
        msg = f"{type(e).__name__}: {lines_[-1][:220] if lines_ else '(no message)'}"
        print(f"[RAISED] {name} {msg}")
        raised.append((name, msg, repr(e)[:2000]))
        return
    got = got.astype(np.int64)
    exp = expected.astype(np.int64)
    m = np.ones(NW, bool) if cmp_mask is None else cmp_mask
    bad = int((got[m] != exp[m]).sum())
    tag = "INFO" if info_only else "PASS"
    if bad == 0:
        print(f"[{tag}] {name} {note}")
        if info_only:
            return
    else:
        i = int(np.flatnonzero(m & (got != exp))[0])
        msg = f"{bad}/{int(m.sum())} mismatch, e.g. w={i}: got {got[i]} want {exp[i]} {note}"
        if info_only:
            print(f"[INFO-mismatch] {name} {msg}")
        else:
            print(f"[FAIL] {name} {msg}")
            fails.append(name)


# ---------------------------------------------------------------- kernels ----
# 统一形状:顶层 for over NW(线程 = w),每线程写 o[w]。

@qd.kernel
def k_invocation_id(o: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.invocation_id()


@qd.kernel
def k_elect(o: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.elect()


@qd.kernel
def k_broadcast(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.broadcast(v[w], qd.u32(5))


@qd.kernel
def k_broadcast_first(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.broadcast_first(v[w])


@qd.kernel
def k_shuffle_xor1(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.shuffle(v[w], qd.u32(sg.invocation_id() ^ 1))


@qd.kernel
def k_shuffle_xor3(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.shuffle_xor(v[w], 3)


@qd.kernel
def k_shuffle_down3(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.shuffle_down(v[w], qd.u32(3))


@qd.kernel
def k_shuffle_up1(o: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.shuffle_up(v[w], qd.u32(1))


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


@qd.kernel
def k_lanemask_lt(o: qd.types.ndarray(ndim=1)):
    for w in range(o.shape[0]):
        o[w] = sg.lanemask_lt(sg.invocation_id())


# sync / mem_fence:子组作用域共享内存交换。loop_config(block_dim=32) 逐 kernel 标注
# (只对下一个 for 生效;且 1 block = 1 subgroup,否则同 block 双子组竞争 SharedArray)。
@qd.kernel
def k_sync_exchange(o: qd.types.ndarray(ndim=1)):
    sh = sg_block.SharedArray((32,), qd.i32)
    loop_config(block_dim=32)
    for w in range(o.shape[0]):
        lane = sg.invocation_id()
        sh[lane] = lane * 2 + 1
        sg.sync()
        o[w] = sh[(lane + 1) % 32]


@qd.kernel
def k_memfence_exchange(o: qd.types.ndarray(ndim=1)):
    sh = sg_block.SharedArray((32,), qd.i32)
    loop_config(block_dim=32)
    for w in range(o.shape[0]):
        lane = sg.invocation_id()
        sh[lane] = lane * 2 + 1
        sg.mem_fence()
        o[w] = sh[(lane + 1) % 32]


@qd.kernel
def k_sort_kv(ok: qd.types.ndarray(ndim=1), ov: qd.types.ndarray(ndim=1),
              key: qd.types.ndarray(ndim=1), val: qd.types.ndarray(ndim=1)):
    for w in range(ok.shape[0]):
        sk, sv = sg.bitonic_sort_kv(key[w], val[w])
        ok[w] = sk
        ov[w] = sv


# ---------------------------------------------------------------- checks ----
print(f"== PROBE ② simt.subgroup on Metal ({time.strftime('%F %T')}) ==")
print(f"host-side subgroup.group_size() = {sg.group_size()}, log2 = {sg.log2_group_size()} "
      f"(metal_device.mm pins 32; non-32 would invalidate every lane=w%32 reference below)")
if sg.group_size() != 32:
    fails.append("group_size!=32")

o_i = qd.ndarray(dtype=qd.i32, shape=(NW,))
o_u = qd.ndarray(dtype=qd.u32, shape=(NW,))
o_u64 = qd.ndarray(dtype=qd.u64, shape=(NW,))

# 1 identity / vote
check("invocation_id (= w%32)", k_invocation_id, o_i, (), LANE)
check("elect (= lane0)", k_elect, o_i, (), (LANE == 0).astype(np.int32))

# 2 data movement
v_np = ((np.arange(NW) * 7 + 3) % 1000).astype(np.int32)
v_i = dev(v_np)
check("broadcast(v, 5)", k_broadcast, o_i, (v_i,), v_np[GBASE + 5], note="[OpGroupNonUniformBroadcast]")
check("broadcast_first", k_broadcast_first, o_i, (v_i,), v_np[GBASE])
check("shuffle(v, lane^1)", k_shuffle_xor1, o_i, (v_i,), v_np[GBASE + (LANE ^ 1)],
      note="[OpGroupNonUniformShuffle]")
check("shuffle_xor(v, 3)", k_shuffle_xor3, o_i, (v_i,), v_np[GBASE + (LANE ^ 3)])
check("shuffle_down(v, 3) lanes 0-28", k_shuffle_down3, o_i, (v_i,), v_np[GBASE + np.minimum(LANE + 3, 31)],
      cmp_mask=LANE <= 28, note="[top 3 lanes impl-defined, unchecked]")
check("shuffle_up(v, 1) lanes 1-31", k_shuffle_up1, o_i, (v_i,), v_np[GBASE + LANE - 1],
      cmp_mask=LANE >= 1)

# 3 ballot
pred3 = (LANE % 3 == 0).astype(np.int32)
exp_ballot = np.zeros(NW, np.int64)
for g in range(NW // 32):
    b = 0
    for l in range(32):
        if pred3[g * 32 + l]:
            b |= 1 << l
    exp_ballot[g * 32:(g + 1) * 32] = b
dev_pred = dev(pred3)
check("ballot_first_n(pred, 32)", k_ballot_first_n, o_u, (dev_pred,), exp_ballot,
      note="[OpGroupNonUniformBallot comp0]")
check("ballot(pred) u64", k_ballot_u64, o_u64, (dev_pred,), exp_ballot,
      note="[packs comp0+comp1 into u64; Metal i64 = MSL 2.3]")

# 4 votes
pred_at = ((np.arange(NW) % 64) != 7).astype(np.int32)  # 每 64 线程(2 组)第 7 lane false
exp_alltrue = pred_at.reshape(-1, 32).all(axis=1).repeat(32).astype(np.int32)
check("all_true(pred)", k_all_true, o_i, (dev(pred_at),), exp_alltrue)
grp_any = (np.arange(NW) // 32 == 2).astype(np.int32)  # 仅组 2 有 true
check("any_true(pred)", k_any_true, o_i, (dev(grp_any),), (np.arange(NW) // 32 == 2).astype(np.int32))
v_eq = (GBASE + ((np.arange(NW) // 32 == 3) & (LANE == 3))).astype(np.int32)  # 组 3 lane3 破坏一致
check("all_equal(v)", k_all_equal, o_i, (dev(v_eq),), ((np.arange(NW) // 32) != 3).astype(np.int32))

# 5 reductions (v = lane+1 -> 组和 528,所有部分和都是整数,f32/i32 全程精确)
v_r = dev((LANE + 1).astype(np.int32))
exp528 = np.full(NW, 528, np.int64)
check("reduce_add (lane0 valid)", k_reduce_add, o_i, (v_r,), exp528, cmp_mask=LANE == 0,
      note="[lane>0 holds partials, unchecked]")
check("reduce_all_add (all lanes)", k_reduce_all_add, o_i, (v_r,), exp528)
check("reduce_min (lane0)", k_reduce_min, o_i, (v_r,), np.ones(NW), cmp_mask=LANE == 0)
check("reduce_max (lane0)", k_reduce_max, o_i, (v_r,), np.full(NW, 32), cmp_mask=LANE == 0)

# 6 scans
lane1 = LANE + 1
check("inclusive_add", k_inclusive_add, o_i, (v_r,), lane1 * (lane1 + 1) // 2)
check("exclusive_add", k_exclusive_add, o_i, (v_r,), LANE * (LANE + 1) // 2)

# 7 segmented reduce: head 每 8 lane,v = lane+1 -> 每段前缀 (m+1)(m+2)/2, m = lane%8
head8 = (LANE % 8 == 0).astype(np.int32)
seg_m = LANE % 8
seg_base = (LANE // 8) * 8  # 每段的 v 基准: v = 全局 lane+1, 段内从 base+1 连加到 base+1+m
check("segmented_reduce_add(v=lane+1, head@8)", k_segmented, o_i, (v_r, dev(head8)),
      (seg_m + 1) * (seg_base + 1) + seg_m * (seg_m + 1) // 2)

# 8 lanemask (纯算术 wrapper)
check("lanemask_lt(lane)", k_lanemask_lt, o_i, (), (np.int64(1) << LANE) - 1)

# 9 sync / mem_fence + SharedArray
exp_exch = (np.roll(LANE, -1) * 2 + 1).astype(np.int64)  # 邻 lane 写入的值
check("subgroup.sync + SharedArray exchange", k_sync_exchange, o_i, (), exp_exch,
      note="[loop_config(block_dim=32): 1 block = 1 subgroup]")
check("subgroup.mem_fence + SharedArray exchange", k_memfence_exchange, o_i, (), exp_exch,
      note="[fence 不收敛 lane,info-only]", info_only=True)

# 10 bitonic sort kv: 每 32 线程一个乱序 0..31 -> 排序后 lane l 持有 (l, 100+l)
rng = np.random.default_rng(5)
key = np.empty(NW, np.int32)
val = np.empty(NW, np.int32)
for g in range(NW // 32):
    perm = rng.permutation(32).astype(np.int32)
    key[g * 32:(g + 1) * 32] = perm
    val[g * 32:(g + 1) * 32] = 100 + perm
ok_arr = qd.ndarray(dtype=qd.i32, shape=(NW,))
ov_arr = qd.ndarray(dtype=qd.i32, shape=(NW,))
try:
    k_sort_kv(ok_arr, ov_arr, dev(key), dev(val))
    qd.sync()
    gk, gv = ok_arr.to_numpy().astype(np.int64), ov_arr.to_numpy().astype(np.int64)
    exp_k, exp_v = np.tile(np.arange(32), NW // 32), np.tile(100 + np.arange(32), NW // 32)
    bad = int(((gk != exp_k) | (gv != exp_v)).sum())
    if bad == 0:
        print("[PASS] bitonic_sort_kv (i32 kv, 32 lanes)")
    else:
        i = int(np.flatnonzero((gk != exp_k) | (gv != exp_v))[0])
        print(f"[FAIL] bitonic_sort_kv {bad}/{NW} mismatch, e.g. w={i}: got ({gk[i]},{gv[i]}) want ({exp_k[i]},{exp_v[i]})")
        fails.append("bitonic_sort_kv")
except Exception as e:  # noqa: BLE001
    lines_ = [l.strip() for l in str(e).splitlines() if l.strip()]
    msg = f"{type(e).__name__}: {lines_[-1][:220] if lines_ else '(no message)'}"
    print(f"[RAISED] bitonic_sort_kv {msg}")
    raised.append(("bitonic_sort_kv", msg, repr(e)[:2000]))

# ------------------------------------------------- numerical semantics note ----
# subgroup.reduce = 并行(树形/XOR-butterfly)归约 -> 换序求和,数值语义与探针①同桶。
# v 有小数熵: 树形和 vs f32 顺序和大概率末位不同(只报告,不 gate —— 逐位相等不是契约)。
@qd.kernel
def k_num_sem(opar: qd.types.ndarray(ndim=1), v: qd.types.ndarray(ndim=1)):
    for w in range(opar.shape[0]):
        opar[w] = sg.reduce_all_add(v[w])


try:
    vf = dev(LANE * np.float32(0.37) + np.float32(0.11), np.float32)
    opar = qd.ndarray(dtype=qd.f32, shape=(NW,))
    k_num_sem(opar, vf)
    qd.sync()
    par = opar.to_numpy()
    vv = LANE * np.float32(0.37) + np.float32(0.11)
    seq = np.float32(0.0)
    for l in range(32):
        seq = np.float32(seq + vv[l])  # f32 顺序累加
    ref = float(vv[:32].astype(np.float64).sum())  # 一个子组(32 lane)的和;NW=1024 含 32 个重复组
    diff_bits = int(par.view(np.uint32)[0] != np.asarray(seq, np.float32).view(np.uint32))
    e_par = abs(float(par[0]) - ref) / ref
    e_seq = abs(float(seq) - ref) / ref
    print(f"\n[numerics] reduce_all_add(v=lane*0.37+0.11): tree={float(par[0]):.7e} f32-serial={float(seq):.7e} "
          f"f64={ref:.7e} | bits differ={diff_bits}, rel-err tree {e_par:.2e} / serial {e_seq:.2e}")
    print("[numerics] -> subgroup.reduce = parallel tree reduction (reassociation bucket, same "
          "as probe ①): bitwise equality with serial order is NOT part of the contract "
          f"(this input happened to round identically: bits differ={diff_bits}); "
          "accuracy class identical (both ~1e-7 rel vs f64).")
except Exception as e:  # noqa: BLE001
    print(f"[RAISED] numerics-semantics demo {type(e).__name__}: {e}")

# ---------------------------------------------------------------- perf pin ----
# 约束计数形状:每 (w,j) 谓词命中 -> 计数。naive = 每元素一次 atomic_add(L2 原子 round-trip);
# sg = per-lane 局部计数 + subgroup.reduce_add + elect() 单原子。A5 地雷规避:两臂循环内均无
# 条件 RMW(naive 用 select 出 delta 后无条件原子;sg 臂条件原子在循环外直线代码)。
@qd.kernel
def count_naive(d: qd.types.ndarray(ndim=2), counts: qd.types.ndarray(ndim=1), n: qd.i32, niter: qd.i32):
    total_t = d.shape[0] * 32
    for idx in range(total_t):
        w = idx // 32
        lane = sg.invocation_id()
        per = n // 32
        j0 = lane * per
        for _ in range(niter):
            for j in range(per):
                qd.atomic_add(counts[w], qd.select(d[w, j0 + j] > 0.0, 1, 0))


@qd.kernel
def count_subgroup(d: qd.types.ndarray(ndim=2), counts: qd.types.ndarray(ndim=1), n: qd.i32, niter: qd.i32):
    total_t = d.shape[0] * 32
    for idx in range(total_t):
        w = idx // 32
        lane = sg.invocation_id()
        per = n // 32
        j0 = lane * per
        for _ in range(niter):
            c = 0
            for j in range(per):
                c = c + qd.select(d[w, j0 + j] > 0.0, 1, 0)
            total = sg.reduce_add(c)
            if sg.elect() == 1:
                qd.atomic_add(counts[w], total)


print("\n[perf pin] 约束计数(32 lanes/world 协作,两臂仅原子流量不同): "
      "每元素 atomic vs 局部和 + reduce_add + elect 单原子")
PNW, PN, PNITER = 1024, 64, 8
rng = np.random.default_rng(9)
data = rng.standard_normal((PNW, PN)).astype(np.float32)
dd = dev(data)
want = (data > 0).sum(axis=1).astype(np.int64)


def zero_counts():
    c = qd.ndarray(dtype=qd.i32, shape=(PNW,))
    c.from_numpy(np.zeros(PNW, np.int32))
    return c


_ddchk = dd.to_numpy()
print(f"[diag] dd intact before count: {np.array_equal(_ddchk, data)}, "
      f"maxdiff {np.abs(_ddchk - data).max():.3e}")
cnt_n, cnt_s = zero_counts(), zero_counts()
count_naive(dd, cnt_n, PN, 1)
qd.sync()
count_subgroup(dd, cnt_s, PN, 1)
qd.sync()
for nm, arr in (("count_naive", cnt_n), ("count_subgroup", cnt_s)):
    got = arr.to_numpy().astype(np.int64)
    nbad = int((got != want).sum())
    ok = nbad == 0
    extra = ""
    if not ok:
        i = int(np.flatnonzero(got != want)[0])
        extra = f", e.g. w={i}: got {got[i]} want {want[i]} (total diff {int((got-want).sum())})"
    print(f"[{'PASS' if ok else 'FAIL'}] {nm} correctness vs numpy per-world count "
          f"(pred rate {(data > 0).mean():.2f}, {nbad}/{PNW} worlds bad{extra})")
    if not ok:
        fails.append(f"{nm} correctness")

ROUNDS = 7
times = {"atomic-chain": [], "subgroup.reduce": []}
for _ in range(ROUNDS):
    for nm, fn, arr in (("atomic-chain", count_naive, cnt_n), ("subgroup.reduce", count_subgroup, cnt_s)):
        arr.from_numpy(np.zeros(PNW, np.int32))
        qd.sync()
        t0 = time.perf_counter()
        fn(dd, arr, PN, PNITER)
        qd.sync()
        times[nm].append(time.perf_counter() - t0)
t_n = statistics.median(times["atomic-chain"])
t_s = statistics.median(times["subgroup.reduce"])
ops = PNW * PN * PNITER
print(f"    atomic-chain      {t_n*1e3:8.3f} ms  {ops/t_n/1e6:8.1f} Matom-issued/s")
print(f"    subgroup.reduce   {t_s*1e3:8.3f} ms  {ops/t_s/1e6:8.1f} Matom-issued/s")
print(f"    speedup (subgroup over atomic chain): {t_n/t_s:5.2f}x  "
      f"(median of {ROUNDS}; atomics issued: {ops} vs {PNW*PNITER})  [info-only pin]")

# ---------------------------------------------------------------- verdict ----
print("\n=== AVAILABILITY SUMMARY ===")
print(f"RAISED (unusable -> handback): {len(raised)}; FAIL (numeric): {len(fails)}")
if raised:
    print("\n=== HANDBACK SPECS (unusable APIs, C5 block-tier 三层证据链格式) ===")
    for name, first, full in raised:
        print(f"\n--- {name} ---")
        print("  层1(前端/发射): python/quadrants/lang/simt/subgroup.py -> impl.call_internal ->")
        print("     spirv_codegen.cpp 对应 internal-call 分支发射 OpGroupNonUniform*(L1499-1620),")
        print("     capability 门控见 spirv_ir_builder.cpp L95-135(Metal caps 恒开 basic/vote/ballot/arithmetic)")
        print("  层2(Metal 转换): SPIRV-Cross CompilerMSL(metal_device.mm create_compute_pipeline;")
        print("     subgroup caps -> MSL 2.1;int64 -> MSL 2.3)")
        print(f"  层3(最小复现错误原文): {first}")
        print(f"  完整异常: {full}")

print(f"\nSUBGROUP METAL PROBE: "
      f"{'ALL PASS' if not fails and not raised else ('PASS (with handback items)' if not fails else 'FAIL: ' + ', '.join(fails))}")
sys.exit(1 if fails else 0)
