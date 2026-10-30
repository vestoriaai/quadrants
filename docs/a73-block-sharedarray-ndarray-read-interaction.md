# A73: 块级 SharedArray+sync 与 ndarray 读交互——【已结案:不可复现,方法学错误】

> **结案更正(2026-10-30)**:本登记的"ndarray 读返回零"现象**不复存在**。
> 根因是隔离探针的 harness 缺陷:登记时引用的 V3+ 复现脚本**定义了 kernel
> 却从未 launch**(结果检查直接读未写入的零初始化 buffer)——"读零"是
> buffer 从未被写入,不是读路径损坏。带 launch 的对照全部通过。
>
> **对 quadrants 无修复需求,不排入计划。** 以下原文保留供追溯,
> 结论作废。需求清单不设 A73 修复条目(避免幽灵工单)。

## 原始登记(已作废)

## 症状

同一 @qd.kernel 内，当 block-cooperative 段（含 `qd.simt.block.SharedArray`
声明、多次 `qd.simt.block.sync()`、嵌套 while 循环）与后续的
**大数组 ndarray 读**（`ndarray[worldid, col]`，2D f32）共存时，
ndarray 读返回 0（而非实际存储值）。

**触发条件**：块级段含有大量 sync（>5 次）且 SharedArray 尺寸较大
（NVP×NVP f32 ≈ 2-4KB）时，后续 ndarray 读全部返回 0。
简单 SharedArray（4 元素）+ 少量 sync 不触发。

## 期望行为

块级 SharedArray + sync 段之后的 ndarray 读应返回正确值（与无块级段时一致）。

## 复现环境

- quadrants 1.3.4.dev176（56db7c491）
- arch=cpu（LLVM 后端；metal 未测）
- mujoco_qd humanoid 模型（nv=23, nefc≈100, nworld=4096）

## 结案证据(2026-10-30,消费侧复验)

1. **45 参数 echo 探针(带 launch)**:44×ndarray(f32 2D)+ 1 int、
   block_dim=16、迭代循环 + sync,位置 0/20/30/31/32/33/34/40/44
   全部读回正确 → 参数数量/位置无上限效应。
2. **完整 522 行 megakernel(45 参数,块级 + 大 SharedArray + 多 sync +
   嵌套 while/static 循环 + 跨 stage ndarray 读写)真实 differential**:
   constraints.xml / collision.xml(接触)/ humanoid.xml(nv=23)
   × 5-20 步 × **CPU + metal 双 arch**,与 production 分 kernel 路径
   qacc 逐位一致(dQacc=0.00e+00),solver_niter 逐步一致。
3. 原 isolate V0/V1/V2(带 launch)重跑全 PASS——失败只出现在
   (未 launch 的)V3+ 变体。

## 仍有效的 block-tier 工程约束(非缺陷)

- 3 参 range 未过 AST 验证 → strided for + if 门。
- while 条件内 ndarray 读未验证 → 静态界或先读入局部量。
- break 在块级循环内未验证 → flag 门控 + 幂等 stage。

## 登记

- 发现：mujoco-qd Phase C megakernel 开发
- 日期：2026-10-26;结案:2026-10-30(消费侧复验:522 行真 kernel
  differential 双 arch 位精确 + 45 参数位置探针)
