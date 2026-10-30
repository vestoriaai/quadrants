# A73: 块级 SharedArray+sync 与 ndarray 读交互缺陷——ndarray 读返回零

## 症状

同一 @qd.kernel 内，当 block-cooperative 段（含 `qd.simt.block.SharedArray`
声明、多次 `qd.simt.block.sync()`、嵌套 while 循环）与后续的
**大数组 ndarray 读**（`ndarray[worldid, col]`，2D f32）共存时，
ndarray 读返回 0（而非实际存储值）。

**触发条件**：块级段含有大量 sync（>5 次）且 SharedArray 尺寸较大
（NVP×NVP f32 ≈ 2-4KB）时，后续 ndarray 读全部返回 0。
简单 SharedArray（4 元素）+ 少量 sync 不触发。

## 最小复现

```python
# V0: 纯 serial per-world（无 block/SharedArray/sync）→ 正常 ✓
# V1: + SharedArray（无 sync）→ 正常 ✓
# V2: + block_dim + sync → 正常 ✓
# V3: + cholesky-mega 体（~200 行，NVP×NVP Hs + 多段 sync + 嵌套循环）
#     → 后续 ndarray 读返回 0 ✗
```

隔离脚本：mujoqo-wt-metal /tmp/isolate_jaref.py（V0/V1/V2 三组）+
bisect_mega.py（V1/V2 变体）。

完整复现：mujoqo-wt-metal solver_mega.py 迭代循环，cholesky stage 后
linesearch stage 读 `ctx_Jaref_in[worldid, row]` 返回 0（expect 0.5）。

## 影响

- mujoco_qd Newton solver 的 megakernel（Phase C）在 cholesky stage
  后的 ndarray 读全部返回 0 → solver 结果错误。
- 影响所有在同一 kernel 内混合 block-coop 计算段与大数组 ndarray 读的场景。

## 期望行为

块级 SharedArray + sync 段之后的 ndarray 读应返回正确值（与无块级段时一致）。

## 复现环境

- quadrants 1.3.4.dev176（56db7c491）
- arch=cpu（LLVM 后端；metal 未测）
- mujoco_qd humanoid 模型（nv=23, nefc≈100, nworld=4096）

## 建议修复方向

1. AST 层：检查块级 sync/SharedArray 对后续 ndarray 读的 IR 影响
   （可能的 alias/生命周期错位）
2. 运行时：检查 block-tier sync 后 ndarray base 指针是否被错误失效
3. workaround：将 ndarray 读拆分为独立 kernel launch（当前生产路径
   已如此——每 stage 独立 kernel，无此问题）

## 登记

- 发现：mujoco-qd Phase C megakernel 开发
- 日期：2026-10-26
- 隔离实验：/tmp/isolate_jaref.py（V0/V1/V2 三组全 PASS 确认非基础问题）
