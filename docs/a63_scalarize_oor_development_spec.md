# A63 — scalarize 越界 const MatrixPtr 降级（robostudio fv 内核 cpu 编译失败修复）

## 问题（用户侧实弹，2026-10-25）

robostudio `batch_kin` 套件在 dev152 (26d8e62c2) 上抓到回归：
`test_fv_parity[cpu]` 编译 `fv_c84_0`（>50k 语句的巨型内核）时
`ScalarizePointers` 抛出

```
scalarize: MatrixPtr offset 63 out of range for local tensor of 63 elements
MatrixPtr stmt: <*f32> $14294 = shift ptr [$50 + $26]
origin alloca:  <*[Tensor (7, 9) f32]> $50 = alloca
```

（robostudio 侧复现：`packages/batch_kin/batch_kin/tests/test_fv_parity.py
-k cpu`，fv = RNEA 融合内核，dev119 (41175f2a2) 同测试全绿。）

## 根因

`GatherScalarizableLocalPointers` 只把**非 const offset** 的 MatrixPtr
降级；**const offset 不查边界**。A24/A54 引入
`cfg_auto_skip_threshold`（=50000）后，巨型内核走轻量 simplify 路径、
不再跑全量 cfg_optimization——过去由 full cfg 在 scalarize 之前折掉的
**投机/死代码越界 MatrixPtr IR**（此处 = Tensor(7,9) alloca 上的
const offset 63）得以存活到 scalarize，`ScalarizePointers` 逐元素改写
时按 A21 的可捕获异常抛出，编译中止。

证据链：①错误 stmt/alloca 对与 fv 源
（`rnea_kernels_fv.py:225 Rk[a,b] = RPl[k+1, a*3+b]`）一致；②同测试
dev119 全绿（无 skip 特性 → 全量 cfg 恒跑）；③失败内核日志带
`cfg_optimization skipped ... cfg_auto_skip_threshold=50000`；④A54
文档明示 "The skip changes codegen"。注：尝试运行时覆盖阈值做对照
（qd.cfg 与 prog.config 两条路径）均未生效——kernel 编译读的是
init 时快照，对照实验不可达；机制判定基于①-④，置信=高。

## 修法（最小接触面）

`GatherScalarizableLocalPointers::visit(MatrixPtrStmt*)` 在 const
offset 分支加边界检查：`offset ∉ [0, num_elements)` → 把该 alloca
降级（`is_alloca_scalarizable_ = false`）。效果：

- 降级的 alloca 保留张量存储，ScalarizePointers 不再对它逐元素改写
  → 不抛；越界 MatrixPtr 留给常规清理（死代码）或如实暴露（真执行
  的越界 = 内核级 bug，scalarize 不掩盖）。
- 语义零变化：scalarize 本就是可选优化；降级 = 放弃该 alloca 的
  标量化（性能不变差于 dev119 行为——dev119 下该 IR 已被 cfg 折掉，
  alloca 同样未被标量化……确切说部分可标量化机会损失，仅限含越界
  IR 的内核，可忽略）。

## 改动

- `quadrants/transforms/scalarize.cpp`：gather visitor const 分支 +
  边界降级（+15 行，含 A63 注释）。
- `tests/cpp/transforms/scalarize_test.cpp`：
  `TEST(Scalarize, OutOfRangeConstMatrixPtrDemotesAlloca)` ——
  Tensor(2,3) alloca + const offset 6 MatrixPtr + 保活 store；断言
  scalarize 不抛、原 alloca 与越界 ptr 原样保留、无逐元素改写。

## 门

| 门 | 内容 |
|---|---|
| cpp UT | `TEST(Scalarize, OutOfRangeConstMatrixPtrDemotesAlloca)` 已入库；本地 wheel 构建配置不含 tests（`QD_BUILD_TESTS=OFF`，开建需拉 googletest），**随 CI/测试配置执行**（登记为待跑项） |
| robostudio 实弹（主验证，已过） | `test_fv_parity` dev153 全绿（cpu 166s / metal 变体过 / 1 skip = 在录跨 arch 守卫）——正是生产失败场景端到端 |
| robostudio 根套件 | 565+/0F 基线不回退 |
| battery | qd 全量 battery（dev153 轮）无新失败 |
