# A63 — GDAR 检查器 ndarray 崩溃修复 + 静态扫描 pass — 开发任务书

日期:2026-10-24 | 线:`mujocoqd/a63-gdar-ndarray-scan` | 基点:robostudio-main@26d8e62c
来源缺陷登记:mujoqo `docs/quadrants-需求清单.md` A63(含判别实验与最小复现)

## 判决

`GloablDataAccessRuleChecker`(validation.cpp)只处理 field 指针形态
(`GlobalPtrStmt`/`MatrixPtrStmt`);ndarray 访问指针是 `ExternalPtrStmt`
(base_ptr=ArgLoadStmt(arg_id), 带 indices/is_grad),load/store/atomic 三路全部
撞 `QD_ASSERT` 中止编译。好 kernel 也崩(非违规检测)。mujoco_qd 100% ndarray
参数 ⇒ 检查器对该生态不可用。

## 方案(两部分,均无 ABI/runtime 变更)

1. **优雅跳过**:`GloablDataAccessRuleChecker` 三处 visit(load/store/atomic)遇
   `ExternalPtrStmt` 直接 return——field checkbit 通路不变,ndarray 不再崩。
2. **静态扫描 pass `GdarStaticScanChecker`**:VALIDATION 模式编译期,线性序收集
   每 ndarray arg(arg_id 键)的 load/store/atomic 事件(ExternalPtrStmt 的
   indices 做 SSA `Stmt*` 相等配对,手法同 `OffloadLevelGlobalCrossIterRAWChecker`),
   三档分类:
   - **definite**:同 index 向量(逐元素 `Stmt*` 相等)的"读→写"程序序对
     (含 atomic 写;atomic 读改写语义与 checkbit 断言一致);
   - **potential**:同 arg 既有读又有写但无可证的确定对(索引不同 Stmt,
     可能别名);
   - **safe**:仅读/仅写/纯 atomic。
   输出 `QD_WARN`("[gdar-static-scan]" 前缀 + kernel 名 + arg_id + 计数)。
   定位=工程统计与代码审查信号,非运行时检查(保守上界,误报由人工/数值测试裁决)。

## 验证方案(含反证判据)

- 新测试 `tests/python/test_ad_gdar_static_scan.py`(fd 级 stderr 捕获):
  safe kernel 无 definite/potential 行;`y[i]=x[i]*y[i]`(读后写同索引)出
  definite;异索引读写出 potential;混合 field+ndarray kernel 不崩(A63 原复现
  反证:修复前必抛 validation.cpp:177)。
- 既有回归:`test_ad_global_data_access_rule_checker.py`(field checkbit 路径
  必须原样)、`test_ad_basics.py`、`test_ad_for.py`、`test_ad_ndarray.py`。
- gates:`run_robostudio_gates.sh`(21 门,重烤 wheel 后跑)。
- 反证判据:若 safe kernel 出现 definite 警告 = 扫描器误报(实现错);
  若 field 违规 kernel 不再抛 = checkbit 路径被破坏(回归)。

## Footprint 预算

- `quadrants/transforms/auto_diff/validation.cpp`(+~130 行)
- `quadrants/transforms/auto_diff/validation.h`(+1 声明)
- `quadrants/transforms/compile_to_offloads.cpp`(+2 行:VALIDATION 分支调用)
- `tests/python/test_ad_gdar_static_scan.py`(新建)

白名单:以上四文件。其余文件不动;发现越界问题只报告。

## DoD

新测试 4/4 绿;既有 AD 测试全绿;gates 21/21;wheel 重烤并在 mujoqo 侧用
`_probe/autodiff_research/probe_gdar_disc.py` 复验(好 kernel 静默/坏 kernel
definite 警告);ledger 条目(设计/数字/footprint/限制)。
