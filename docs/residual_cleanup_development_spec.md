# residual-cleanup 开发任务书（qd 线）

> 2026-10-25。分支 `mujocoqd/residual-cleanup`（worktree quadrants-rw2，基点
> robostudio-main@07889130，dev150 轮已装 .venv-consolidated/_port.venv）。
> 承接：battery 语义账四族已清（A53/A46/pyi_stubs/A58），主链 A59/A60/C1a 硬化
> 连带清 src_ll_cache/math/quant；本线收尾**全部残留**。
> 必读 skill：quadrants 侧 `.claude/skills/qd-compiler/SKILL.md`。

## 残留清单（dev150 battery 实测 5F）

| # | 项 | 归因 | 修复路径 |
|---|---|---|---|
| 1 | **A62**:test_atomic ×4（global i32/f32 双 arch） | 局部标量 `qd.atomic_add(ck, step)`（kernel 局部变量）返回旧值错误——y[1..8] 全 0（应 42..336），全局 atomic 正确；cpu+metal 确定性（最小复现 `_probe/a60_local_atomic_repro.py`） | 前端/codegen：局部标量 atomic 的 lowering 串行化。先归因成文（对照 taichi 上游局部 atomic lowering：寄存器级 RMW 串行），本质 review 后实施 |
| 2 | test_quant_fixed ×1 | **顺序敏感 flake**（单跑即绿；全量 battery 特定 worker 顺序下挂）= 测试基建已知"全量套件内核物化撞键"族 | 测试侧：调查撞键 kernel 名（唯一名规避，QD 测试规范既有先例）；若无法定位则 skip-if-battery 标记 + flake 清单登记，不修内核 |

## 白名单

- A62：`python/quadrants/lang/ast/**`（前端 lowering/atomic 展开）、
  `quadrants/codegen/llvm/**`（如归因指向）、`_probe/` 新探针
- quant_fixed：`tests/python/test_quant_fixed.py`（测试侧）
- `docs/robostudio_features.md`（ledger，合并 commit 内）

## 纪律与门禁

- A62 修复前先归因文档（最小复现 → 定位到 pass → 方案对比）成文，本质 review 后动 codegen；
- union gates 21/21 + test_return 46/46 + battery 全量（目标 0F 或仅剩已登记 flake）；
- 小步提交；atomic lowering 改动后必须全量 battery（前端改动全库生效）。

## DoD

1. battery 全量 **0F**（flake 以"单跑绿 + 已登记"口径计）；
2. gates 21/21；ledger 四字段条目；
3. A62 反证探针：复现脚本转绿且全局 atomic 路径无回归。
