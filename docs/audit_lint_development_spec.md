# audit-lint 线任务书(mujocoqd/audit-lint)

日期:2026-10-19 · 基线:robostudio-main @ 4da921ff · worktree:~/Code/quadrants-auditwt

## 背景

A37 家族第二例 UAF(scalarize 语句所有权,见 robostudio_features.md A37 条目扩展)暴露
两个防御缺口:①IRVerifier(operand 存活断言 + parent 回指针)已存在且每个 transform
后挂了 `verify_if_debug`,但被 `config.debug` 门控——Release 轮子/门禁全程 no-op;
②offload_to_executable 内部分 scalarize 挂点后直落后续 pass,窗口内无 verify。
lint 侧:2026-10-18 盘点定两条"可选最小防护"(CMAKE_EXPORT_COMPILE_COMMANDS +
ruff 本地化)未动手。

## 范围

1. **审计启用层**(不加新 pass,复用 IRVerifier):
   - `verify_if_debug` 增 `QD_IR_AUDIT=1` env 强制门(Release 轮子/门禁可开审计)。
   - compile_to_offloads.cpp 三处 scalarize 挂点后各补一条 `verify_if_debug`
     (:360 kernel 路径、:437 force_scalarize_matrix、:460 func 路径)。
2. **lint 初步支持**:
   - 根 CMakeLists `set(CMAKE_EXPORT_COMPILE_COMMANDS ON)`(零行为影响,产出
     compile_commands.json 供 clang-tidy/IDE)。
   - pre-commit ruff 钩子已在配置(astral-sh/ruff-pre-commit);本机装 ruff 实跑
     python/quadrants 证明网可用;clang-tidy 对本线改动文件实跑一次。

## 验收

- 健康 IR 零误报:`QD_IR_AUDIT=1` 下 int2bool 探针 + test_return 46/46 + union
  runner 21/21(默认 env 照跑)+ union 全程 audit-on 抽验。
- **抓真 bug 实证**:worktree 内临时回退 scalarize 所有权 insert(复刻 familyr 陈旧
  形态),`QD_IR_AUDIT=1` 编 int2bool → 预期响亮 `IR broken ... cannot have operand`
  (QD_ASSERT)而非静默腐坏/段错误;复测后还原。
- clang-tidy/ruff 各实跑一次留痕;compile_commands.json 出现在 build 树。

## footprint 预算

verify.cpp +3(一个 static env 读 + 条件);compile_to_offloads.cpp +3(三条
verify 调用);CMakeLists +1;spec + ledger 文档。零语义变化(默认路径逐位不变:
env 不设时 verify_if_debug 照旧 no-op)。
