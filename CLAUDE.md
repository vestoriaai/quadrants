# quadrants 开发工作准则(robostudio 线)

Genesis AI Taichi fork 的 robostudio 部署分支。**`robostudio-main` 为 single source of truth**:branch-per-line 开发 → gate 全绿 → **squash 合并**(`git checkout robostudio-main && git merge --squash <branch>` 后以单 commit 落库,message 概括该线全部内容;`docs/robostudio_features.md` INDEX 行指向该 squash commit);协调人合并,线上 agent 不 merge/push。

写/改 compiler 前先加载项目 skill `qd-compiler`(`.claude/skills/qd-compiler/SKILL.md`:源码地图、A 系列地雷、修复模式、构建/gate 流程)。

## 线纪律(每条 branch 一个"线")

1. **开工**:从 robostudio-main HEAD 切 `mujocoqd/<line-name>`;任务书写明白范围/验收/footprint 预算。
2. **最小可维护**:footprint(+LOC、触及核心文件)与收益同为一等指标,ledger 条目必写;休眠机制(默认 OFF 且无实测收益)原则上不合入;禁顺手重构。
3. **门禁**:`tests/robostudio/run_robostudio_gates.sh <venv>` 全绿(门数以 runner 头注为准)+ 上游 `pytest tests/python/test_return.py`(46/46)+ 本线新 gate 进 runner。**审计与 lint 双网(2026-10-19 起,audit-lint 线)**:①`QD_IR_AUDIT=1` 下重跑本线触达的 gate/探针(IR 所有权审计,触兜底=响亮命名错误;超大 kernel/A24 家族勿开——隔离投毒放大内存);②`ruff check python/quadrants/` 归零 + 改动 .cpp 过 `clang-tidy -p <build>`(compile_commands.json 已入 CMake,二进制 brew llvm@22 / uv tool ruff)。
4. **ledger**:每条线在 `docs/robostudio_features.md` 加条目(设计/数字/footprint/限制);编译器 bug 进 mujoqo 的 `docs/quadrants-需求清单.md` A 系列(带最小复现,复现件进 `_probe/` 或 `cs_evidence/`,**/tmp 视为不存在**)。**每线理论上只有一个 commit(squash 本身),没有其他**——ledger 条目随 squash 同 commit 落库;INDEX 行的 sha 回填**不再单独开 docs commit**(自引用问题:commit tree 无法包含自身 sha),顺带在下一线合并时补或省略(2026-10-19 用户裁定,废除旧的"INDEX 回填 follow-up commit"惯例)。
5. **提交**:英文 subject,结尾 `Co-Authored-By: Claude Code <noreply@anthropic.com>`;小步提交。

## 多 agent 并行(2026-10-12 事故后成文)

- 主工作树同时只归一条 native 线(构建树互斥);python-only 线用 `git worktree` + overlay。
- 一切脚本**禁硬编码绝对路径**(用 `__file__`/worktree 相对);跑前 `pwd`、跑后 `git status` 核对。
- 先 merge 最新基线再动手;非白名单文件的"问题"= 同步信号,只报不改。
- 零 commit 的长段违规;窗口见底先落盘 memory/ledger。
- venv 归属:每线自建验证 venv(`_probe.venvN`),建了的基线 venv 不许他人 force-reinstall。

## 构建与归因

- wheel:root 下 `MAKEFLAGS=-j8 NO_PROXY="*" .venv-build/bin/python build.py wheel`(sccache 增量;python-only 改动分钟级);装**自己的**验证 venv。
- **双 wheel 归因法**:区分"既有核心问题"vs"本轮回归"——老基线 wheel(venv3=dev36)与当前 wheel 对跑最小探针,行为一致 = 既有,不一致 = 回归。探针进 `_probe/a_evidence/` 并断言生效。
- Metal 数值 flaky 先对号 A5/A10-A12(行序律/`+=` 形式/kernel 边界可见性),metal 门禁单列,CPU 基线放行。
- GPU 共享:perf 前 `bash tests/robostudio/_gpu_busy.sh`;机器方差 ±4×,结论用交错中位。
