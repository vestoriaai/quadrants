# 开发规格:CPU 并行 kernel 头状态 per-iteration 化(缺陷修复线)

需求来源:mujoqo/docs/qd-defect-cpu-并行计算偏移写竞态.md(根因实锤 + 修复方向 1'
+ 后端影响,commit 9647105/0b368bf)。分支:mujocoqd/cpu-head-state-per-iteration
(worktree /Users/mini0/Code/quadrants-cpuwt)。

## 问题(实锤)

qd CPU 后端,`for __t` 并行循环**之前**声明的标量被提升为 global temp(每 launch
一份共享槽):并行任务对其无同步 read-modify-write → 竞态(静默数值错);即便
serialize 也只是"launch 共享累积"(非 warp 语义)。探针
mujoqo/_probe/cpu_head_state_leak_probe.py:头声明 acc 输出 1..16 单调,体内声明
全 1(warp 正确)。

## 修复语义

warp per-program 语义:**头声明 = 每程序实例执行**。并行任务内被 STORE 的跨任务
提升量不得留在 global temp——alloca+init 移入并行任务侧(每迭代重建);只读消费
的跨任务传值保持提升不变(IR 逐位不变)。

## 修复落点

`quadrants/transforms/offload.cpp`:
- `PromoteIntermediateToGlobalTmp` / `PromoteLocalToGlobal`(`local_to_global_` 分配)
  的判据:消费方 offload 为**并行 struct-for** 且其中含对该值的 **Store**(直接
  store 或 atomic 前的 load-modify)→ 不提升,改为在该并行任务内重建
  (alloca + 串行任务的 init 语句克隆进任务入口/TLS)。
- 注意跨任务链:值可能经多个任务传播后才进并行任务;判据按"任一下游并行任务含
  Store"判定。含糊情形(无法静态确定)一律保守 = 按含 Store 处理(正确性优先)。
- care point:**AD 相关 pass** 也消费提升量(tests/python 的 AD 用例必须过)。

## 实施步骤

1. **先读** `.claude/skills/qd-compiler/SKILL.md`(compiler 源码地图/A 系列地雷/构建
   流程)与 `docs/qd_compiler_dev_notes.md` 相关节。
2. 复现确认:把探针精简移植为 worktree 内 `_probe/head_state_probe.py`,修复前跑出
   单调 1..16。
3. 实现 offload.cpp 判据 + 重建;`tests/cpp/transforms/` 若有 offload 相关单测需同步。
4. 构建:`MAKEFLAGS=-j8 NO_PROXY="*" .venv-build/bin/python build.py wheel`
   (worktree 自建 build 目录;python-only 无关,本次动 native → 必须构建)。
5. 自建验证 venv(命名 `_probe.venvhstate`,python3.12),装自己的 wheel。
6. 验证:探针全部形态过(head 多线程全 1 / serialize 也全 1 / body 不变); multiflex
   flexedge 复现(参照 mujoqo docs 记录)确定;AD 用例过。
7. 门禁:`zsh tests/robostudio/run_robostudio_gates.sh`(全绿,含既有 gate 1-17;
   新 gate 注册见下)+ `pytest tests/python/test_return.py` 46/46 + AD 用例。
8. 新 gate:`tests/robostudio/gate_cpu_head_state.py`(gate 19):探针三形态断言
   (head 全 1 / body 全 1 / serialize 全 1——**修复后 serialize 语义变化是有意
   的**,gate 注释注明)+ 无回归。注册进 runner。

## 纪律(硬性)

- 白名单:`quadrants/transforms/offload.cpp`(如需触及其它 transforms 文件,
  OWNER NOTE 单独 commit 说明理由)、`tests/robostudio/gate_cpu_head_state.py`(新)、
  `tests/robostudio/run_robostudio_gates.sh`(追加 gate 19)、
  `docs/cpu_head_state_development_spec.md`(状态段)、`docs/robostudio_features.md`
  (ledger)、`_probe/head_state_probe.py`(新)。其余只报不改。
- 小步 commit(诊断复现/实现/gate/ledger 分开),英文 subject,尾缀
  `Co-Authored-By: Claude Code <noreply@anthropic.com>`。
- **磁盘守护**:开工前与构建前后 `df -h /`,<5Gi 停手报告(本轮动 native,构建
  目录 + wheel 需要空间;当前 11Gi 可用)。
- **不碰** /Users/mini0/Code/quadrants 主树、/Users/mini0/Code/quadrants-bvhwt
  (另一条 BVH 线在跑)、/Users/mini0/Code/mujoqo。native 构建在**本 worktree 内**
  自洽(build 目录隔离)。
- 不 merge、不 push(协调人合并)。

## 完成定义(DoD)

1. 探针三形态全过(修复后语义);multiflex 复现确定
2. gate_cpu_head_state 注册并全过;union gates 全绿;test_return.py 46/46;AD 用例过
3. ledger 条目四字段(设计/数字/footprint/限制)
4. 报告:commit 列表、gate 摘要、语义变更影响清单(哪些既有 kernel 行为变化)

## 明确不做

- mujoqo 侧 10+ 串行化 workaround 的移除与回归(待 wheel 重烤后的集成线)
- Metal 端到端验证(metal 电池冻结中;门禁按 qd CLAUDE.md 口径 CPU 基线放行,
  metal 结果单列记录)

## 状态(2026-10-18,landed on branch mujocoqd/cpu-head-state-per-iteration,待协调人 squash)

**实现**(commits e1d96a73 探针 / 774c90e5 实现 / 71f717f6 gate / 90124ad2 判据收窄 /
后续 serialize 排除 commit):offload.cpp 新增 pre-Offloader 变换
`rebuild_head_state_per_iteration`,判据 = **写∧读 ∧ 非序列化 kernel 顶层循环**
(range_for/struct_for/mesh_for):克隆 alloca + 循环前无环前缀 store 进循环体入口,
循环子树内全部引用重接;alias(pointer alloca)簇并查 union 后整簇重建;
`is_shared`(qd.simt.block.SharedArray,block 级共享窗)排除;写-only 形态
(test_ret_4k 类,`ret=vec(0); for i: ret[i]=i; return ret`)不重建,提升路径逐位不变。

**语义二元性(ADJUDICATED R3,2026-10-18;本节为权威文本,合并时请协调人将此段
搬入 docs/qd_compiler_dev_notes.md 相关节)**:qd CPU kernel 顶层循环两种声明语义——
**非 serialize = per-program**(头状态写∧读 → 每迭代重建,warp 对齐,本缺陷线的修复面);
**serialize=True = 显式顺序任务**(头状态跨迭代携带,顺序程序语义)。后者是语言级
声明契约,消费方 = quadrants 自己的 algorithms CPU 路径(_cpu_paths.py docstring:
"one thread, program order, so the loop-carried accumulator is exact";计数排序直方图
`scratch[d]+=1` 与 256-digit 前缀和 `run=run+b` 每循环显式 serialize 标注,携带为
算法本体)。双轮实证:rebuild 含 serialize 形态时 select/sort_radix/t10 sort 全挂
(分支基线轮全绿),IR 层两种形态同形、静态不可区分,唯一可用判据即 serialize 标志。
集成线(mujoco 侧)摘 serialize workaround 后,对应 kernel 自动获得 per-program
修复(非 serialize 化后落本判据面)。

**判据边界**:写∧读才 rebuild;写-only(循环内不读、循环后消费)不 rebuild——前提是
"无循环后读"形态。未来若出现"写-only + 循环内读"新形态,本判据会漏(rebuild 与否
取决于读是否落循环内),应响亮拒绝或补 lint 规则,不许静默漏网(已记 ledger 限制)。
另:跨迭代携带 + 非 serialize 的头变量循环(如未标注 serialize 的顺序扫描)在
>512×threads 迭代时 pre-fix 即竞态(512-chunk 阈值下的偶然串行不再保护),本修复后
确定性 per-iteration——两类都不是合法携带写法,顺序算法必须显式 serialize。

**验证数字**:探针三形态(每形态 8 次 launch):head 全 1 / body 全 1 / serialize
确定性 1..16(= R3 有意语义,pre-fix 同值);gate 19 六形态 × 20 launches 全过
(head 全 1 / serialize 1..64 / body 全 1 / read-only 43 / computed-offset
确定性 / write-only ret[i]==i);test_return.py **46/46**(含 test_ret_4k);
AD 电池双轮对照:**436 个 test_adstack 失败为分支既有**(pre/post 失败清单逐条相同,
零新增),其余 11 个 AD 文件 214/214 双轮全绿;union runner 19 门全量绿
(gate 18 槽位留 BVH 线合并时落)。

**行为变化面清单(合并审查用)**:(1) 非 serialize 顶层循环内写∧读的头声明标量/
向量 → 每迭代重建(缺陷类修复;唯一语义变化面);(2) serialize 循环行为不变;
(3) read-only / write-only 头变量行为不变;(4) is_shared 共享窗行为不变;
(5) known-good 偶然形态:头 RMW + 非 serialize + ≤512×threads 迭代的 CPU 循环,
pre-fix 靠 512-chunk 阈值偶然单线程(如 algorithms 曾依赖),现若未标 serialize
即转 per-iteration——顺序算法必须显式 serialize(上文判据边界)。

**遗留**:(1) 离线缓存陈旧坑(本轮实证):同版本戳不同二进制的 wheel 会命中旧
offline cache(kernel_compilation_manager key 未含二进制内容),gate/电池前需清
~/.cache/quadrants/qdcache;归缓存键线,非本线修复;(2) write-only+循环内读未来
形态的响亮拒绝/lint 补位;(3) mujoqo 侧 10+ 序列化 workaround 移除回归(集成线);
(4) metal 端到端(metal 冻结中,门禁 metal 臂单列)。
