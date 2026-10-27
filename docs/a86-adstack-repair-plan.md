# A86: adstack/自动反向修复计划(追踪四十八)

日期: 2026-10-27 | 状态: 计划书 v1(Phase 0 未执行, 假设未验证)
关联: mujoqo-iftwt `docs/a86-bptt-closure-report.md` §6/D1、账本追踪三十四/四十二/四十六

## 0. 现状与症状(已知事实)

| # | 事实 | 证据 |
|---|---|---|
| K1 | builder 内核(while+嵌套 if+标量跨迭代携带)的自动反向产出 inf/垃圾 | mujoqo S1 链 M_scratch.grad=inf; c623 reverse IR: 除数来自 unclamped `stack load top` |
| K2 | 栈配对不对称实录: `$7880014` load-top-无-pop(peek), 对照 `$7880011` 配对消费 | 追踪三十四/三十五 IR 审计; **是否设计(broadcast 读)从未定性** |
| K3 | step_ift 单步边界下, 每 stage 的伴随应用精确(1.5-2.3e-07) | 追踪四十六 a86_steprec; 即: 边界外的自动反向才是坏的 |
| K4 | 受影响内核族: cholesky factorize/solve、RNE、transmission —— 全部含 while/分支依赖控制流 | mujoqo `_src/smooth.py` builder 内核 |
| K5 | 现有 IFT/FD 边界已绕开全部受影响路径(三 sample 全 PASS) | mujoqo cdd2d30; 即本缺陷当前为**潜伏**, 不阻塞 sample |

## 1. 根因假设(按优先级; Phase 0 前均未验证)

**H1(强假设, 新识别): `eliminate_recomputable_ad_stack_pushes` 在纯重放+原位覆写环境下不健全。**
- 机制: 该 pass(auto_diff.cpp:62, eliminate_recomputable_pushes.cpp)删除"可重算"的
  push —— 判据为 chain leaves 落在 kernel args / 已栈化 allocas / **未在 IB 内变异的
  SNode**。但判据没有覆盖: **被同一段重放里*其他 kernel* 原位覆写的普通数组**。
  forward 时 push 的值 v 由数组状态 S_fwd 计算; 反向重算时数组已是 S_later
  (后续 kernel 原位覆写过) → 重算值 ≠ 前向值 → 栈内容错位 → 下游
  load-top(含合法 peek)读到垃圾 → inf。
- 与 mujoqo 现场吻合: euler 的 M/qLD scratch 全是原位覆写; 受影响内核全在
  原位覆写密集区。
- 一日可验证: 加 config knob 关掉该 pass → 重跑 mujoqo S1 无边界链 →
  inf/垃圾消失与否即判。

**H2: make_adjoint 反向走查的栈配对缺陷(A65/A68 协议实现)**。
- 机制: 反向 kernel 的伴随走查按 forward 语句逆序重放控制流, guard 值经
  guard-stack 传递; 分支依赖的 push 数要求反向在**同一路径**上镜像
  push/pop —— 若 walk 在 peek(不消费)与 pop 之间混入分支重入, LIFO 对齐破坏。
- 已知实例: A68 while 协议(forward 每执行迭代 push 一个 marker, 反向 LIFO
  消费)与 guard-stack 混用时的对齐未系统验证。

**H3: 静态不可判定构型**: 分支依赖的 push 数使任何静态配对分析在该构型下
  都不成立 —— 需要协议级 fallback(checkpoint/值快照, 见 §2 方案 C)。

优先级排序理由: H1 可一日证伪/证实且解释面广(覆盖 K1 的 inf 与部分假梯度);
H2 次之(A65/A68 是我们自己的新代码, 审计线索直接); H3 是 H1/H2 都排除后的
结论性诊断。

## 2. 修复方案选项

**方案 A(H1 成立时): 消除 pass 的健全性门控 / kill-switch。**
- 形态: config 项(如 `ad_stack_elimination=off/on`)或判据扩展 —— chain leaves
  增加"跨 kernel 可能被覆写"类(保守化: 只要 leaf 数组在 IB 外有任何写入点
  即视为不可重算)。
- 代价: 栈内存与运行时 push 增大(determine_ad_stack_size 已有上限机制);
  触发 stack overflow 的 kernel 需要上限调整或降级。
- 杠杆: 可能以 <1 天修复恢复全部 builder 内核自动反向的正确性。

**方案 B(H2 成立时): make_adjoint 反向走查的配对修复。**
- 形态: 修正 walk 中 peek/pop 的发射顺序; 或统一协议为"每个 load-top 都
  有配对 pop(消费型)"并禁止无主 peek。

**方案 C(H3 成立时): 不健全内核的 checkpoint fallback。**
- 形态: 静态分析判定配对不可验证的 kernel, 放弃 adstack, 改为 per-iteration
  值快照(ring buffer / forward_state_spill 机制扩展) —— 正确但内存与运行时
  成本高, 仅对该 kernel 降级。

**方案 D(长期, 与 A/B/C 正交): 栈协议规格化文档 + 编译期平衡校验 pass**
- 形式化不变量: 对每个 AdStackAllocaStmt 栈 s, 沿任意执行迹,
  push(s) 与 pop+消费型 load-top(s) 逐事件 LIFO 配对; 校验 pass 在编译期
  assert(可配置为 warning), 违规即时报而非静默产错值。

## 3. 阶段计划(每阶段带 gate 与退出条件)

**P0 — H1 验证(预计 0.5-1 天)**
- 加 config knob 关 eliminate_recomputable_ad_stack_pushes。
- 实验: mujoqo S1 无边界链(p5_s1_onestep_fd 去掉 IFT 边界或直接跑
  factor_solve_i 的 auto 反向)双跑对比。
- Gate: inf/垃圾消失 → H1 成立, 走方案 A; 不消失 → H1 排除, 进 P1。

**P1 — 栈平衡审计工具(预计 1 个工作日)**
- IR pass: 对每 kernel 按 AdStackAllocaStmt 逐一统计 push/pop/load-top
  事件序列, 输出配对报告(静态可查部分); 同时支持运行期插桩计数
  (已有 [gb]/stack 审计手法, 追踪三十四/三十五)。
- Gate: 在最小复现与 c623 上重放追踪三十四/三十五的发现并分类
  ($7880014 peek 定性: 设计 or bug)。

**P2 — 定位(预计 1-2 个工作日, 不确定性最高)**
- 在受控内核族(§4)上逐构型定位违反点; 每个发现附最小复现 + IR 前后对比。
- Gate: 每个候选缺陷有"最小 kernel + 错值复现 + IR 证据"三件套。

**P3 — 修复(预计 1-3 个工作日, 视 P2 结论落在方案 A/B/C 哪档)**

**P4 — 测试(预计 1 个工作日)**: 见 §4。

**P5 — mujoqo 接入与收益兑现(预计 1-2 个工作日)**
- 撤 FD 边界 → S3 纯自动链回归(精度门: 保持 ≤1.47e-04 量级);
- P4 反向画像重做(解析链的反向成本应 ≈ 正向 2-3 倍);
- 清理: ad_ift 仅保留求解类解析 IFT(长期保留, 属 D0 立项本意)。

## 4. 测试计划

1. **内核族单测**(新, qd 侧): 控制流构型矩阵 —— for+while / if-in-while /
   while+break / 嵌套 while / 多栈 / 标量跨迭代携带 / cholesky 形(实锤族)。
   每个: forward vs 手工参考 + 伴随 vs FD(rel 门, f32 下 1e-4 量级) +
   双 eps 稳定性检查。
2. **编译期平衡校验**(方案 D 产物): 对 mujoqo 全量 battery 内核跑一遍,
   输出违规清单(既有 battery 是现成的回归面)。
3. **集成**: mujoqo S1/S2/S3 冒烟 + S3 纯自动链(撤边界)回归 + battery
   7/7 风格一致性门。
4. **确定性**: 单线程/多线程双跑逐位一致(原位覆写+多线程原子序是已知变量)。

## 5. 资源估计

- 执行者: 本 agent(quadrants 仓 + mujoqo 仓均已就绪, A65/A68 同区域既往
  工作经验可直接复用)。
- 编译成本是主要摩擦(cfg_optimization off 纪律已固化, CLAUDE.md)。
- 总量: **6-10 个专注会话段**(P0 后不确定性显著下降; P2 前的估计置信度低,
  P2 后可收敛到 ±1 段)。
- 上游核对(半日): 该 pass 源自上游 genesis quadrants —— 先查上游 issue/
  commit 是否已知此限制(可能已有修复或讨论), 避免重复造轮子。

## 6. 风险

| 风险 | 应对 |
|---|---|
| H1/H2 均不成立(H3 类不可判定构型) | 方案 C checkpoint fallback; 成本升级但路径明确 |
| 修复破坏目前通过的内核 | P4 全量 battery + 内核族回归门前置 |
| 上游已有修复/不同设计 | P0 前先做半日上游核对 |
| 栈内存增长触发 overflow | determine_ad_stack_size 上限机制 + 降级路径 |
| mujoqo 撤边界后精度回退 | P5 回归门: 保持 1.47e-04 量级, 否则边界按列回退 |
