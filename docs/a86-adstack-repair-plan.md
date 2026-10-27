# A86: adstack/自动反向修复计划 v2(追踪四十八; v1 经第一性原理评估后修订)

日期: 2026-10-27 | 状态: 计划书 v2(Phase 0 实验未执行)
关联: mujoqo-iftwt `docs/a86-bptt-closure-report.md` §6-7/D1-D5、账本追踪三十四/四十二/四十六
v1→v2 变更: H1 由"经验假设"升级为"结构不健全的第一性原理推导"; 修复方向由
"kill-switch 门控"改为"叶子合法性收紧 + 只读标注"; 补 purity 检验与 act 覆盖缺口。

## 0. 已知事实(带证据)

| # | 事实 | 证据 |
|---|---|---|
| K1 | builder 内核(while+嵌套 if+标量跨迭代)自动反向产出 inf/垃圾 | mujoqo S1 链 M_scratch.grad=inf; c623 reverse IR 除数来自 unclamped `stack load top` |
| K2 | 栈配对不对称实录($7880014 peek-无-pop), 是否设计**未定性** | 追踪三十四/三十五 IR 审计 |
| K3 | step_ift 单步边界下每 stage 伴随应用精确(1.5-2.3e-07) | 追踪四十六 a86_steprec |
| K4 | 受影响内核族: cholesky factorize/solve、RNE、transmission | mujoqo `_src/smooth.py` |
| K5 | IFT/FD 边界已绕开全部受影响路径, 三 sample 全 PASS | mujoqo cdd2d30 —— 本缺陷当前**潜伏**, 是"未来新内核的正确性 + 撤边界性能"问题 |

## 1. 第一性原理推导(替代 v1 的经验假设)

**架构不变量**: 纯重放 tape + 可变状态之下, 反向核消费的每个前向值必须满足二者之一:
(i) 该值所在存储在 forward 写入后至 reverse 读取前**未被任何执行改变**;
(ii) 该值在 forward 时被**快照**(adstack push 即此机制)。
不存在第三选项。

**由此推导 eliminate_recomputable_ad_stack_pushes 的结构性不健全**:
该 pass 的判据(leaves ∈ {ArgLoad, Const, LoopIndex, AdStackLoadTop/Alloca};
SNode leaves 门控"未在 IB 内变异")检查的是**本 kernel IR 内**的变异 —— 但
tape 系统中"谁会在 forward 与 reverse 之间改写该数组"是**运行期信息**
(重放序列编译时未知)。具体地:
- **ArgLoad 叶子**: kernel 数组参数的内容在 forward 与 reverse 之间可被任何
  更晚的 forward kernel 原位覆写(mujoqo: qvel/qfrc_*/M/qLD/efc_* 全是)。
  反向时 DAG 重执行读到的是**终态内容** → 重算值 ≠ 前向值 → 该值的一切
  下游伴随错。
- **BackupSSA 的跨块 SSA 重执行**(make_adjoint)同构同险。
- 上游纯净用法(每 kernel 只读 args 不原位改他者)下不触发 —— 解释为何此
  缺陷在上游潜伏而在 MuJoCo-warp 式原位状态变更移植中爆发。
- 推论: **判据对 tape 用法是必要非充分的; 静态分析原则上无法覆盖
  (重放变异集是运行期信息) —— 唯一健全出路 = 收紧叶子合法集 或 快照。**

结论: H1 由"假设"升级为**推导出的结构缺陷**; P0 实验只负责确认 mujoqo 现场
触发路径(哪个数组/哪条链), 不再负责判断"是否有缺陷"。

## 2. 修复设计(v2, 从不变量推导)

**F-a(正确性主体): 叶子合法性收紧 —— 可重算链禁止可变 ArgLoad 叶子。**
- RecomputableChainAnalyzer 的 leaf 集删去 ArgLoad(或收紧为带只读标注的
  arg); Const/LoopIndex/栈化 allocas 保留。
- 效果: 触碰可变数组的链保留 push(快照 ✓ 满足不变量 ii); 纯常量/索引链
  仍消除(✓ 保留大部分收益: loop-index/上一迭代值链是频次最高的形态)。

**F-b(性能保留): 只读标注机制。**
- mujoqo 的 Model 侧数组(m.*)在 put_data 后不可变 —— 标注只读后, 触碰
  m.* 的链恢复可消除(实例: gain/bias/mass 常量链, cholesky 的行地址表)。
- 形态: qd 层 API(如 `qd.mark_read_only(arr)`)或 arg 注解; 消除 pass 的
  analyzer 识别标注。

**F-c(守门): 消除 pass 增加 config knob(默认维持现行为), 供诊断与回归
  A/B; 长期默认值随 F-a/F-b 落地后再翻转。**

**不选择**: 全局 kill-switch(v1 方案 A)—— 一刀切恢复正确性但把消除收益
(头注自述 5-10x push 成本节省)全部丢掉; 收紧+标注在保留多数收益的同时
满足不变量。

## 3. 阶段计划(gate 与退出条件)

**P-1 上游核对(半日, 最先做)**
- 该 pass 源自上游 genesis quadrants: 查 issue/commit/设计文档是否已知
  "原位变更下不健全"; 若上游已有修复/讨论, 直接对齐。
- Gate: 结论(已知/未知/已修)写入本文件再进 P0。

**P0 现场触发确认(0.5-1 天)**
- 构建最小复现: 含 ArgLoad 叶子的 while+if kernel(手工写, 非依赖 mujoqo),
  forward 改写叶子数组 → reverse → 断言错值; 现行 HEAD 复现。
- 再跑 mujoqo S1 无边界链加 knob(F-c)对比: inf/垃圾消失路径确认。
- Gate: 最小复现成立 + knob 闭环 → 进 P2; 不成立 → 推翻推导, 回到 P-1。

**P1 平衡审计工具(1 个工作日)**
- IR pass: 按 AdStackAllocaStmt 输出 push/pop/load-top 事件序列与配对报告;
  附运行期计数插桩。定性 K2($7880014 peek: 设计 or bug)。

**P2 修复实现(1-2 个工作日)**
- F-a 收紧 + F-b 标注 + F-c knob; 伴随链(quat 列)回归。

**P3 测试(1 个工作日)**
- 控制流构型矩阵单测(for+while/if-in-while/while+break/嵌套 while/多栈/
  标量携带): 伴随 vs FD(rel 门 1e-4 量级, f32)+双 eps 稳定性。
- 平衡校验跑 mujoqo 全量 battery 内核, 输出违规清单。
- 单/多线程逐位一致。

**P4 mujoqo 接入(1-2 个工作日)**
- S3 撤 FD 边界 → 纯自动链回归(门: 保持 1.47e-04 量级);
- S1/S2 回归; P4 反向画像重做(解析链反向应 ≈ 正向 2-3 倍);
- ad_ift 仅保留求解类解析 IFT(长期, D0 本意)。

## 4. 资源估计(v2, 场景化)

- P-1+P0: 1-1.5 天。P1: 1 天。P2: 1-2 天。P3 测试: 1 天。P4: 1-2 天。
- 合计 **5-7.5 个工作日**(v1 的 6-10 会话段 → 场景化后收紧; P-1 若发现
  上游已修则 P0-P3 塌缩为升级适配, ≈2 天)。

## 5. 风险

| 风险 | 应对 |
|---|---|
| 收紧后消除收益大幅缩水(栈内存/overflow) | F-b 只读标注找回大头; determine_ad_stack_size 上限机制 + 降级路径 |
| 上游已修/设计如此(有意的 trade-off) | P-1 前置核对; 若上游有意, 本计划转为"mujoqo 场景的 sound-subset 上游 PR" |
| quat 分量扰动触发非光滑(单位范数约束) | 列级有效性检查复用翻转守卫模式 |
| 修复破坏现有通过内核 | P3 全量 battery + 内核族回归门前置 |
| mujoqo 撤边界后精度回退 | P4 回归门: 保持 1.47e-04 量级, 否则按列回退边界 |

## 6. 随发现补充的缺口登记(本轮新增)

- **边界输入覆盖**: step_ift/rollout_ift 的 FD 输入组为 (qpos, qvel, ctrl);
  通用模型需加 act(na>0)——pendula na=0 未覆盖, 登记为通用化缺口。
- **purity 检验**: step 是否为 (qpos,qvel,ctrl) 的纯函数未显式验证(全量
  restore vs 部分重置对比)—— 若不纯, 冒烟 g_ref 方法论需改全量 restore。
- **探针确定性配置对齐**: 跨进程对比实验须统一 cpu_max_num_threads=1
  (此前种子对比的跨进程差异含此因素)。
