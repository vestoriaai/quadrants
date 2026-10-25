# A65 — reverse AD:混合顶层形态 kernel 的伴随解锁 — 需求评估(成文待 review)

日期:2026-10-24 | 线:`mujocoqd/a65-reverse-top-scalar`(基于 a63 线堆叠)
触发:mujoqo autodiff 线 P1(实测证据链完整,见下)

## 问题定义

`reverse_segments`(kReverse 编译路径)对"顶层语句 + 顶层 for"混合程序直接拒绝。
该形态是 mujoqo 移植的**承重惯用法**(R17/R18 head-declaration/早退模拟):实测
mujoqo 全库 367 kernel 中 200 个(54%)mixed;**梯度路径 135 kernel 中 mixed 为
主干**(_M/_cdof/_cinert/_kinematics_branch/_cacc_branch/_next_* 等)。拒绝 ⇒
编译器自动反向对 mujoqo 主链不可用,只能全手写伴随(Route A,~2x 代码量)。

## 实测证据链(本日)

1. **形态矩阵**(最小 kernel 四案):纯 for=OK;头部死标量/头部活标量/尾部标量=全部 REJECT
   (`reverse_segments.cpp:74`)。
2. **朴素放行(本线 trial commit)**:分段反转机制本就存在(代码读证),移除
   QD_ERROR 后——纯 for/已删死头的 mujocoqo kernel 伴随全部通过
   (_next_time/_position/_velocity/_compute_damping_deriv 等);
   但**含顶层标量段的伴随 kernel 运行期 SIGSEGV**(faulthandler 定位:
   `kernel.py:870 launch_kernel`,即伴随 IR 本身无效;复现 kernel=`_sensor_acc`
   及最小 case `done=False` 头标量)。
3. **round 伴随缺失**(独立小项):`make_adjoint.cpp:267` UnaryOp else 分支
   QD_NOT_IMPLEMENTED;本线已补 d(round)=0 零伴随(与 logic_not 同款)。
4. mujoqo 侧机械清理:94 处死 `doneqd=False` 头已删(可证语义保持;smooth 84P,
   全量回归失败集与基线逐位一致=既有挂账)。

## 2026-10-24 深挖增补(根因链完整定位)

本日试放行实验(round 零伴随 + 移除拒绝)推进到精确根因:

1. **已安全解锁**:纯 for kernel、死头 kernel(_next_time/_next_position/
   _next_velocity/_compute_damping_deriv 等,配合 mujoqo 侧 94 处死头清理)、
   round 算子(零伴随已入 make_adjoint)——上述全绿。
2. **carried 类(头初始化 + 循环内累积)的完整根因链**:
   - 前端 transformer 已把 `acc += x[i]` 转成局部 alloca 上的 atomic
     (primal IR 实锤:`@tmp3 = atomic_add(<*f32>@tmp1, x[i])`,tmp1=head alloca);
   - demote_atomics(kReverse 路径先于 auto_diff)再把局部 atomic 拆成
     LocalLoad+BinaryOp+LocalStoreStmt;
   - make_adjoint visit(LocalStoreStmt) 对该 store 生成
     accumulate(val, load(adjoint(dest)))——adjoint alloca 定义在循环体外的
     head 段,伴随 kernel 内该指针无效 → SIGSEGV at adjoint launch;
   - 伴随 IR 实锤:串行反转算法被发成 block_dim=adaptive 的并行循环,
     循环内 atomic 写共享栈槽 = 必然崩溃。
3. **两次拦截尝试未触发**(reverse_segments 前端 lhs 身份 x2 作用域问题、
   make_adjoint LocalStore 守卫 x pass 顺序假设)——需正式会话用 QD_DUMP_IR
   全链 dump 逐 pass 追,黑盒迭代每轮 ~5min 构建不经济。
4. **结论(2026-10-25 修订)**:make_adjoint LocalStore 跨段守卫在**真实移植
   kernel 上验证有效**——mujoqo P2 分段探针中 `_next_position` 的 carried 形态
   被正确响亮拒绝(RuntimeError 带重构指引)。case3 最小 case 的 segv 属于
   **第三条未覆盖路径**(守卫未触达;疑:atomic 因 loop-unique 条件未被 demote,
   走 visit(AtomicOpStmt) 之外的路径,或 LocalLoad 伴随侧)——该路径仍需
   QD_DUMP_IR 逐 pass 追。blanket 拒绝对 carried 类方向正确,已恢复;
   本线保留:round 零伴随(真实 kernel 依赖)+ LocalStore 守卫(已验证)+
   全部诊断件。
5. mujoqo P1 影响:自动反向可及面 = 死头/tail kernel;carried 类(~43 活头
   kernel)按 Route H 既定回退走 grad_replaced 手写伴随(P0③ 已验证该注入
   架构),或下会话以专项继续 A65 编译器修复。

## 2026-10-25 实施结果(frontend first-touch 检测落地)

**已落地(本轮)**:
1. **reverse_segments 混合形态定向放行 + 前端 first-touch 检测**:`CarriedDestFinder`/
   `collect_loop_writes`/`walk_first_touch`——头段赋值变量在循环段"先读后写"(真
   累积)→ 响亮拒绝;先写后读(每迭代临时)→ 放行;只读(循环不变量)→ 放行。
   形态矩阵四案全部符合预期(plain/dead-head/tail 放行,carried 拒绝)。
2. **round 零伴随**(保留,真实 kernel `_next_activation` 依赖)。
3. mujoqo 侧 94 死头清理配合后,梯度路径试编译 12 个伴随连续通过
   (_next_time/_next_position/_next_velocity/_next_activation/_small_cholesky/
   _solve_done 等)——比 blanket 拒绝时代多解锁一批。

**新边界(诚实登记)**:`solver._update_gradient_cholesky`(Newton 小矩阵 Cholesky,
嵌套 qd.static 循环 + 标量临时)的伴随生成仍 segv——front-end 检测通过(先写后读
形态),segv 在 make_adjoint 深处,属**另一层编译器缺口**。诊断工具链已备:Debug
wheel 构建路径(注意 build.py 钉死 RelWithDebInfo,须改 CMakeCache 后增量重建)+
QD_DUMP_IR。另:while-loop kernel(`_tendon_dot` 族)的 WhileStmt 伴随仍未实现
(QD_NOT_IMPLEMENTED),是独立缺口。

**根因认识修正(重要)**:make_adjoint 层(SSA/后 simplify)的 first-touch 分类
**不可行**——store-to-load forwarding + DCE 删除 init store 后,每迭代临时呈现
read-first(误报,实证 `_small_cholesky` 的 `diagonal_inv`);正确层级是前端
(本实现)。同时 QD_ERROR = log + `__builtin_unreachable`(UB,不抛异常)——
编译器 pass 内做"响亮拒绝"必须 throw QuadrantsError。

## 2026-10-25 专项② 实施轮结果(WhileStmt 反向 visitor 已落地, 链验证未全绿)

**已实施(提交 e93129ca1..074426a9e)**:
1. make_adjoint `visit(WhileStmt)`——线性控制回放反向 visitor:carried allocas 全非
   real 才接受(real 携带响亮拒绝 = adstack-for-while 未做的部分);支配链入口态
   恢复(根块→容器程序序扫描取最近前置 store);体伴随走查用指针副本迭代(修嵌套
   erase 失效)。
2. `visit(WhileControlStmt)/visit(ContinueStmt)` no-op——控制回放形式下
   break/continue 的伴随由克隆条件重放覆盖。
3. post_adjoint_cleanup 接受 WhileStmt;ir.h `as<T>` 失败类型打印(永久调试辅助);
   kernel.cpp 非 Block root 诊断。

**验证结果(锚 = _tendon_dot_for 场景链)**:
- transmission 伴随生成**完整通过 36/36 stmts**(WhileStmt 反向 visitor 生效实证);
- 伴随编译的并行任务组随后在 **`ext_arr_to_ndarray_c18_0`(kNone 工具 kernel)的
  `Kernel::init` 内 `as<Block>()` 失败**——body 实证为 GlobalLoadStmt(typeid 打印);
- **定性**:伴随走查对共享 IR 态的改写(transmission/tendon_armature 的多 while +
  adstack 栈回放形态)泄漏/破坏了后续 kernel 创建路径, 或 make_adjoint 生成的
  while 反向 IR 与 adstack 栈 op 不一致(嵌套 while + int carried + 守卫分支的
  组合形态)。基线时序实证:该失败在 A65 WhileStmt visitor 之前的映射运行即出现
  ——**pre-existing**, 非 visitor 引入;
- kernel 侧配套:_tendon_dot 机械改写(while→for + pulley 守卫反转 + terminated
  闩锁, 修复移植版潜在死循环)+ h_incremental continue 反转 + _comvel_branch 反转
  ——tendon parity 35/35、上游 AD 测试全绿。

**剩余主体(专项② 续)**:
- WhileStmt 反向 + adstack 交互对"嵌套/守卫 while 形态"的 soundness(堆损坏/根
  非 Block)——需 make_adjoint 栈 op 与 WhileStmt 体逐语句对的 IR 级审计
  (QD_DUMP_IR 全链 + -O0 make_adjoint 路径已验证可行);
- 该问题独立于 μjujoqo 自动伴随链——修复后 `_tendon_dot_for`/`_transmission`/
  `_tendon_armature` 的伴随应可完整编译。
## 分阶段规划(2026-10-25 定稿)

- **阶段 0(已完成)**:源头治理——mujoqo 侧 94 死头清理 + p1 残留清理线 9 处跨 `__t`
  累积真 bug 修复(梯度路径 live-carried 43→0);round 零伴随。
- **阶段 1(已完成)**:前端定向检测——reverse_segments 混合形态放行 +
  first-touch 两遍法(真累积拒绝/每迭代临时与只读放行);矩阵四案回归锚。
- **阶段 2(专项,3-5 人日)**:`_update_gradient_cholesky` 伴随 segv——Debug wheel
  回溯 → 定位(疑:嵌套 static 循环 + 提升 allocas 的伴随栈分配)→ 修复 → 矩阵 +
  试编译 + 数值门。
- **阶段 3(专项,2-4 人日)**:WhileStmt 伴随缺口(`_tendon_dot` 族)。**裁决建议:
  机械改写器优先**——①`while j < num-1` 在源级就是 `for j in range(num-1)` 的语法糖
  (迭代数一致、pulley-continue 在 for 下保持、无多余迭代,语义等价可逐位对照);
  ②改写后 AD 走 RangeFor+adstack 成熟路径(附录 A 支持的运行期 ndarray 界形态);
  ③不碰编译器,风险低;④lint 双向同步禁新增 while。仅当未来出现"无法改写的真数据
  依赖循环"(如迭代精化 `while x<tol`)才立项编译器 while 伴随(高风险:adstack×
  while 组合的边界 bug 面大,RangeFor adstack 的历史 bug 清单是前车之鉴)。统一原则:
  **并行语义下不合法的形态→拒绝(教育);合法但未被支持的形态→机械改写到支持形态
  (工具化);只有合法且无法改写的才进编译器**。
- **阶段 4(持续)**:长尾——每暴露一个新拒绝按族归类,同族批量修;每族过
  L1/L3 FD 门(§6.4)。
- **出口判据**:mujoqo 无接触链(标记 `disableflags = contact|sensor`)整步 Tape
  自动伴随全绿 + contact/BPTT/FD 三探针保持绿 + 上游 test_ad_* 全绿。
- **不做 A65 的代价**:整步 FD 块已覆盖功能(梯度可用但伴随成本 O((nv+nu)×2) 次
  重放);A65 是把反向成本压到 ~2-3× 前向 + 免逐 kernel 手写的路径,非阻塞项。

## 修复方向(供 review 裁决)

伴随生成对"非 for 顶层段"的真实支持。候选:
- A(推荐起点):诊断 segv——dump 伴随 IR(QD_DUMP_IR)定位无效代码形态
  (疑:局部 alloca 的伴随跨段/无 offload 上下文);若为局部标量的伴随分配
  问题,补 make_adjoint 对非 Offloaded 上下文局部 store 的处理;
- B:顶层标量段包成 serial offload 后走既有伴随;
- C:reverse_segments 检测"顶层标量不参与环携带数据流"时自动前移进 for 头
  (仅对死头安全,活头仍拒)——最小侵入但覆盖率存疑。
- 验证:形态矩阵四案梯度数值对照 FD;mujoqo P1 全链 trial;既有 AD 测试全绿。

## Footprint 预算

reverse_segments.cpp / make_adjoint.cpp / 测试;预计 3-10 人日(视 segv 根因深度)。

## 2026-10-25 A68 解决轮(WhileStmt 反向三缺陷修复, 无接触链自动伴随全绿)

**A68 定性(逐阶段 dump 二分 + lldb)**:
- 失败点 = scalarize.cpp:847 `stmt->stack->as<AdStackAllocaStmt>()`,receiver 实为
  GlobalLoadStmt;调用链 = codegen per-task worker → offload_to_executable → scalarize
  ("parallel compile task failed" 来源 ParallelExecutor)。
- 手段: compile_to_offloads 主线程补 a3-a9 阶段 dump + offload_to_executable(per-task)
  补 01-14 阶段 dump(QD_DUMP_IR=1, task_index 消歧)+ a68_scan_dumps.py 扫描器。
- 二分: a3_pre_autodiff ok → a4_after_autodiff BAD → make_adjoint 发射的
  AdStackLoadTop stack 操作数悬空(a4 有 4 条, post simplify 剩 1 条存活到 scalarize,
  DIE 释放后 UB 拾到 GlobalLoadStmt)。主线程 offload 把悬空操作数重写成
  `global load(GlobalTemporaryStmt)`(仅 load 无 store,佐证悬空)。

**根因(visit(WhileStmt) 三缺陷, 全部修复)**:
1. `erase(while_stmt)` 删除正向 while → 伴随发射共享的正向 body 语句(if cond、
   提升栈)悬空。修 = 保留正向 while(range-for 协议对齐)。
2. clone body 全清 → `while true` 无终止。lowered 形态实证:
   `while true { const1; WhileControlStmt(1); 守卫更新; if guard {} else { const0;
   WhileControlStmt(0) }; body }` —— 守卫头保留方案不可行(栈顶是失败测试的
   cond, 立即 break)。修 = 合成迭代轨迹栈: 正向终止 if 后每迭代 push 1-marker
   (栈底 0 哨兵), adjoint while 头 peek,sentinel 浮出 break;LIFO 使回放迭代
   k 恰消费第 k-from-last 迭代的 push 块(迭代 N→1 序, 伴随累加顺序无关)。
3. 体伴随走查正序 → consumer 晚于 producer 发射, 链式法则读 partial。
   修 = std::reverse(与 RangeForStmt 同理)。
配套: make_adjoint(block, ad_stack_size) 贯通;real-typed carried 响亮拒绝保持;
ir_shaping IdentifyIndependentBlocks WhileStmt 拒绝(Logger::error 实为 log+throw
实证)→ 不抛不 descend;Tape.grad 摘除 target_tape(grad 阶段 kernel 发射不再被
记录, BPTT/FD 块的宿主重跑不再触发无关 adjoint 编译)。

**验证(全绿)**:
- 无接触链 102/102 adjoint 编译通过, islands OFF(_tendon_dot/_transmission
  的 island 不再承载, 保留作 escape hatch);
- 端到端自动伴随 FD 门(p1_autoadj_fd, 新): rel_err = 2.03e-06 —— while 回放
  数值正确性的首个硬判据;
- BPTT 1.49e-4 / contact 4.67e-5 / FD 4.67e-5 三探针绿;
- 上游 test_ad_* 357 passed;
- 零梯度澄清(p1_zerograd_probe): 全 tape 后读 .grad 为零 = overwrite 消费
  协议 + qvel.grad 真值为零(宿主 FD 实证), 最小 tape qacc.grad=-0.05 精确。

**新增边界/账目**:
- real-typed carried while 仍响亮拒绝(adstack-for-while 真形态 = 未来工作,
  轨迹栈只解决非 real 线性体的终止/配对);
- visit(WhileStmt) 的 termination-if 定位要求 lowered 形态含 IfStmt-branch break
  (当前 lower_ast 保证), 无则响亮 QD_ASSERT_INFO;
- ir_shaping 不再 descend while body → while 内嵌套 range-for 不作为独立 IB
  (由 MakeAdjoint 走查统一处理), 对无 while kernel 零影响。
