<!-- INDEX — merged lines (newest last; each entry's full section via its header below) -->
<!-- qd-copy-api 2ce5a850 · c2-emitter-fixes 83b2d1a2 · subgroup-cpu-support 8ebf8620 · reassociate-flag 19ff0276 · reassoc-subgroup-probe 9372c300 · r5-host-visible 67e27bc4 · spirv-vectorize 68bd7965 · algorithms-2 b12c125c · compiler-line 33e7dabf · n2-to-numpy-type 4a76ce7c · a10-nested-loop-raw c52711b5 · a19-stlf-fallback 3793c86d · a20-loop-var-scope 01d2037e · metal-i64 01d2037e · a7-m2-compound 869a00a0 · qd-native-bvh 76541b81 · cpu-head-state-per-iteration 4a400a11 · a32-static-return-order e0f35bdd · a34-bool-cast 91351dbd · a37-compile-uaf 4da921ff · familyr-realfunc-composite 57391b9f · audit-lint 2bb6b7f4 · ruff-import-sort 41175f2a · a46-recursion-guard (this) -->

# robostudio_features.md — features ledger for robostudio-main

> **维护约定（本文件头即规则）**：每一个合并进 robostudio-main 的
> feature commit，**必须在同一次 merge 里更新本账本**——新条目（含
> 描述/背景需求/状态三栏）或既有条目的状态推进。合并 PR/分支时核对
> 此项；账本与分支内容不一致 = 合并不完整。
>
> 每条目格式：**描述**（做什么 + 触碰哪些文件）/**背景需求**（为什么：
> 哪个战役/事故要求的）/**状态**（landed / verified / 哪个 gate 名）。
> 逐 commit 谱系见 `git log robostudio-main` 与
> `docs/consolidation_robostudio_main.md`。

## S2 — loop-var assignment guard
- **描述**：运行时循环变量（range-for/struct-for/mesh-for target）在循环体内
  被赋值时，AST 前端直接抛清晰的 `QuadrantsSyntaxError`；此前会 lower 成
  `LocalStoreStmt(LoopIndexStmt)` 撞 dest-type assert（晦涩 RuntimeError），
  增广赋值形态更是未捕获 C++ 异常直接杀死进程。ndrange/grouped 目标保持
  可赋值（alloca-backed）。文件：`python/quadrants/lang/ast/ast_transformer.py`
  （注册点）、`ast_transformer_utils.py`（`register_loop_var` /
  `check_assign_not_loop_var`）。移植自 overlay `patches/s2_fix.patch`。
- **背景需求**：robostudio S2 战役（2026-10-08）：parallel-reduction 内核
  在循环里写累加变量触发进程 abort（exit 134），scratch/bspline 线首发。
  `register_loop_var` 即原 install_qds.sh overlay 检测 marker。
- **状态**：landed（1d645464）+ verified — gate `probe_s2_loopvar_assign`
  RESULT: PASS（union runner gate 1/6）。

## S3 — static-if branch-name join
- **描述**：`qd.static` 分支只在编译期执行一次，语义上其绑定应落入外层
  作用域；原来 `build_stmts` 的 per-block scope 吞掉绑定，join 之后用该名
  即报 `Name "y" is not defined`。现 static 分支以 `scope_guard=False` 在
  外层作用域构建（运行时 if 的 alloca 仍分支内，不提升）。文件：
  `python/quadrants/lang/ast/ast_transformer.py`（build_If）、
  `ast_transformer_utils.py`（build_stmts 签名）。移植自 overlay
  `patches/s3_fix.patch`。
- **背景需求**：robostudio S2/S3 战役（2026-10-09，quadrants 坑清单
  "static-if 作用域"）：内核族大量使用 `if qd.static(...)` 分支配置 +
  join 后使用模式，全数被 NameError 挡死。`scope_guard` 为第二 marker。
- **状态**：landed（2d9a99ca）+ verified — gate `probe_s3_static_if_forms`
  10/10（union runner gate 2/6）。

## C0S — cache-epoch + dead-runtime capture walk
- **描述**：编译产物缓存按 SNodeTree 生命周期分段：`KernelCompilationManager`
  的 `cache_epoch_` 混入内核 key，`Program::add_snode_tree`(id≥1) 与
  `destroy_snode_tree` 递增（单一 tree 的常规程序永不 bump，盘上 key 稳定）；
  python 侧同事件清 `materialized_kernels` + `compiled_kernel_data_by_key`
  （round-2 修复：launch 直查 compiled_kernel_data_by_key）。死亡 runtime
  防护：field 记录出生 runtime（`_qd_owner_runtime`），cache-miss 浅走
  py_args 拒绝已清 runtime 的捕获（后被 union 整合收窄，见下条）。文件：
  `quadrants/compilation_manager/kernel_compilation_manager.{cpp,h}`、
  `quadrants/program/program.cpp`、`python/quadrants/_snode/fields_builder.py`、
  `python/quadrants/lang/impl.py`、`kernel.py`。移植自 overlay
  `patches/c0s_fix.patch`（overlay 提交 a3be630..5059321）。
- **背景需求**：robostudio C0 战役（多 init / 多 SNodeTree 家族事故，
  curobo_alignment_plan C0-D；upstream taichi 1.7.4 同样崩）：旧缓存跨
  tree 复用 → 脏 field 绑定 → OOB/SIGSEGV。该修复是已部署 dev7 wheel
  的一部分——不移植则整合 wheel 静默倒退。
- **状态**：landed（6ea5ec4b）+ verified — gate `battery_c0s_multi_init
  --single-init` 4 arms PASS（union runner gate 6/6）；多 init 形态语义
  变更见 S-guard 条。

## S-guard — stale-field refusal（含语义变更诚实注记）
- **描述**：`PyQuadrants.generation` 每次 reset() 递增；`Field` 创建时记录
  generation，全部宿主路径（getitem/setitem/from_numpy/to_numpy/to_torch/
  fill/copy_from，ScalarField + MatrixField）经 `_check_not_stale()` 拒绝
  已 finalize 程序的 field；`impl.subscript` 为 field 进 IR 的唯一收口点
  同样拒绝；reset() 在仍有存活 field 时告警。文件：
  `python/quadrants/lang/field.py`、`impl.py`、`matrix.py`、
  `tests/python/test_stale_field.py`（4 测试）。cherry-pick 自
  overnight/inv-cache-keys@80445172（port/sguard-stale-field）。
- **背景需求**：夜探 Inv1/Inv2（2026-10-11，tech_debt_todo T25b/S-guard）：
  `qd.init#2` finalize Program#1 并重置进程级 HostMemoryPool，Python Field
  存活带裸 SNode 指针 = use-after-free（同 arch 也炸）——robostudio C0-D /
  多家族事故的机制定案，S=响亮 stale-field 检测为对尺寸修复。
- **语义变更（诚实注记）**：re-init 后复用旧 field 从"碰巧能跑"（实为
  UAF、靠池内存未覆写伪装成正确值）变为**响亮拒绝**
  （QuadrantsRuntimeError "finalized"）。C0S battery 的多 init 形态
  （a2b/ab_a/jit_heavy 重建陈旧家族并重发）因此按设计 RAISE；受支持契约
  = one-init-per-process（`--single-init` 形态 4/4 PASS）。这是**行为
  变更**：依赖"re-init 后旧家族继续出正确值"的流程须改为重建对象或
  单 init。
- **状态**：landed（b778ab20）+ verified — `tests/python/test_stale_field.py`
  4/4（union runner gate 5/6，需 `-p xdist -n0`）；生产 venv 直查拒绝行为
  亦验证。

## u8 DLPack
- **描述**：`to_dlpack` 导出 u8/i8/u16/i16/u32/u64 ndarray（G1 类型矩阵
  双向写可见、G2 零拷贝、G3 device 忠实、G4 别名契约、G5 端到端内核
  样本）。文件：`quadrants/python/dlpack_funcs.cpp`。谱系：upstream
  feat/u8-dlpack@4b2e3c77（经 feat/readback 成为基座；overlay 侧移植
  f0447a0 zero-drift）。
- **背景需求**：qd-render-poc R4/§3.3：u8 渲染样本管线需要 uint8 与
  torch 的 DLPack 互操作。
- **状态**：landed + verified — gate `gate_u8_dlpack`：cpu ALL GATES
  PASS；metal **恰好**钉死的 f64 cell 失败（qd SPIR-V 无 f64，T19；
  f64 全绿之日须有意更新 pin）——union runner gate 3/6。

## readback R2/R3
- **描述**：R2 = metal `readback_data` 对 host-visible 源走 direct map
  路径；R3a = 外部数组镜像缓冲持久池（64MiB 容量上限、drain 点回收，
  免每次 readback 的 `newBufferWithLength` 往返）；R3b = per-call commit
  模型文档化。文件：`quadrants/rhi/metal/metal_device.{h,mm}`、
  `quadrants/runtime/gfx/runtime.{h,cpp}`、`python/quadrants/lang/_ndarray.py`。
  谱系：feat/readback@4fe1bbe2+7aaaa7d7（分支基座）。
- **背景需求**：T8（cuRobo 对标读回杠杆）：to_numpy 64MB 5.56ms vs torch
  2.12ms 的余量分解；每次读回都付一次 buffer 分配往返。
- **状态**：landed + verified — gate `gate_readback_r2r3`：64MB
  to_numpy ≤7ms / from_numpy ≤5ms（实测 ~5.4/3.4ms 档）+ 1KB/4MB/64MB
  字节正确性扫（union runner gate 4/6）。

## Readback gate-4 perf pins — machine-variance caveat(mujocoqd/compiler-line A/B 实测,2026-10-11)
- **数据(修正版)**:两次相邻构建实测 to_numpy 7.72 / 9.45ms、from_numpy 7.89 / 6.01ms
  ——事后核查两次构建**都含** per-access volatile hunk(该 hunk 已随 C5 的 add -A 进
  HEAD,"对照"未成立),因此这两组数字之差 = **纯机器态方差(±40%)**,与 volatile 改动
  无关;union gate 4 的绝对毫秒钉值(≤7.0/≤5.0;原值 5.561/3.239 出自另一机器态)在本机
  当前态下不可稳定满足。正确性字节扫全过,与后端改动无关。volatile-vs-plain 的真实性能
  A/B 未做(可选后续)。
- **结论/建议**:gate 4 的绝对 ms 钉值应改为**相对钉值**(如同跑 numpy memcpy 的 ≤2×)或
  跑前记录机器基线;本轮 mujocoqd/compiler-line 的 5/6 中 gate 4 FAIL 判定为方差,非回归。
- **状态**:open(钉值策略调整待用户裁决;本轮不静默放宽门限)。
- **描述**：C0S 的 cache-miss 粗粒度捕获走查让位给 S-guard：带
  generation 戳的捕获交由精确收口点（`impl.subscript` →
  `_check_not_stale`）拒绝；粗走查仅继续拦截**无** generation 戳的捕获
  （外来/前代 field 对象的安全网）。文件：`python/quadrants/lang/kernel.py`。
- **背景需求**：两修复合并的次序冲突：粗走查先开火会拒绝"携带但未读"
  陈旧 field 的内核，破坏 S-guard 自己的
  `test_kernel_not_reading_stale_members_still_compiles`（scan_nd 不读
  field，必须照常编译）。
- **状态**：landed（60003393）+ verified — stale battery 4/4（gate 5/6）。

## Gate suite — tests/robostudio/ + run_robostudio_gates.sh
- **描述**：分支内自助验证套件：S2 探针、S3 形态矩阵、u8 门（cpu+metal，
  钉死 f64 期望失败）、readback 性能 pin、stale-field pytest battery、
  C0S `--single-init` battery，一条命令跑完全部并要求 6/6。文件：
  `tests/robostudio/{probe_s2_loopvar_assign,probe_s3_static_if_forms,
  gate_u8_dlpack,gate_readback_r2r3,battery_c0s_multi_init}.py`、
  `_gpu_busy.sh`、`run_robostudio_gates.sh`（前五个为验证整合时的原脚本
  逐字 vendored；readback 门自 post_install_gate.sh 的 heredoc 逐字提取）。
- **背景需求**：整合后复查：分支自称 source of truth 却无法自证——
  探针散在 robostudio worktree / simtest 第三仓 / overlay 树，宝可失。
- **状态**：landed（2efc351a）+ verified — `UNION GATE: 6/6 PASS`
  （consolidated wheel 实跑两次）。

## Blit-pool fix — ext-arr mirror homogeneity（粘性 h2d map 失败根修）
- **描述**：`GfxRuntime` 的外部数组镜像改为**无条件 host 可见**
  （`host_write=true`，含 detected access==NONE 的死参数——它们自身的
  blit/d2h 门仍按 access 跳过，但其镜像会在 drain 时进入
  `ext_mirror_cache_` 池）。修复前：access==NONE 的参数拿到 Private
  （Metal）镜像 → 该镜像入池（池**只按 size 键控**）→ first-fit 把它
  发给后续需要 blit 的活参数 → `map()` 对 Private 永远 invalid_usage →
  抛 "Failed to map ext arr data buffer for host_to_device blit"，毒化
  条目回池、重试再抓再败 = **粘性、按问题确定、重试不可恢复**。附带
  `MetalMemory::mapped_ptr` 的 QD_BLIT_DEBUG 诊断（区分 can_map=false 与
  contents nil）与 h2d 失败消息的 arg_id/size/alloc_id。文件：
  `quadrants/runtime/gfx/runtime.cpp`（镜像分配 + 失败消息）、
  `runtime.h`（池不变量注释）、`quadrants/rhi/metal/metal_device.mm`
  （诊断）。
- **背景需求**：cavity T26 战役（2026-10-11）：4 组分裂（4×287 MiB 驻留
  ESDF 场）下 mppi_coll 的 host_to_device blit 从 ~10 发射起成簇粘性
  失败——189 集中 idx 1/3/4/8/9/10 每个新进程必现（全是 16 发射的
  iters_used=5 题；4 发射题幸免），重试不可恢复，与 ~2.25% 随机 flake
  不同；robostudio 被迫把 cavity 接线退到 n_group=1（575 MiB）。探针
  `probe_blit_4group.py`（n_group=4 复现 8/10 失败 / n_group=1 全净）+
  QD_BLIT_DEBUG 定位：每次失败都是同一个 can_map=false 的 672000B 池
  条目。
- **状态**：landed + verified — 同一复现 n_group=4 修后 **0 失败**
  （10/10 过）；`UNION GATE: 6/6 PASS`（含 stale battery）。cavity 全量
  189 @ n_group=4（修后 wheel）：**176/189、routes/overflow 集逐位同、
  决策 parity 189/189、mapfails_total=0**（vs 记录基线
  t26_cavity_cmpolish：同 176/189）——cavity 可从 n_group=1 切回 4 组
  （p50 0.11→0.09s；cavity 仅 2 polish 题，总墙收益小，价值在解除
  retreator + 恢复 table 已验证的 4 组形状）。wheel=
  quadrants-1.3.4.dev25+g3297e86fe。

---

# mujocoqd/algorithms-2 条目(2026-10-11;**已合入 robostudio-main**,--no-ff)+ 下述 compiler-line 条目(2026-10-11;已合入,merge 33e7dabf --no-ff)

> 来源战役:mujoco-qd 移植 P0(quadrants 替换 mujoco_warp kernel;方案与探针在
> `/Users/mini0/Code/mujoqo/`)。基线 dev25 探针重跑后开工;每个 landed 条目带 gate 进
> `tests/robostudio/`。合并时 union gate 5/6:gate-4 readback 性能钉值为机器方差
> (正确性全绿,见"Readback gate-4 perf pins"条,钉值策略待裁决);其余 5 门全过,
> 另有本线新 gate(early_return / tile_matmul / algorithms_cpu / metal_loopdep /
> tile_runtime_loop)全绿(Apple 侧 A2/A5 为 PINNED 状态按设计)。源码级快速回归:
> upstream test_return.py 46/46。

## M2 — early_return 单出口 AST 改写(opt-in)
- **描述**:`@qd.kernel(early_return=True)` / `@qd.func(early_return=True)`。变换器
  `ast/ast_transformers/early_return_transformer.py`(新增)在 build_stmts 前把运行时
  if/for/while 内的 return 机械改写为单出口(`__er_ret{i}` 按注解 `qd.cast` 零初始化 +
  `__er_done` 标志 + 后继语句包 guard + 循环内 break)。**break 策略分治**:func 内一律
  break(内联后嵌套,合法);kernel 最外层并行循环只 guard 不 break(前端禁最外层 break,
  且 SIMT 下发散早退无收益)。`qd.static` 子树不动(static return 本就合法)。旗标贯穿:
  kernel.py `use_early_return` / func.py `early_return` / kernel_impl 装饰器。函数体文件:
  function_def_transformer.py(插桩点)、kernel_impl.py、kernel.py、func.py。
- **背景需求**:mujoco_warp 全库 kernel/func 早退 ~1500 处,前端一律拒绝——移植最大单项
  机械成本;`@qd.real_func` 的 Metal/SPIR-V 路径崩溃(A1)堵死了替代路。
- **状态**:landed(分支)+ verified — gate `probe_early_return_forms.py` 8 形态
  **metal + cpu 双 ALL PASS**(守卫式/while+return/void/静态旁路/元组/嵌套 break/
  无注解负例/跨调用 flag 重置)。改写正确性基准 = 原语义纯 Python 孪生。

## M4 — build_Return 元组注解崩溃修复
- **描述**:`ast_transformer.py` build_Return func 分支:`-> tuple[...]` 解析为多槽
  return_type 时,`return a, b` 求值成不可变 tuple,逐槽 `qd.cast` 原地赋值即
  TypeError。修复:tuple 归一为 list。
- **背景需求**:A4(mujoco-qd P0);上游 test_return.py 元组注解用例仅标
  cpu/cuda/amdgpu,同一根源。
- **状态**:landed(分支)+ verified — 带注解元组 @qd.func 单出口形式 metal 数值
  正确;早退形态由 M2 改写器承接。

## C3 — Tile16x16 matmul_ / matmul_at_ / zero_upper_
- **描述**:`simt/_tile.py` 工厂新增三方法。`matmul_(A,B,alpha)`:self += alpha·A@B;
  `matmul_at_(A,B,alpha)`:self += alpha·A@Bᵀ(转置由 lane/寄存器索引互换自然获得,
  分块分解尾更新 `W -= L_ik@L_jkᵀ` 的标配);`zero_upper_()`:cholesky_ 保留上三角,
  本方法按 lane 清上三角。外层 k 为 runtime range(寄存器压力经验沿 _trsm)。
- **背景需求**:mujoco-qd Newton 分块 Cholesky 的列 outer 公式仅 17–36 GF/s;
  tile-tile 乘法是进入真分块路径的前提。
- **状态**:landed(分支)+ verified — gate `probe_tile_matmul.py`:unit A@B ±alpha
  vs numpy 1.9e-6;blocked 2×2 分解+求解(A@Bᵀ 更新)1e-6 级,metal ALL PASS
  (tile 仅 GPU 后端,CPU 按设计报错)。

## M6 — Metal/Vulkan 静默失效选项响亮警告
- **描述**:`lang/misc.py` init() 在 arch 选定后,metal/vulkan 上请求 `debug=True` /
  `check_out_of_bound=True` 时 `_qd_core.warn`(warning_code=INERT_DEBUG_OPTS)——
  此前文档写明"never(无 in-kernel assertion 机制)"但用户零感知。
- **背景需求**:mujoco-qd P0 D1;移植期任何越界都是未定义行为而用户以为有检查。
- **状态**:landed(分支)+ verified — metal+debug=True 触发警告,普通 init 不触发。

## A5 — Metal 循环携带内存依赖被破坏(**open,最高严重度**)
- **现象(最小复现 `tests/robostudio/probe_metal_loopdep.py`)**:同一数组元素在
  runtime 循环内经 if 条件读改写(`acc[0] = acc[0] + xs[i]`),Metal 丢更新(4.0 vs
  正确 6.0;cpu 同代码正确)。判别矩阵:无条件嵌套 ✓、`+=` 增广 ✓(静默原子)、
  显式 RMW + 分支 ✗(serialize/嵌套皆错)、cpu 全对。每次迭代读到循环前的旧值 =
  load 被跨迭代缓存/外提。`qd.cast` 形初始化不改变行为(非 LICM 层)。
- **影响**:极常用模式(物理接触遍历、直方图、链表遍历)静默错结果。
- **归属定案(2026-10-11,决定性实验)**:`QD_DUMP_MSL=1` 导出的 MSL **逐行核验语义正确**
  (load 在 if 内/循环内);把该 MSL 喂给独立 ObjC harness
  (`mujoqo/_probe/a5_runner.m`,newLibraryWithSource 全新编译、单线程、struct 布局
  _Static_assert 校验)脱离 quadrants 全栈直接执行:**仍 4.0**;对照内核(无条件累加)
  同 harness = 2.0 正确。→ **Apple MSL→GPU 编译器 miscompile"循环内条件累加"模式**,
  quadrants 各层(CHI IR/SPIR-V/MSL)均无责;47 行 MSL 可直接报 Apple。per-access
  `Volatile` MemoryAccess 已试、无效(SPIRV-Cross 未落进 MSL)。
- **扰动矩阵(p1 去 else / p2 平凡条件 / p3 提载出分支 / p4 MSL 复合赋值)**:全部 4.0
  ——触发条件就是"分支内 RMW"本身,对 MSL 微形状不敏感。
- **用户级规避(即刻可用)**:条件累加一律写 `arr[i] += v`(静默原子降级,正确);
  **避免**显式 `arr[i] = arr[i] + v`。
- **状态**:open(根因=Apple 工具链;上游动作=以 a5_runner.m + MSL 报 Apple;quadrants
  侧缓解 = 评估条件 RMW 的 select 无分支化 lower,需设计评审)。与 A2 共享"Apple 编译器
  对循环携带模式"怀疑。gate 已钉死,Apple 修复或缓解落地后翻绿。

## A1/A2 — real_func SPIR-V 崩溃 / tile runtime-loop 错值(**open,机制假设已获 MSL 证据**)
- A1:最小复现 `p0_realfunc.py`(metal 0/5,cpu 4/5);`print_ir` 定界 = CHI IR 正常,
  崩在 `spirv_ir_builder.cpp query_value`(SSA 使用先于定义)→ SPIR-V 后端函数调用
  发射层。
- A2:MSL diff 定位(dev25+ wheel,QD_DUMP_MSL=1,`p0_block_chol.py` runtime 版 vs
  `/tmp` 静态展开版,m=48):runtime 版在 for 头部携带 **16 个 tile 寄存器的长 phi 链**
  (`_1990=_1991, _1993=_1994, ...`, rank-1 内循环跨迭代携带全部 tile 行寄存器),静态版
  无此结构(直线 SSA)。与 A5 同家族:**Apple 编译器对运行时循环携带值(长 phi 链 /
  循环携带内存依赖)处理有缺陷**。
- **缓解设计(已定)**:tile 存活跨循环迭代时**必须静态展开块循环**(mujoco-warp 的
  matrix_size_static 工厂模式天然满足);文档 tile.md 加限制说明 + 长期报 Apple。
- **状态**:open — A2 缓解=文档+lint(可选);A1 待 C++ 深挖。复现/gate 已固化。


---

# mujocoqd/algorithms-2 条目(2026-10-11;已合入 robostudio-main(merge b12c125c --no-ff))

> 来源需求:quadrants-需求清单 M5 的 C5 余项(基数版 sort CPU SIGBUS 根查)与 C4(API 冻结 +
> segmented sort + scratch 预算文档)。零 native 改动,全部在 python 层。gate:
> `tests/robostudio/probe_algorithms_cpu.py`(121 门,cpu)、
> `tests/robostudio/probe_algorithms_segmented.py`(cpu+metal 双跑)。

## C5 余项 — 基数版 sort CPU SIGBUS 根因 + 修复(串行 LSD 基数)
- **现象(上条线遗留)**:C5 首轮为 CPU 后端写"串行 LSD 基数 sort"时 SIGBUS,退回归并版
  (`_cpu_sort`)并把陈旧注释留在 dispatch 里("serial LSD radix sort ... scratch[0:256] as
  digit counters"),根因未查。
- **根因(两层,均已最小复现并钉进 gate)**:
  1. **主因(历史 SIGBUS)**:`qd.loop_config(serialize=True)` 只 directive **下一个**循环
     (`ast_builder.parallelize(1)` + `strictly_serialize()`,消费于下一个 for)。基数版有
     zero/histogram/scan/scatter **四个**顶层 for,只有第一个被串行化,其余三个在 CPU 线程池上
     **默认并行** → 计数排序竞态:scatter 读到半成品 scratch 基址 → `dst[t]` 越界写 → SIGBUS
     (越界恰在界内时表现为静默错值;负键 n=1000 形状 100% 复现错值)。归并版幸存的机制:while
     不消费 directive,其唯一尾随 for 恰好被串行化。已 landed 的 scan/reduce/select 单 for 形态
     天然安全。修复 = 每个运行时循环自带 directive(`_cpu_paths.py` 全部串行 func 已按此审计)。
  2. **次因(block-tier GPU 阶梯在 CPU 不可下沉,三层证据链)**:GPU 基数阶梯依赖 block-tier
     原语,CPU 后端逐层拒绝:block.sync 类 → trace 期 ValueError("not supported for arch",
     前端包成 QuadrantsCompilationError);block/subgroup 集体操作 → subgroup.group_size()=0
     断言;裸 SharedArray → codegen_llvm 把共享 alloca lower 成 **addrspace(3) ExternalLinkage
     全局**(GPU 语义),CPU JIT 无后备存储 → kernel JIT 失败("Function ... not found",
     jit_cpu.cpp:198)。三者均为响亮错误,HEAD 上无 SIGBUS 可复现(三个 expected-error 探针
     钉死)。**native 改动规格(让渡,本线不做)**:CPU 后端共享内存/屏障模型需 (a) ORC 侧为
     addrspace(3) 全局提供后备(线程池全局 arena)或 (b) codegen_llvm 在 CPU arch 把
     SharedMemoryStmt lower 为进程级 arena + 屏障原语,并补 block.*/subgroup.* 的 CPU 映射 —
     工程量大且被串行路径完全覆盖,无消费需求。
- **修复**:CPU `sort` 派发到新 `_cpu_radix_sort`(串行 LSD 基数,逐循环 serialize):twiddle →
  end_bit//8 个稳定字节趟(LSB→MSB,keys/tmp_keys ping-pong,values 随行)→ 逆 twiddle;语义与
  GPU 阶梯逐条对齐(end_bit 掩码、稳定 tie、numpy 序 NaN——按位核验、scratch[0:256] 计数器在
  sort_scratch_slots 契约内);O(4n),n=1e5 1.56ms(numpy 稳定 argsort 4.75ms);CPU 侧 n 同时
  接受 0-d ndarray(GPU 契约形)/host int/i32 标量核参。归并版删除(单实现,历史见本条)。
- **状态**:landed + verified — `probe_algorithms_cpu.py` 121 门全过(6 dtype × 12 边界 n、
  NaN 按位、end_bit 8/16/24/32、has_values=False、0-d n、64 位键、三个 expected-error 探针、
  n=1000 负键回归钉),两遍连跑稳定。

## C4② — sort_segmented(分段稳定键值排序)
- **描述**:`qd.algorithms.sort_segmented(keys, tmp_keys, values, tmp_values, segment_ids,
  tmp_seg, scratch, n, num_segments, key_dtype, has_values, end_bit, log256_max_n)` +
  `sort_segmented_scratch_slots(n, D) = n + sort_scratch_slots(n, D)`。分段 = 连续区间
  `[ids[j], ids[j+1])`,调用方给 num_segments+1 个起始偏移(末位=哨兵 n;恰为 warp
  segmented_sort_pairs / CUB 的数组形状,mujoco SAP 直接可用);空段(ids[j]==ids[j+1])合法,
  段序保持。文件:`python/quadrants/algorithms/_segmented.py`(新)、`_cpu_paths.py`(CPU 逐段
  串行)、`__init__.py`。
- **设计裁定(需求清单原设想"段号<<32|键"不可行)**:Metal 无 64 位键/原子,复合键方案在短期
  目标后端不存在。实现 = (run-rank, key) 的字典序 LSD 基数:起始 flags + select 的 scan 阶梯
  导出每元素 run-rank,字节趟先 key(次键)后 rank(主键),keys/values/rank 三数组随行 ping-
  pong;全程 32 位数字。rank 由 flags 排他扫描修正而来(excl + flag - 1——排他扫描把 run0 自己
  的起始也算进后续元素,整树旋转一位的 off-by-one 已实测钉死);tile-histogram 区上移到
  scratch[n:] 与活 rank 共存。
- **Gate**:`probe_algorithms_segmented.py [metal|cpu]`——numpy 逐段稳定参考解,键按位
  (NaN-safe)值精确:变长段、空段、单元素段、单段、n∈{1,2,257,5000}、负键/NaN、
  has_values=False、u32/i32/f32(metal)±u64/i64/f64(cpu)全过;SAP 形状吞吐 pin(128 world
  × 256 geom)按 gate-4 教训 WARN-only:cpu 0.24ms(135 Mkeys/s)、metal 0.38ms(85.6 Mkeys/s)。
- **状态**:landed + verified(cpu+metal 双跑全过;metal 跑前 `_gpu_busy.sh` 确认空闲)。

## C4① — API 冻结(qd.algorithms 稳定化)
- **描述**:`reduce_{add,min,max}` / `exclusive_scan_{add,min,max}` / `select` / `sort` /
  `sort_segmented` 去 **Experimental** 标注(docstring + 文档横幅),契约写入
  `docs/source/user_guide/algorithms.md`:签名/语义/scratch 契约冻结、后续只做加法;
  reduce_by_key_add 维持 experimental(min/max 变体未落地);parallel_sort/PrefixSumExecutor
  维持 deprecated 移除轨。后端覆盖段改写(CPU 后端已支持, Metal 32 位限制, CPU 计数三形态),
  scratch 预算公式表(B=256, cdiv 链闭式, 逐 op 精确式 + 渐近 ~N+O(N/256) u32 槽)。
- **契约要点(dtype/稳定性/in-out)**:键 dtype 集 {u32,i32,f32,u64,i64,f64}(Metal 32 位子集);
  sort/sort_segmented 稳定(LSD 字节趟构造性稳定);NaN 序 = numpy.sort(按位核验);end_bit 为 8
  的正偶倍数 ≤ 键宽(偶数趟落回 keys);sort 系 out = in-place(keys 原地,scan/select out 与 arr
  必须异 buffer);scratch 契约各 op 对应 helper,无设备端检查(graph 契约)。
- **状态**:landed(文档 + docstring;无行为变更,纯承诺固化)。

## C2 — SPIR-V serial inner-loop vectorization (SIMD-within-thread, opt-in)
- **描述**:compile_config 新旗标 `spirv_vectorize`(qd.init(spirv_vectorize=True) / env
  QD_SPIRV_VECTORIZE;已序列化进 offline-cache key)。开启后,SPIR-V 后端(Metal/Vulkan,
  physical-storage-buffer 设备)对"最内层串行 range 循环 + 可证 stride-1 ndarray 访问"在
  **发射层**重写为 [≤3 迭代对齐 prologue] + [float4 向量主循环] + [标量 remainder],逐元素
  语义与标量路径 **bitwise 一致**:循环携带归约保持顺序标量累加(危险形状走内层 runtime
  循环 order-barrier),非携带局部量按 lane 向量化、循环出口 spill lane-3(=标量"最后一次
  迭代获胜"语义)。安全规则(任一不满足即保守回退标量):语句白名单(Const/Alloca/
  LocalLoad/LocalStore/BinaryOp/UnaryOp/GlobalLoad/GlobalStore/LoopIndex/ExternalPtr/
  Shape/ArgLoad)、f32 标量 ndarray、无 if/while/原子/随机/嵌套循环/continue/cast、存储必须
  stride-1、同 arg 多访问点必须同 base(不同 arg 假设不别名,与 LLVM 后端一致)、归约累加器
  不得作为 lane 值消费、循环值不得外泄。文件:`quadrants/codegen/spirv/spirv_codegen.cpp`
  (vec_analyze / try_emit_vectorized_range_for / vec_emit_* )、`spirv_ir_builder.{h,cpp}`
  (v4f32 助手 + name-scope)、`program/compile_config.h` + `python/export_lang.cpp` +
  `analysis/offline_cache_util.cpp`(旗标贯穿)。
- **背景需求**:mujoco-qd 需求清单 C2(Metal 算力利用率 1–5% 的根因=串行内层循环无 SIMD;
  同代码 CPU(LLVM) 75 GF/s vs Metal 45.6)。调研结论见本文末"C2 性能定案"。
- **Gate**:tests/robostudio/probe_spirv_vectorize.py(gate 7/7):4 kernel 形状 × nv∈
  {24,23,5,1}(对齐 prologue/remainder、跨 arg class 不一致、tiny 全标量回退)双臂子进程
  scalar/vec 逐位 hash 对照 —— 12/12 ALL PASS。bench 全家(K1/K2/K3/K4/saxpy @NV∈
  {1,3,5,23,24,25,64})同证 bitwise。union gate 7/7 PASS(venv4 实跑)。
- **状态**:landed(分支 mujocoqd/spirv-vectorize)+ verified(bitwise 契约全绿;默认关闭)。

## R5 — host-visible ndarray 分配策略(opt-in,obs/读回免桥接税)
- **描述**:ndarray 级 opt-in 旋钮 `qd.ndarray(dtype, shape, host_visible=True)`(默认 False,
  零参路径逐字节不变)。分配路由:`Ndarray` ctor → `Program::create_ndarray(host_visible=)` →
  `ProgramImpl::allocate_memory_on_device(host_visible=)` → GfxProgramImpl 置 `host_read=true`
  → Metal 既有规则(host_read||host_write ⇒ MTLStorageModeShared,R2 定案;**metal 侧零改动**)
  把该 ndarray 的 MTLBuffer 直接分到 Shared。读回快路(Python 侧仅 flag 命中时生效):
  scalar + matrix/vector(恒 AOS,numpy total_shape C-contig == 物理缓冲)的
  `to_numpy`/`from_numpy` 跳过桥 kernel(`ndarray_to_ext_arr`/`ext_arr_to_ndarray`),走新
  C++ 入口 `Program::ndarray_copy_to_host/from_host` = `synchronize()`(排干;与 R2 直读同一
  "waitUntilCompleted 后 contents 可见"契约;写向即"host 写先于 enqueue 对后续 cmdbuf 可见",
  与镜像 record 期 memcpy 同前提)+ map_range + memcpy + unmap(**DeviceAllocation 重载**
  ——DevicePtr 重载在 CpuDevice 是 QD_NOT_IMPLEMENTED);to_numpy 输出 np.empty(全量覆写)。
  消掉的固定税:桥 kernel Python 侧 launch 机器(R2/R3 报告残差解剖 0.131ms/次)+ ext 镜像
  分配/回收 + np.zeros 页分配。旋钮选 ndarray 级而非 init 级:host-visibility 是 buffer 属性
  不是 program 属性,obs 只需要输出/读回缓冲;init 旗标会把全部 ndarray+field 内存翻成 Shared
  = 全局行为变更,违背 opt-in。**footprint:+484/−27 行,16 文件**:native
  `quadrants/program/{ndarray,program,program_impl}.{h,cpp}`(+101)、
  `quadrants/runtime/program_impls/{gfx/gfx_program,llvm/llvm_program}.{h,cpp}`(+22)、
  `quadrants/python/export_lang.cpp`(+15);python `lang/{impl,_ndarray,matrix}.py`(+60);
  gate/bench `tests/robostudio/gate_r5_host_visible.py`(新,190)、
  `bench_r5_host_visible.py`(新,79)、runner 8/8(+21);ledger 本条。
- **背景需求**:quadrants-需求清单 M5-Metal 第 3 项 R5(readback 物理地板;qd-render-poc
  readback_r2r3_report:1KB to_numpy 0.163ms 中 0.131ms 是桥 kernel launch 机器,R2/R3 拿不掉
  ——除非源缓冲本身 host-mappable,即本条)。mjlab obs 集成路径必经优化。
- **量化(gate/bench 内同进程交错 A/B,min-of-k × 4 轮,提交 621f40e2 wheel,
  `_gpu_busy.sh` 空闲;当日机器态把 default 臂抬高 ~5×,venv4 dev55 基线同日同值 0.78-0.84ms
  交叉证实 = 机器态而非回归;比例钉值免疫方差)**:
  | 尺寸 | to_numpy default→hv | from_numpy default→hv |
  |---|---|---|
  | 1KB | 0.80-0.86ms → **0.013-0.028ms**(~30-60×) | 0.84-0.92ms → 0.012-0.025ms |
  | 64KB | 0.88ms → **0.015-0.043ms** | 0.87ms → 0.015-0.041ms |
  | 1MB | 0.72-0.89ms → 0.026-0.146ms | 0.83-0.87ms → 0.025-0.118ms |
  | 64MB | 5.8-7.9ms → **1.44-1.75ms**(不回退,~4-5×) | 3.9-6.4ms → 1.43-1.60ms |
  分解(64KB hv):np.empty 0.3µs + `ndarray_copy_to_host` 15-20µs(其中空闲排干
  13.8-19µs,64KB memcpy 亚 µs)= 全链 ~15-20µs —— 对 R2/R3 报告口径的 0.131ms 税
  **~7-9×**,任务书"数十 µs 以下"达成;64MB hv ~1.45ms ≈ 纯 memcpy 物理地板
  (~45GB/s),含 np.empty 免页分配收益。
- **Gate**:`tests/robostudio/gate_r5_host_visible.py`(union gate 8/8):字节等价 sweep
  hv-vs-default-vs-numpy(i32/f32/u8 × 1KB/1MB/64MB,按位)、obs 模式排序验证(kernel 写后
  快路读、快路写后 kernel 读)、matrix/vector 形态、旋钮内省(default `_qd_host_visible`=False
  且 C++ 缓冲非 host-mappable;grad 缓冲保持 default)、metal 相对 perf 钉值(小缓冲 ≤x0.5、
  64MB ≤x1.2 —— 按 gate-4 教训只用进程内相对钉,不设绝对 ms 钉);cpu 臂亦全过(LLVM 忽略
  旗标,快路在 CPU 上同样成立,unmap 走 DeviceAllocation 重载)。
- **状态**:landed(分支 `mujocoqd/r5-host-visible`,621f40e2+c0822480+f91b28e2)+
  verified — `UNION GATE: 8/8 PASS`(venv5 实跑;venv5=dev57+g621f40e22 R5 wheel)。
  范围外(记录):to_torch/from_torch 与 DLPack 未接快路(numpy 按验收;host_visible 缓冲上
  `to_dlpack` 天然零拷贝,mjlab torch 路径如需 memcpy 快路为 +6 行后补);field(qd.tensor)
  路径未动——field 走 snode tree 分配,obs 集成如用 field 需另立项。

## C2 性能定案(2026-10-11,M4 实测)— 2× 验收未达成,瓶颈=DRAM 字带,非请求吞吐
- **数据**(同二进制旗标 A/B,三/四轮交错取中位;当日机器态抖动实测 ±4×,同代码 scalar
  自身可在 200µs↔800µs 间摆动):nworld=4096:K1 scalar 13.6/14.8/14.1 vs vec
  14.6/14.3/16.5 GF/s;K2 7.4/8.7/7.0 vs 8.9/7.4/9.8 —— **parity(≈1.0–1.2×)**。
- **瓶颈定位**(microbench 链):dot@65536×24:dot1(2 load/el)194µs ≈ dot4(同 load、4×
  flops、4 独立累加器)198µs ⇒ **算力/ILP 免费非瓶颈**;dotL(1 load/el,半字节)99µs=
  一半时间 ⇒ **严格 load 字宽瓶颈,~63 GB/s ≈ 同进程 copy 65 GB/s(DRAM 墙)**。即大尺寸
  下标量发射已打满有效 DRAM 带宽,向量化减少的是"请求数"而非"字节数",无收益空间;小尺寸
  则被 dispatch(µs 级 launch 税)支配。volatile 载入(A5 缓解)经对照非瓶颈
  (QD_NO_VOLATILE_LOADS 诊断临时开关,已撤销)。col-major(跨线程连续)布局同速 ⇒ 跨线程
  scatter 非瓶颈。这从 codegen 层独立复证了 §4.1 结论(Mac 统一内存带宽 CPU 可吃满、SIMT
  形状不可达峰值)。
- **含义**:bitwise(保持 FP 求值顺序)约束下,within-thread 向量化在带宽墙形状上无 2× 可
  拿;突破 2× 需要重结合(4 独立累加器,实证可行但**违反 bitwise 验收**)或换并行分解
  (warp 级/分块),属新立项。旗标默认 OFF,A/B 工具链与 bitwise 机制保留。
- **状态**:open(性能线)/landed(机制线)—— 如需重开:先在非带宽墙形状(如 nworld 小、
  L2 驻留、算力密集)找收益点,或立项"顺序重结合 opt-in 旗标"(须用户裁定量语义)。

## 探针线(2026-10-11,纯 python 探针,零 native 改动)— ①重结合收益量化 ②subgroup Metal 可用性
- **背景**:C2 性能定案的重开路线①(顺序重结合 opt-in 旗标,量语义待裁)+ M5 第 4 项
  (subgroup 归约)的前置可用性验证。分支 `mujocoqd/reassoc-subgroup-probe`
  (worktree `/Users/mini0/Code/quadrants-worktrees/reassoc-subgroup-probe`,基线
  43e39634,venv4 dev55 native overlay,import 头 commit=68bd7965,
  `probe_algorithms_cpu.py` ALL PASS 后开工)。
- **①`tests/robostudio/probe_reassociate_payoff.py`**:纯源码层 A/B(同 kernel 的
  顺序累加版 vs 手写重结合版:4 部分和步距 4 + 树形合并 + 标量尾,不改编译器),
  M4 Metal,8 形状 × 5 计时窗口。计时纪律:GPU 空闲 + load 门(统一内存上重 CPU 负载
  挤占带宽、且对 rea 臂 MLP 更敏感 → 系统性低估)、ABA 锚定轮(每轮 seq×2 夹 rea,
  两针差 >25% 弃轮;实测同 kernel 双峰 ~3.8× 模式切换)、perf 尺度两臂一致性门
  (防「快是因为算错」)、nw=1 vs numpy f64 精度表。
- **①结论(旗标语义建议)**:(a) **值得开**:欠占用(per-world 线程)+ 长链(m≥48)
  的 solver 形状 —— symmatvec(Gram 形,下三角)flops:load 最高、收益最稳
  ~2.1×(4 窗口 1.42–2.21×);matvec m=48(per-world)2.2–2.8×;chol_dot m=64
  ~1.5–1.6×。(b) **不值得开**:流式大 nworld(对照组 1.00±0.01×,复证 DRAM 墙)、
  短链 chol m=32/48(0.84–1.34×)、满占用 (w,i) 并行消融(跨窗口双峰 0.26–3.91×
  且两臂互换 = 计时/编译态伪迹,无方向信号;TLP 已藏延迟时 ILP 无空间,多累加器
  寄存器压力有害)。(c) **数值代价**:重结合版误差 ≤ 顺序版(6/8 形状),worst
  1.8e-6 rel vs f64,1e-5 物理门下余量 6×;逐位差异 17–100% 输出(契约 = 非 bitwise)。
  (d) **codegen 落点**:C2 vec 机制的 order-barrier 反面 —— 在
  `spirv_codegen.cpp` 的 vec_analyze/try_emit_vectorized_range_for 处,对循环携带标量
  归约累加器,旗标开时以 k=4 部分和 + 步距 4 展开(复用 prologue/main/remainder 骨架,
  去掉 order-barrier 内层顺序环)替代顺序累加,出口树形合并;新 compile_config 旗标
  `reassociate`(默认 OFF,进 offline-cache key);安全白名单沿用 C2 语句集;
  **k=4 不取 k=8**(m=48 实测 k=8 一致差于 k=4:部分和链更短 + 寄存器压力)。
- **②`tests/robostudio/probe_subgroup_metal.py`**:逐 API 最小 Metal kernel 正确性
  (线性派发 lane=w%32,宿主串行参考)+ 约束计数 perf pin。
- **②结论:subgroup 在 Metal 全栈可用**(24/24 PASS,零 RAISED,无 handback 项):
  invocation_id/elect/broadcast/broadcast_first/shuffle/shuffle_xor/shuffle_up/
  shuffle_down/ballot_first_n/**ballot(u64 形也过,MSL 2.3 int64)**/all_true/
  any_true/all_equal/reduce_add/reduce_all_add/reduce_min/reduce_max/inclusive_add/
  exclusive_add/segmented_reduce_add/lanemask_lt/**sync + SharedArray**
  (loop_config(block_dim=32) 每 kernel 显式标注)/mem_fence/bitonic_sort_kv 全部数值
  对照通过。perf pin:atomic 链(每元素一次原子)vs 局部和 + reduce_add + elect 单原子
  = **6.2–7.4×**(两臂同 32-lanes/world 映射,仅原子流量不同)。前端人机工学坑(记录,
  非缺陷):shuffle/broadcast/shuffle_up/down 的 index/offset 实参必须 u32 类型
  (python int 报 QuadrantsTypeError: expected u32 for argument #2;探针内以 qd.u32()
  显式转换绕过)。**数值语义单列**:subgroup.reduce = 并行树归约 = 重结合同桶
  (bitwise 不保序,精度类 ~1e-7 rel 不变),若进生产路径其量语义与①旗标同一裁决。
- **状态**:probe 全部跑通(exit 0);两文件为探针交付物,不落任何生产路径;旗标语义
  裁决与 subgroup 归约立项待用户决策。

## REA — `reassociate` 旗标:C2 向量主循环归约重结合(opt-in,默认 OFF;mujocoqd/reassociate-flag)
- **描述**:compile_config 新旗标 `reassociate`(qd.init(reassociate=True);已序列化进
  offline-cache key)。**语义 = 蕴含 `spirv_vectorize`**(二选一裁决:其机制只存在于 C2
  向量路径内,独立旗标单独打开时静默无效是 footgun,故 flag alone 即启用 C2)。开启后,
  C2 float4 向量主循环中**资格合格**的循环携带标量归约累加器从「逐迭代 alloca 回写」改为
  **k=4 部分和(步距 4)+ 循环头 phi 携带**:acc[0] 以循环前累加器值(prologue 结果)播种,
  acc[1..3] 以运算单位元播种(add/sub: +0.0,mul: 1.0);主循环内 alloca 往返消失,循环
  出口把 sum-0 存回 alloca,**标量 remainder 继续挂在 sum-0 上**(探针同构:prologue →
  lane0 → 尾部 → 合并),remainder 之后树形合并 `(t+s1)+(s2+s3)`(mul 乘法合形)。
  **order-barrier 内层 lane 环对合格归约拆除**。资格规则(vec_rea_qualifies):add/mul
  任意侧、sub 且累加器在左(`x - acc` 逐元素变号不是部分和分解 → 保留 barrier);累加
  操作数 DAG 必须可被 scalarize_static 逐 lane 复刻(add/sub/mul/div/truediv + neg/abs +
  env 叶子),其余维持既有 fast / barrier 路径。**OFF 路径代码不变**:全部新发射都在
  vec_reassociate_enabled() / 空 reassoc 向量之后(硬门 G1 逐位复验)。
- **背景需求**:探针线①(codegen 落点裁决):欠占用 + 长链 solver 形状 1.5–2.8×,k=4 优于
  k=8;C2 性能定案的重开路线①。
- **Gate**:`tests/robostudio/probe_reassociate_flag.py`(union runner 新 gate 9/9):
  三子进程臂 scalar/vec/rea(S-guard 单 init 一进程一臂)+ 进程内 perf 子进程。硬门:
  **G1** OFF 逐位不变(scalar==vec 全部 12+3 输出 hash,含 C2 门全家 nv∈{24,23,5,1});
  **G2** rea≠vec(重结合确实发生,nv∈{24,23} + payoff 形状);**G3** k_elementwise 三臂
  全等(旗标不碰未合格循环);**G4** rea≠scalar(旗标单开即含 C2,蕴含语义);**G5**
  rea 臂 vs numpy f64 < 1e-5(chol 1.47e-07 / matvec 5.71e-07 / sym 1.74e-06,chol 用探针
  的正定 PSD 数据配方——tril(normal) 对消数据会把这些积和的 rel 指标炸到 1.4e-4@ulp2031,
  PSD 数据 1.5e-7@ulp2,是数据几何不是 codegen)。**G6 WARN-only** 进程内交错 perf:
  ON 代码生成(plain 源)vs 同进程顺序标量基线(同源 + 内层 `loop_config(serialize=True)`,
  vec_analyze 以 special-loop-flags 拒绝 → 纯标量发射),时钟热身后 8 对 ~30ms 深突发、
  交替顺序、逐臂取 min(机器双峰态只加时,min 收敛到真实成本;median 并报)。两次被否决
  的估计器:跨进程 ABA 子进程轮(模式粘进程,比值跨运行变号)、短突发 anchor 弃轮(接受集
  选择偏差,同病)。
- **数字(venv6,本分支 9a55c585 树的 wheel;GPU 空闲;深突发 min;三连跑 A 臂 min 稳定
  ±6%)**:
  | 形状 | ON(rea)µs | 顺序标量 µs | S/A min | S/A median |
  |---|---|---|---|---|
  | matvec m=48 nw=1024 niter=8 | 1263 | 5155 | **4.08×** | 1.76× |
  | symmatvec m=64 nw=512 niter=8 | 966 | 4701 | **4.87×** | 1.47× |
  | chol_dot m=64 nw=1024 niter=2 | 16190 | 19008 | 1.17× | 1.27× |
  G6 钉值(matvec/sym ≥1.4×,min 口径)三连跑全过(matvec 3.50–4.89×,sym 1.81–4.87×;
  S 臂 min 有模式方差 1.7–6.0ms,median 口径保守 1.4–2.1×)。chol m=64 1.2× 量级(长
  链 + 满载,与探针 1.5× 同向)。跨进程 vec 臂(C2-bitwise)µs 仅作上下文:matvec
  1263(rea)vs 1229–4163(vec,模式漂移);sym 966 vs 908-1324。**与探针数字的关系**:
  探针 2.2–2.8× 是 source-rea vs source-seq(双方标量载入);本旗标 ON 在 C2 float4 主
  循环之上,vs 顺序标量基线 min 口径 3.5–4.9×、median 口径 1.4–2.1×,同向同量级。
- **附带发现(均为 HEAD 既有,非本线引入;dev59 wheel 复现一致,已量化,未修——超出
  minimal 改面,待立项)**:① 内层向量化循环体引用任务级标量(kernel arg)时,归约 DAG
  走 barrier 路径或 VectorCompute 时 emitter env 无该 argload 条目 →
  `unordered_map::at` 崩(argload 被前端上提到任务级;vec_analyze 有时以
  "argload used as value" 拒绝有时放行,放行即崩);② 向量化循环体内的 coeff-0
  (循环不变)ndarray 载入:向量 walk 注册其名字后,remainder walk 重复注册 →
  `register_value: already exists` 崩;③ C2 门的 k_elementwise 内层循环实际从未向量化
  (argload 作值被拒),其 hash 恒为标量路径输出(C2 门 bitwise 结论不受影响,记录在案)。
- **footprint**:native +185/−6 行 5 文件:`codegen/spirv/spirv_codegen.cpp`(+143,含
  vec_rea_qualifies / rea 发射 / 注释)、`codegen/spirv/detail/spirv_codegen.h`(+20,
  VecReassoc + 两个声明)、`program/compile_config.h`(+13)、`python/export_lang.cpp`
  (+5)、`analysis/offline_cache_util.cpp`(+1);gate `tests/robostudio/
  probe_reassociate_flag.py`(新,499 行)+ union runner gate 9(+16)。无 metal runtime
  改动,无前端改动。
- **状态**:landed(分支 `mujocoqd/reassociate-flag`,未 merge)+ verified(硬门 5/5 +
  union gate 9/9 + test_return 46/46;默认 OFF,OFF 路径逐位不变由 G1 复验)。

---

# mujocoqd/subgroup-cpu-support 条目(2026-10-11;分支 `mujocoqd/subgroup-cpu-support`,design-first 草稿 → 实现)

> 来源需求:用户指令"整体支持 subgroup"。Metal 侧已由探针线②证实全栈可用(24/24,零
> handback,原子替换 pin 6.2–7.4×);缺口在 CPU 侧——三种形态全报错:block.sync →
> trace 期 ValueError("not supported for arch");subgroup 集体操作 →
> `Program::subgroup_size()`=0 断言;裸 SharedArray → `codegen_llvm.cpp
> TaskCodeGenLLVM::visit(AllocaStmt)`(is_shared 分支)把共享 alloca lower 成
> addrspace(3) ExternalLinkage 全局,CPU JIT 无后备存储 → kernel JIT
> 失败("Function not found",jit_cpu.cpp lookup)。三个 expected-error 探针钉在
> `probe_algorithms_cpu.py`。本线按 C5 余项里的 native 让渡规格(a)/(b) 落 CPU 侧语义。

## 语义设计(定案)

**subgroup 层 = 虚拟组语义(group_size=1)**:CPU 上每线程 = 自己的 subgroup。
`Program::subgroup_size()` 对 cpu 返回 1(native,一处);`invocation_id()≡0`、
`elect()≡1`、shuffle/broadcast 全系 = identity、`ballot/ballot_first_n` = bit0、
reduce 系 = 自身、inclusive = 自身、exclusive = 单位元、segmented = 自身、
`all_true/any_true/all_equal` = 谓词自身/恒 1、`bitonic_sort_kv` = 恒等
(`log2_size=0` 在 reductions.py/sorting.py 本来就是文档化的退化情形,静态展开空环)。
`subgroup.sync()` = no-op(单 lane 子组屏障平凡收敛);`subgroup.mem_fence()` =
真实 `std::atomic_thread_fence(seq_cst)`(保守超集)。**逐位确定**。
选择理由:候选 (i) 每 thread 独立 subgroup 是唯一零成本可移植且诚实的方案;
候选 (ii) 虚拟 32-lane 串行模拟需要协程化 straight-line 代码,不成立且重。

**block 层 = 真并发虚拟 block(每 virtual thread 一条真实线程)**:GPU 语义
"block_dim 个虚拟线程同时执行循环体,SharedArray 是 block 窗口,barrier 收敛"在
CPU 上按字面兑现——`cpu_block_parallel_for`(runtime.cpp 新增)把 [begin,end) 按
block_dim 切 block,**逐个** block 生成 `n_live` 条 std::thread(vt=0..n_live-1,
vt 即 `block.thread_idx()`,iteration = block 基址 + vt),barrier =
mutex+condvar 代际屏障(经 `RuntimeContext::block_barrier` 交付),block 之间
join(SharedArray 窗口不跨 block 交叠,GPU 一致)。CUDA 侧模型逐条对齐:
`block.sync` 收敛同 block 全部虚拟线程且带内存可见性(屏障自带 happens-before);
SharedArray 跨虚拟线程可见。block 级集体(`block.reduce_* / *_scan_* /
sync_*_nonzero / radix_rank`)**零改动**——block.py 的共享内存骨架在
`SUBGROUP_SIZE=1`(`NUM_SUBGROUPS=block_dim`,每线程 publish 自槽 + 屏障 +
thread-0 顺序折叠)下自动正确。**响亮拒绝**:非 range-for 任务(serial/struct_for)
上的 block-tier 操作、`rand` 进 block-tier 循环(rand 状态按 cpu_thread_id 索引,
vt 会越界)、`block_dim∉[1,1024]`、`block.global_thread_idx()`(维持 ValueError)
——全部编译期/trace 期报错,无静默错值路径。
选择理由:barrier 分离的写读模式(write-all → sync → read)在"顺序逐迭代执行"
模型下**本质上不可实现**(thread 0 读时 thread 255 尚未执行),顺序模拟必然静默
错值;真线程 + 真屏障是唯一字面正确的兑现。CPU 侧不要求性能(每 block spawn
线程),换语义诚实。FP 语义:block 级 reduce/scan = thread-0 顺序折叠,同 launch
内逐位确定(确定性断言进 gate);vs GPU 树形归约 = 既有"容差桶"裁定(1e-7 rel),
整数路径逐位一致。

**实现落点(C5 规格 (b) 变体:共享 alloca → addrspace(0) 零初始化模块全局)**:
`visit(AllocaStmt)` is_shared 分支对 cpu arch 生成 PrivateLinkage + 零初始化的
addrspace(0) 全局(JIT 可链接;bls_buffer 同款先例)——比规格原文的"arena 指针 +
运行时 arena 分配"更小:因 block 串行推进,每 (task, alloca) 一个全局即等价于
per-block 窗口,免掉 RuntimeContext arena 指针、offset 累计与 per-block malloc。
`make_cpu_multithreaded_range_for` 对 block-tier 任务跳过(保住用户
[begin,end) 与 block_dim);`TaskCodeGenCPU` 预扫 body(is_shared alloca /
cpu barrier internal call / RandStmt / AdStackAllocaStmt)后派发
`cpu_block_parallel_for`,adstack `static_num_threads` 按 max(num_cpu_threads,
block_dim) 放大(vt 作 cpu_thread_id 时槽位不越界)。python 侧:`subgroup.py`
CPU 分支(映射表)+ `block.py` CPU 分支(`cpu_block_barrier` /
`cpu_block_mem_fence` / `linear_thread_idx` / `sync_*_nonzero` 走既有 emulated
路径)。新 internal op:`cpu_block_barrier`、`cpu_block_mem_fence`(均
PLAIN_OP + runtime context)。**无新 compile_config 旗标**(后端完备性,非 opt-in)。

## 资格矩阵(CPU 兑现方式 / 响亮拒绝)

| 操作面 | CPU 兑现 |
|---|---|
| subgroup 数据移动/投票/reduce/scan/segmented/sort/lanemask | 全部兑现(group_size=1 恒等语义,逐位确定) |
| subgroup.sync / mem_fence | no-op / seq_cst fence |
| block.sync / mem_fence / thread_idx | 真屏障(代际 CV)/ seq_cst fence / vt |
| block.reduce_* / *_scan_* / sync_*_nonzero / radix_rank | block.py 骨架零改动自动正确(SUBGROUP_SIZE=1) |
| SharedArray(配合 loop_config(block_dim) 的 range-for) | addrspace(0) 零初始化全局,block 窗口语义 |
| block.global_thread_idx | 响亮 ValueError(维持) |
| rand / 非 range-for 任务上的 block-tier / block_dim>1024 | 响亮编译期错误 |

## 实现落点(与草稿的差异,已按实现修订)
1. **SharedArray 后备 = addrspace(0) 零初始化 PrivateLinkage 模块全局**(草稿写的是
   "运行时 arena + RuntimeContext arena 指针",实现更小):`visit(AllocaStmt)` is_shared
   分支在 cpu arch 改发 defined global(JIT 从 .bss 链接;bls_buffer 同款先例),
   block 串行推进 ⇒ 每 (task, alloca) 一个全局即等价 per-block 窗口,免掉 arena
   malloc/offset 累计/新 RuntimeContext 字段(RuntimeContext 只加了 block_barrier 一个
   字段,给屏障交付用)。零初始化(GPU 是未初始化)——严格更 defined,对
   barrier 分离的 produce/consume kernel 不可区分。
2. **共享 alloca 被 LLVM 消除的平凡 kernel 不进 block-tier 派发**:uses_block_tier_ops
   以 is_shared alloca **或** barrier internal call 判定;无屏障且访问平凡的 kernel 其
   共享 alloca 会被 LLVM 直接消除(无跨线程契约的 kernel 本就无此契约,安全)。
3. **守卫必须放主线程**:QD_ERROR = Logger::error = throw std::string(nanobind translator
   的 std::string 分支接住 → RuntimeError)。per-task codegen 跑在 ParallelExecutor
   worker 上,throw 无 handler → abort;故权威校验放 `compile_to_offloads` 尾部
   (主线程,offload 之后 OffloadedStmt 已存在),codegen_cpu 留同名 last-resort 守卫。
   过程中实测:worker 上 throw std::string / QuadrantsIrError 均 abort(后者 nanobind
   无 std::exception 兜底翻译,已试验后回退 translator 改动,维持既有机制)。
4. **block_dim 契约**:CPU 侧 ASTBuilder::block_dim 既有 power-of-two 断言 ⇒ 有效域
   {1,2,...,1024};不显式 loop_config 时 default_cpu_block_dim=32(GPU 缺省 128,
   同策略,缺省块照跑)。
5. **subgroup python 映射的实现细节**:常数须走 `make_constant_expr`/`ops.cast`
   (i32(0) 在 plain-function trace 语境不可调用);ballot_first_n/ballot → bit0;
   @func 层(all_true_tiled/reductions/sorting butterfly)零改动,log2=0 空展开自动正确。

## 门禁(全部跑通,2026-10-12,venv7 = /Users/mini0/Code/mujoqo/_probe.venv7)
- **新探针 `tests/robostudio/probe_subgroup_cpu.py`(565 行,37 门)ALL PASS**:
  host group_size==1/log2==0;invocation_id/elect/shuffle×3/broadcast×2 恒等(逐位);
  ballot/ballot_first_n bit0;all_true/any_true(=pred)/all_equal(=1);reduce×4
  (=自身)/inclusive(=自身)/exclusive(=0)/segmented(=自身);lanemask/bitonic 恒等;
  block-tier:sync+SharedArray 置换读(bitwise)、8×block_dim=32 窗口隔离(bitwise)、
  缺省 block_dim=32 无 loop_config 交换(bitwise)、block.reduce_all_add(i32)=256、
  inclusive/exclusive min/max 前缀(bitwise)、sync_count_nonzero=64、f32 block reduce
  launch-twice 逐位确定 + vs f64 ref rel 3.23e-09(1e-7 桶);subgroup.sync no-op 与
  block.sync 混用;4 个响亮拒绝(serial 任务/rand/block_dim=2048/global_thread_idx)。
- **`probe_algorithms_cpu.py` 120 门 ALL PASS**(两遍连跑稳定):三个 staircase
  expected-error 钉翻绿为正值门——staircase_block_sync(置换读 bitwise)、
  staircase_block_reduce(=256)、staircase_shared_array(cross-thread 读 bitwise)。
- **union runner 第 10 门**(`run_robostudio_gates.sh`,label /10):metal 臂 =
  probe_subgroup_metal 24/24 ALL PASS(既有钉,python 改动 arch 门控,GPU 路径零变化);
  cpu 臂 = probe_subgroup_cpu + probe_algorithms_cpu。**UNION 10/10 PASS**(venv7,
  GPU 空闲,2026-10-12 00:48)。
- **`test_return.py` 46 passed**(venv7,pytest+xdist)。

## footprint
native +402/−13 行 11 文件:`runtime/llvm/runtime_module/runtime.cpp`(+184,
cpu_block_parallel_for/代际 CV 屏障/cpu_block_barrier/cpu_block_mem_fence +
STRUCT_FIELD(block_barrier))、`codegen/cpu/codegen_cpu.cpp`(+69 含守卫与
create_offload_block_range_for 派发、adstack static_num_threads 放大)、
`transforms/compile_to_offloads.cpp`(+36,主线程权威校验)、`ir/statements.{h,cpp}`
(+71,uses_block_tier_ops/uses_rand_stmt 预扫)、`codegen/llvm/codegen_llvm.cpp`
(+33,SharedArray cpu 分支 = addrspace(0) 零初始化全局)、`program/program.cpp`
(+12,subgroup_size()=1)、`program/context.h`(+9,block_barrier 字段)、
`transforms/make_cpu_multithreaded_range_for.cpp`(+11,block-tier 跳过)、
`inc/internal_ops.inc.h` + `ir/type_system.cpp`(+12,cpu_block_barrier /
cpu_block_mem_fence 注册)。python +120/−58 两文件(simt/subgroup.py CPU 恒等映射 +
simt/block.py CPU 派发)。门禁:probe_subgroup_cpu.py(新,565 行)+
probe_algorithms_cpu.py 翻绿 + union runner gate 10(+34)。文档:
user_guide/subgroup.md(CPU 列 + CPU backend semantics + 数值语义节,+105/−37)、
block.md(CPU 列 + virtual blocks 节,+41/−4)。**无 metal/cuda/amdgpu runtime 与
codegen 改动**(arch 全部门控),无新 compile_config 旗标,offline-cache key 无新维度
(IR 形状差异自然进 key)。

## 数字
- 虚拟组语义验收 = **逐位一致**(全部门禁,无一处数值 FAIL);block-tier 整数门逐位;
  f32 block reduce(256 虚拟线程,顺序折叠)vs numpy f64 ref = rel 3.23e-09,
  同 kernel 两次 launch 输出逐位相同(确定性)。f32 参考形状 (i*0.37+0.11) 之和。
- 原子替换 pin(Metal,探针线②)不受影响:union gate 10 metal 臂原值复现 ALL PASS。
- wheel:`dist/quadrants-1.3.4.dev71+g902047d30.d20261011-cp312-cp312-macosx_13_0_arm64.whl`
  → `/Users/mini0/Code/mujoqo/_probe.venv7`(python3.12,tuna 依赖)。

## 边界与响亮拒绝矩阵(实现核对版)
| 形态 | CPU 行为 |
|---|---|
| subgroup 全 API | 兑现(group_size=1 恒等,逐位确定) |
| block.sync/mem_fence/thread_idx + block 全部集体 + radix_rank | 兑现(真并发虚拟 block) |
| SharedArray(带/不带显式 loop_config) | 兑现(缺省 block_dim=32) |
| block-tier on serial/非 range-for 任务 | 响亮编译期错误 |
| qd.random in block-tier | 响亮编译期错误 |
| block_dim 非幂等或 >1024 | 响亮编译期错误 |
| block.global_thread_idx | 响亮 ValueError(维持) |

- **状态**:landed + verified(分支 `mujocoqd/subgroup-cpu-support`,未 merge;
  probe_subgroup_cpu 37/37 + probe_algorithms_cpu 120/120 两遍稳定 + union 10/10 +
  test_return 46/46)。

---

# mujocoqd/qd-copy-api 条目(2026-10-12;**已合入 robostudio-main**,merge 2ce5a850;union 11/11 @ venv4/dev82)

**qd.copy(dst, src, count=None) 原生 API**(impl.py,`from quadrants.lang import *` 导出):
warp `wp.copy` 的 python-scope 对应物。ndarray 源走 `copy_from`;host 数据(numpy/list)按
dst 元素 dtype cast(to_numpy_type 走 VectorType.dtype 解包)+ reshape 到 dst 的 numpy 形状
(向量/矩阵 dtype 自动附加元素维);`count` 填充前导元素(from_numpy 整数组语义 → host
readback 合并实现,代价已文档化)。返回 dst 可链式。动机:mujoqo 移植在 wp.copy 调用点上的
wp_copy 别名 shim 层得以删除,调用点 `qd.copy(dst, src)` 与上游形状一致。
**Gate**:qdcopy smoke(numpy/list/ndarray/vec3 元素维/count;cpu+metal)ALL PASS + union
11/11。footprint:impl.py +55/−2,纯 python 增量。

# mujocoqd/c2-emitter-fixes 条目(2026-10-12;分支 `mujocoqd/c2-emitter-fixes`,基于 8ebf8620)

> 修复 REA 条目"附带发现"里的三个 HEAD 既有 C2 emitter/analyze 缺陷(dev59/dev71 wheel
> 均复现,非任何在飞线引入)。约束:旗标 OFF 路径逐位不变(全部改动位于
> vec_analyze / vec_emit_vector_loop / try_emit 的 ON-only 路径);每缺陷独立 commit +
> 独立 gate 臂。

## ① arg-in-DAG env miss(崩 → 正确向量发射,commit 7880d68f)
- **根因(代码定位 `spirv_codegen.cpp` vec_emit_vector_loop)**:向量化循环体的 DAG 引用
  **体外**语句时(任务级标量 kernel-arg load;或被 `cache_loop_invariant_global_vars`
  上提的循环不变 global load 及其上的纯算术),emitter env 无条目 →
  `env.at()` 抛 `unordered_map::at` / scalarize 叶子 `QD_ASSERT(env.end())`(release=UB)。
  vec_analyze 一直**放行**这类循环:其 "argload used as value" 规则只检查**体内**
  ArgLoadStmt,而 hoister 早已把标量 argload 上提到任务级 → 崩溃即放行的后果。
- **"时拒时放"定性**:analyze 是 IR 的纯函数,同一 kernel 形态判定确定;表象不确定来自
  两条规则的覆盖差——还原循环(体内无 global store)放行后崩;含体内 store 的循环被
  缺陷③的 dtype 误拒(拒绝理由还打印成 "non-f32 memory access")。③修复后,arg-in-DAG
  的准入只剩真实规则,不再有"撞上另一条误拒"的表象。
- **修法**:`env_lookup()` 三级解析(env → 按 raw_name 查任务级已注册值 → 兜底现发射并
  记忆化)。体外值按 SSA def-before-use 必为循环不变,复用发射值精确且与标量路径逐位一致;
  重发射会撞 register_value 重复守卫,故走名字复用。
- **连带根修(同 commit)**:还原 fast 路径资格收紧——累加操作数 DAG 的叶子必须是
  **体内 stride-1** GlobalLoad:循环不变 uniform 叶子(hoisted load 或体内 coeff-0 load)
  会被 Apple MSL fast-math 把四条 lane 积和**析因重排**成 `(x0+x1+x2+x3)*c`(实测
  `s += x[j]*c` 漂 ~2 ulp,dev notes §7.2 "acc + x*c 常数因子形态"的 uniform 叶子特化),
  此类还原回落 order-barrier lane 环(循环携带 → 程序序 → 逐位)。纯 coeff-1 load DAG 的
  fast 路径不变(C2 门 bitwise 复验)。
- **复现臂(G1)**:`k_red_arg`(argload 入还原 DAG)/ `k_red_extemp`(hoisted 不变 temp)/
  `k_red_c0hoisted`(hoisted coeff-0 load 入还原 DAG)。修复前:vec 臂
  `IndexError: unordered_map::at: key not found`(三形态全部);修复后:三形态 bitwise
  scalar==vec + 无 IR-builder 错误。

## ③ store 循环从未向量化 / k_elementwise(commit 2ebc88d4)
- **根因(`vec_analyze` memory-group dtype 检查)**:dtype 取自 `mem->element_type()`,
  而 `GlobalStoreStmt` 不产值、`ret_type` 恒为 unknown → **所有含体内 global store 的
  内层循环**被误拒("non-f32 memory access")。还原循环(体内无 store)向量化了,
  elementwise 循环从未向量化;C2 门 k_elementwise bitwise 锚通过的原因是两臂都走纯标量。
  **REA 条目对 ③ 的"argload 作值被拒"归因不准**:体内 argload 已被 hoister 上提,
  该规则对 k_elementwise 从未触发,真凶是 dtype 误拒(确定性拒绝,非不确定性)。
- **修法**:dtype 改取 ext-ptr pointee(load/store 通载)。分析单侧改动,OFF 路径不变。
- **效果**:elementwise 形态在缺陷①的 env_lookup 支撑下发射正确 float4 代码
  (`y[w,j]=a*x+3`:uniform a 与 const 3 各 OpCompositeConstruct splat,三循环结构,
  dump SPIR-V 验证)。
- **复现臂(G2)**:`k_ew_arg`(k_elementwise 同形)/`k_ew_extemp`/`k_ew_c0`:
  bitwise scalar==vec,且 `k_ew_arg` 任务 dump 的 SPIR-V 含
  `OpLoad %v4float`(Aligned 16)/`OpStore`(Aligned 16)——修复前 dump 无任何 float4
  (静默标量)。
- **gate 期望更新**:`probe_reassociate_flag.py` G3 注释改写(k_elementwise 现已向量化,
  锚仍三臂 bitwise 相等)+ baseline 注记更新(两个"可能伪造 OFF 发射"的构照自本线起
  可正常编译)。

## ② coeff-0 remainder 重复注册(崩 → 正确向量发射,commit 4f09cab7)
- **根因(`try_emit_vectorized_range_for`)**:同一循环体被走三遍(prologue/vector/
  remainder)。prologue 的名字注册有 push/pop 保护,**vector walk 的没有**:第一个体内
  标量值语句(体内 coeff-0 载入 = ScalarMemLoad;多访问点同数组读使 hoister 放弃上提,
  故能留在体内;或 ScalarCompute 重发射)把 raw_name 注册进活表,remainder walk 重注册 →
  `register_value: "tmpNN already exists"` QD_ERROR(本分支构建下 = RuntimeError 崩)。
- **修法**:vector walk 包 `push_value_scope`/`pop_value_scope`(与 prologue 既有模式
  同构)。循环后按名查询的只有 alloca 与任务级值(均注册于 push 之前);spills/
  reassemble 出口携带 Value 对象,不受影响。
- **复现臂(G3)**:`k_ew_c0same`(`y[w,j] = x[w,j] + x[w,0]`,同数组双访问点):
  修复前 vec 臂 `RuntimeError: [spirv_ir_builder.cpp:register_value@1129] tmp12 already
  exists.`(scalar 臂正常);修复后 bitwise scalar==vec + 零 IR-builder 错误。

## 门禁
- **`tests/robostudio/probe_c2_emitter_fixes.py`(新;union runner 新 gate 11,label /11)**:
  三缺陷各一硬门臂(G1 三形态 / G2 bitwise+SPIR-V 结构 / G3 bitwise+零重复注册),
  scalar 臂 = 旗标全 OFF。**ALL HARD GATES PASS**(venv8,GPU 空闲,2026-10-12)。
- **union runner 11/11 PASS**(venv8,2026-10-12;原 10 门全数保留通过 = OFF 路径零变化
  全电池复验):S2/S3/u8/readback/S-guard/C0S/C2 bitwise/R5/rea/subgroup/c2-emitter-fixes。
- **`test_return.py` 46 passed**(venv8,pytest+xdist)。
- **probe_spirv_vectorize(C2 门 7)ALL PASS**、**probe_reassociate_flag(门 9)ALL HARD
  GATES PASS**(rea 准确度 chol 1.47e-07 / matvec 5.71e-07 / sym 1.74e-06)不变;perf
  WARN-only(机器双峰态,本线无 perf 钉值需求)。

## footprint
native +62/−18 行 1 文件:`codegen/spirv/spirv_codegen.cpp`(env_lookup + 五处查点切换 +
fast_dag 资格收紧 + store dtype 改 ext-ptr pointee + vector walk scope 包裹;含注释)。
**无前端/runtime/其他 codegen 改动,无新 compile_config 旗标,offline-cache key 无新维度**
(生成的 SPIR-V 形状差异自然进 key)。门禁:probe_c2_emitter_fixes.py(新,264 行)+
union runner gate 11(+19/−5);probe_reassociate_flag.py 注释修正(+11/−11)。
- **commits**:7880d68f(①)→ 2ebc88d4(③)→ 4f09cab7(②)(③先于②:dtype 误拒
  解除后缺陷②的复现形态才可达 vector walk;①先于③:k_elementwise 放行后依赖 env_lookup
  发射)。
- **wheel**:`dist/quadrants-1.3.4.dev78+g4f09cab72.d20261011-cp312-cp312-macosx_13_0_arm64.whl`
  → `/Users/mini0/Code/mujoqo/_probe.venv8`(python3.12,tuna 依赖镜像)。
- **状态**:landed + verified(分支 `mujocoqd/c2-emitter-fixes`,未 merge)。

# mujocoqd/n2-to-numpy-type 条目(2026-10-12;分支 `mujocoqd/n2-to-numpy-type`,基于 f906f29b)

**to_numpy_type 收 VectorType/MatrixType**(`python/quadrants/lang/util.py`,+8,
python-only):此前只收标量 DataType,复合类型落到函数尾 `assert False`;消费方
(mujoqo `io.py` D19 解包绕 + 3 处同类)被迫逐层 `.dtype` 解包。现函数头加一分支:
`isinstance(dt, MatrixType)`(VectorType 是其子类,单判定覆盖两型)→ 递归
`to_numpy_type(dt.dtype)`;口径 = **元素 np dtype**,形状/extent 语义归调用方
(不附形状,满足 io/sensor 场景)。late import `MatrixType` 断 import 环
(util ↔ matrix)。零 native、零旗标、offline-cache key 无涉。
**背景需求**:mujoqo 需求清单 N2(warp 能力缺口 #3,"小改 util,顺手项")。
**语义钉(gate 12)**:`tests/robostudio/probe_n2_to_numpy_type.py`(新,155 行):
S1 标量 12 分支逐对象钉(`is np.float32` 级,标量回归底线);S2 复合直接调用
(vec3/vec6 f32/i32/f64 + mat2x3 f32/mat3x3 f64)→ 元素 dtype;S3 嵌套复合公开
构造不可达(cook_dtype 拒)→ SKIP + 明标合成 2 层解包钉(递归深度无关);S4 复合
dtype ndarray from_numpy/to_numpy 往返(vec3/vec6 i32/mat2x3 + 标量对照;值 + dtype);
S5 未知 dtype 保持响亮(raise,不静默)。CPU-only 进 union runner gate 12(API
架构无关;编号 13 留给 A10 线,本分支 runner 计 /12,+17/−5)。
**验证**:venv9(`/Users/mini0/Code/mujoqo/_probe.venv9`,python3.12,tuna):
probe RESULT: PASS(cpu);**union runner 12/12 PASS**(既有 11 门全数保留通过 =
标量调用点零变化复验;2026-10-12);`test_return.py` 46 passed。**验证以
site-packages 叠改态执行(盘约束):venv9 底座 = dist wheel `dev82+g0de907cd2`
(≡ robostudio-main native),其上仅叠改 `lang/util.py` 单文件(`_lib` 未触);
wheel 留待合并态统一烤制(纯 python 改动,分钟级)。**
**footprint**:python +8/−0(`lang/util.py`)、tests +155(probe,新)、runner
+17/−5。mujoqo 侧 io.py:146 解包绕退役(改回直接调用)与全库回归属 mujoqo
闭包,不属本线。
**状态**:landed + verified(分支 `mujocoqd/n2-to-numpy-type`,未 merge)。
## A10 复核 — 列序"miscompile"系证据算法次序错误,编译器无罪(mujocoqd/a10-nested-loop-raw)
- **描述**:需求清单 A10(嵌套 runtime 循环 RAW:列序 Cholesky 双后端 nan,定案"自家
  codegen bug、下一 native 线首项")复核**推翻**。原证据(mujoqo `_probe/cs_evidence/
  cs_dbg_chol2.py` 族)的列序形态在 `i<j` 分支除以 `U[j,j]`,而 `U[j,j]` 由同一中层循环
  **更晚**的 `i==j` 分支写入——顺序语义读初值 0 → ±inf → nan;该程序的 **numpy 孪生同样
  nan**,qd 输出与孪生逐元素一致(nan 模式含内;cpu+metal;dev36 与当前 wheel 双验)
  = 双后端忠实编译,无 codegen 缺陷。次序正确的列序 Cholesky(对角先算,仅依赖更早列;
  func+3-D+serialize 移植形态,n=8/24)双后端全过:cpu 0.00e+00 / metal ≤4.77e-07。
  **零 native 改动,无修复需求**;原 A10 修复 plan(dump/diff IR)不再需要。
- **背景需求**:A10 原条目把本项列为"下一 native 线首项";本轮在最小化复现阶段
  (带精确语义孪生对照)即证伪。row-order 律**保留**,理由更正:行序形态的除数
  (对角/更早行值)在其写入后可见,列序原形态除数在写入前——是算法数据依赖结构问题,
  非编译器缺陷;A5/A8/A11/A12(Apple MSL loop-carried 族)不受影响,仍按原账。
- **语义钉(gate 13)**:`tests/robostudio/probe_a10_nested_loop_order.py` —
  A 正确列序 inline 2-D(n=8/24)/ B func+3-D(n=8/24)/ C 行序控制(2.38e-07)/
  **D 无效形态 ≡ numpy 孪生精确断言**(nan 模式 + 有限值;faithful-compilation pin,
  防未来 codegen 变更悄悄改变退化程序输出)。cpu+metal 双臂进 union runner
  (**gate 13**;编号 12 留给 n2-to-numpy-type 线,本分支 runner 计 /12)。
- **footprint**:tests +237(新探针)、runner +21/−3;docs 本条目。零 native、零旗标、
  cache key 无涉。
- **状态**:landed + verified(分支 `mujocoqd/a10-nested-loop-raw`,未 merge;gate 13
  cpu+metal 双臂 RESULT: PASS,venv4 = 0de907cd ≡ HEAD native,2026-10-12)。

# mujocoqd/a19-stlf-fallback 条目(2026-10-13;分支 `mujocoqd/a19-stlf-fallback`,基于 ec916733)

**A19:store_to_load_forwarding 硬 assert → 保守回退**(`quadrants/ir/
control_flow_graph.cpp`,+15/−3,commit 79e660fe):`CFGNode::
store_to_load_forwarding` 在"被 forward 的 result 是 TensorType 而消费 load 非
TensorType"时硬 assert"load_src 必须是 Const-offset MatrixPtrStmt 且 result 是
MatrixInitStmt"——mujoqo `_sensor_touch` 编译即炸
(`control_flow_graph.cpp:store_to_load_forwarding@425`)。IR 实证(mujoqo
`_probe/a19_ircatch.log`,崩前最后 dump = `ray_geom` func 编译单元标量化后):
病态形态 = **tensor 值整体存入标量类型 alloca**(scalarize store 侧未拆分的
病根)+ 随后同 slot 的纯标量 load——此时"Const-offset MatrixPtr 元素提取"
precondition 恒不成立,且若强行 forward 会把 tensor 塞进标量 mul(旧 assert
挡的本就是坏 forward,但方式是炸死编译)。修复 = 形态不满足(含 offset 越界
防御)即 `continue` 保持 load:forwarding 是优化,放弃只少优化、正确性无损。
**同函数同族审计**:Alloca+TensorType 两分支本就是保守 `continue`;
Identical store elimination 段无同族 assert(same_value 空安全、get_store_data
noexcept)——单点修复。
**语义钉(gate 14)**:`tests/robostudio/probe_a19_stlf_fallback.py`(新,611 行):
vendor mujoqo `_sensor_touch` kernel + ray 闭包(ray_{geom,plane,sphere,capsule,
ellipsoid,cylinder,box}/_ray_{map,quad,eliminate} + normalize_with_norm/transposed,
Apache-2.0;ConeType/MJ_MINVAL 内联,math/ray namespace shim)——**触发单元是
完整 kernel 编译**(纯 ray 闭包或纯前端合成形态 c1-c8 均不触发)。双 wheel 行为:
旧 wheel(0de907cd)死于 @425 assert(gate FAIL);修复 wheel 编译通过 + 钉值
sensordata 全零 + ray_sphere 直调命中臂(dist=1.5/normal(0,0,-1))。
CPU-only(缺陷为 IR 形态缺陷,架构无关)。**注意**:纯 `ray_geom` 闭包单编译
不触发,@425 需要完整 kernel 上下文,蒸馏记录见 mujoqo `_probe/probe_a19_distill.py`。
**验证**:探针 a(mujoqo `probe_a19_jump.py touch`)旧 wheel `*** MatrixPtrStmt
ASSERT ***` → 新 wheel `LAUNCH OK (no assert)`;simple victim 数值不变;mujoqo
三门禁(venvA19 = dev95+g79e660fea):history 50/59(+24 解锁)、constraint
71F/77P(dev78 基线 73F,净解锁 2)、io 80F(= dev78→dev95 基线回归 geom1
NameError,非本修复,另行登记)。
**已知边界 / A20 家族证据(本线不动手,移交)**:(a) `StateParityTest::
test_state_history_get_set`:旧 wheel = A19 assert 拒编译(FAIL);新 wheel =
编译过但 LLVM `ICmp` AssertOK abort(Instructions.h:1187)——kernel launch 即
崩进程(证据 mujoqo `_probe/a19_history_v2.log`);(b) ray_geom real_func 分发
返回 dist=-1 而 normal 有效(双 wheel 一致,`_probe/probe_a19_sphere_iso.py`);
(c) 病根疑似 scalarize 的 tensor-into-scalar-alloca store(见上)。三者同邻域,
候补 A20,由协调人派线。
**footprint**:C++ +15/−3(`control_flow_graph.cpp` 单点)、tests +611(gate 探针,
新)、runner +17/−4(gate 14 + 标签 /12→/14 对齐,门数 13→14)、docs 本条目。
零旗标、offline-cache key 无涉(无新配置)。
**状态**:landed + verified(分支 `mujocoqd/a19-stlf-fallback`,未 merge;gate 14
双 wheel 定向验证,union 全门与 test_return 见本线最终报告)。

## A20 — branch-scope NameError 可操作诊断 + "dev78→dev95 回归"定性推翻(mujocoqd/a20-loopvar-scope)
- **描述**:`get_var_by_name` 的 builtins 回退失败时,若该名字曾绑定于**已闭合的 block
  scope**(runtime if 分支/循环体/内联 func 局部——`VariableScopeGuard` 弹栈时记入
  retired 集合),改抛可操作诊断:点名变量、说明"分支内首次绑定、闭块后不可见"、指向
  预声明约定(`geom1 = 0`);两处 raise 改 `from None`,掐掉误导性的 chained
  `AttributeError: module 'builtins' has no attribute ...`(正是它把 A20 排查引向
  builtins 回退路径)。typo(从未绑定)保持原 generic 报错。零语义/解析行为变更。
  文件:`python/quadrants/lang/ast/ast_transformer_utils.py`(+14/−2)。
- **背景需求**:mujoqo io_test geom1 NameError 家族(需求清单 A20,登记为"dev78→dev95
  基线回归,34 条,三 ast/ 嫌疑 commit")。本线最小复现+多 wheel 归因**推翻回归定性**:
  触发形态 = 名字在 runtime if 分支内**首次绑定**(tuple-unpack 自 func 返回,
  `collision_primitive.py _primitive_narrowphase` 的 `geom1, geom2 = geom_collision_pair(...)`
  等;非 loop 变量),后续兄弟块读取。该形态在 v1.3.3 / dev25 / dev36 / dev78 前端
  (overlay)/ **dev82 整 wheel** / dev94 / dev95 全部同样失败——从未在任何 wheel 上编译过;
  三嫌疑 commit 实为 dev8(S2)/dev10(S3)/dev27(M2+A4),全在 dev78 之前,且
  dev78..dev95 窗口零 ast/ 改动;dev82 整 wheel io_test = 80F/87P 与 dev95 完全一致
  (dev78=11F 应系当时 mujoco_qd 侧尚未走到该 kernel,reach 差异,非编译器回归)。
  **解堵配方在 mujoqo 侧**:`_primitive_narrowphase` 体内补预声明
  (`geom1 = GeomV()` 先例即 collision_core.py:113;`margin=0.0` 等标量/vec 默认)——
  `_probe/a20_predecl_fix.py` 钉编译+数值(23.5 正确);补丁副本(仅 _probe/_dbg 抛置
  验证,未动 mujoqo 文件)实测 geom1 签名 36→0,下一层阻塞 = A7 家族
  (`write_contact` 等早退,18 处 "Return inside non-static")。S3 的"runtime if 不做
  alloca 提升"设计决策维持(提升+零初始化=语义变更,未立项,如需另走裁决)。
- **门禁**:本线新 gate = **union gate 14**(`tests/robostudio/probe_a20_branch_scope_diag.py`,
  cpu 文本级:D1 可操作诊断/D2 typo 报错不变(test_name_error 契约)/R1 配方编译+数值);
  上游 `test_return.py` 46/46;union runner **14/14 PASS**(venvA20 =
  dev96+g8b80ce9e,本分支自建 wheel;gate 14 在未修复 dev95 wheel 上如实 FAIL 于 D1)。
  归因/配方探针:`_probe/a20_min_repro.py`(预声明形态全 wheel 通过,对照)、
  `a20_actual_shape.py`(蒸馏中间件)、`a20_branch_bind.py`(忠实形态复现+诊断断言)、
  `a20_predecl_fix.py`(配方钉)。
- **footprint**:前端 +14/−2;tests:新探针 +155、runner +16/−4(含 /12 硬编码总数
  去 陈旧化);ledger 本条目。零 native、零旗标、cache key 无涉;休眠机制无。
- **状态**:landed + verified(分支 `mujocoqd/a20-loopvar-scope`,未 merge;待协调人
  裁决:①mujoqo `_primitive_narrowphase` 预声明补丁派单;②A20 需求清单条目更正
  ("回归"→"既有前端缺口+warp 风格代码缺预声明");③runtime-if 提升语义是否立项)。

## Metal-i64 — i64 类型通路定性(记载推翻)+ segmented sort 64 位键补齐(mujocoqd/metal-i64)
- **定性(本线主产出,dev95 基线 cpu+metal 双探针实证)**:"Metal 无 i64" 记载过时——
  `metal_device.mm:1107-1110` Apple3+ 已置 `spirv_has_int64` + `physical_storage_buffer`
  (`feature_64_bit_integer_math = family_apple3`),SPIR-V→MSL 经 spirv_cross 原生 `long`。
  probe_metal_i64.py 12 组形态全实证 metal PASS:i64 load/store(含 INT64_MAX/MIN)、+ - *、
  `// %`(floor 语义)、`(g<<32)|s` 打包、位与或异、比较控制流、i32/f32/bool→i64 cast、
  clamp、bit_cast i64↔u64(radix twiddle 同款)、u64 load/store+sort、
  **plain `sort()` i64 键(= mujoco_qd filter_key 主路径 collision_flex.py:2073 所用)本就 PASS**。
  键宽收窄方案(需求清单原登记)被用户否决后定性——结论:filter_key 主路径零收窄、零绕行,
  键宽保持 i64,无需任何编译器(native)修复。
- **唯一实证缺口 + 修复**:sort_segmented i64 键 GPU 通路(含 metal;CPU 走
  `_cpu_paths._cpu_seg_radix_sort` 主机路径不受影响)。断点 = `_segmented.py` `_seg_scatter`
  sel==1 分支 `bit_cast(keys_in[i], u32)` 宽度硬编码 → 64 位键 64→32 异宽 →
  spirv_codegen.cpp:visit@1033 响亮拒绝(codegen 行为正确)。修复 `1f8600f9`(+16/−5,
  python-only):照 `_seg_hist` 既有 key_width 32/64 静态分支模式,digit 从 u64 键全宽提取后
  预提 8-bit 进 u32 喂 `radix_rank_match_atomic_or`(其 u32 key 契约不变,bit_start=0);
  out-of-range lane 保持 0xFFFFFFFF→dump-digit 约定;**32 位路径逐位不变**(同语句集外加
  一层 static 包裹)。
- **性能**(8M 元素,交错 i64/i32 配对中位,GPU 空闲,dev109 两轮一致):copy x1.03-1.12、
  打包 x0.74-0.91、plain sort x2.15-2.44(= 8 遍 vs 4 遍通过数翻倍,预期内);
  全部低于 4× 热路径阈值,无 hot-path。绝对 GB/s 受标量 codegen kernel 形状限制(~15-30 GB/s),
  比值才有效。
- **门禁**:本线新 gate = **union gate 16**(`tests/robostudio/probe_metal_i64.py`,
  cpu+metal 双臂;cpu 臂含 segmented-i64 主机路径,metal 臂 t11 翻硬门——修复前该探针在
  未修复 wheel 上如实 xfail 于 bit_cast 拒绝;8M perf 段 informational,>4× 打 ALERT;
  runner 顺带删一处 a20 遗留的硬编码 `/14` 陈旧总数回显)。union **16/16 PASS** +
  上游 `test_return.py` **46/46**(venv `_probe.venvi64` = dev109+gba344e6ef,本分支自建
  wheel,requirements-test.txt 测试栈 pytest 8.4.2)。
- **footprint**:python-only `_segmented.py` +16/−5;tests:探针 +621(含修复前定性记录与
  xfail 机制)、runner +27/−3;ledger 本条目 + INDEX 行。零 native、零旗标、cache key 无涉、
  休眠机制无。
- **状态**:landed + verified(分支 `mujocoqd/metal-i64`,未 merge,待协调人;注意:本分支
  基点已含 a20 合并 233f99c2——a20 线的 merge 当时落在工作树 HEAD 即本分支 reflog 可证,
  合并本分支会连带带入 a20)。mujoqo 侧遗留(非本线):collision_flex.py:1202 worldid/group
  先用后定义(arch 无关,归移植线);a20 ledger 条目自身 foot/状态块重复、INDEX 行悬垂,
  归 a20 所有者修。

## A7/M2 根修 — early_return 复合返回 + 守卫合并(mujocoqd/a7-m2-compound)
- **描述**:`@qd.kernel`/`@qd.func(early_return=True)` 两缺陷根修(方案 B,评估报告
  a7-m2-eval 实施),单文件为主:`python/quadrants/lang/ast/ast_transformers/
  early_return_transformer.py`。**缺陷①(复合槽)**:prologue 槽初始化 `qd.cast(0, T)`
  对 VectorType/MatrixType 直接死于 `cook_dtype`(前端 `ValueError: Invalid data type`,
  **不是**评估报告静态预测的 LLVM/SPIR-V 断言点——探针实证修正);现 primitive 槽保持
  `qd.cast(0, T)`,compound 槽走 `qd.cast(T(0), T.dtype)`:`T(0)` 单标量广播构造
  (make_matrix_expr 自带 T 的元素 dtype,槽为正确形状张量;mat83(0) 单标量广播亦合法),
  外层 cast 保持 init 的 call 形态(:222-226 over-hoisting 纪律)并显式元素 dtype。
  **缺陷②(作用域)**:return 后语句改"run 合并"——纯语句 run 合并进单个
  `if __er_done == 0:` 守卫(兄弟块首绑退休 → A20 家族 NameError 从根消除);含
  return 的兄弟复合语句保持独立守卫并重开 run(首个 cut 曾全局单守卫,被既有
  func_guard/func_per_call_reset 形态当场抓出"taken return 后同体语句继续执行覆写槽"
  回归,双 wheel 归因确认后改为 run 切分);字面 return 并入当前 run 并丢弃块级死尾
  (旧逐语句守卫恰好跳过它,合并后必须显式 drop);首个无条件 return 不加守卫。
  f_guard/f_void/k_nested 形态 IR 与旧 pass **unparse 逐位相同**(_structure_check +
  新旧双模块 unparse 等价钉死),f_loop_ret 差异恰为预期的纯语句合块。**顺手校验/响亮
  拒绝**(编译期):值 arity vs 槽数不匹配、bare+value 混用、嵌套 tuple 注解
  (typing.get_origin 判别,别名拼写同杀)、struct/自定义类返回——后两类信息明指
  "手工单出口(flag-guard)写法";注解元素按 rewrite 期 globals 求值分类
  (primitive/compound/拒绝),`-> tuple[T]` 单槽拼写顺带可用。
- **背景需求**:mujoqo D8 移植(A7 账):mujoco_qd 全库 early_return=True 零使用,
  18 处字面 return-in-loop 炸弹 + 58 处手工单出口存量(ret0qd/doneqd 3851 行/23 文件)
  均被①②挡死;write_contact 本体标量不受挡,挡的是 gjk/sdf/ray/box_box 复合返回族。
- **门禁**:本线新 gate = **union gate 17**(`tests/robostudio/probe_early_return_forms.py`,
  cpu+metal 双臂,TOTAL 16→17):既有 7 形态复跑 + **F1-F14 矩阵**(全 numpy 孪生,
  cpu 27 门/27 PASS;metal 26 PASS + 1 有据 SKIP,见下)——F1 vec3 槽 tuple/F2 mat3 广播
  槽/F3 tuple(vec,i32)/F4 **box_box 真实形态** Tuple[vec8f,mat83f,mat83f] 4 站点/
  F5 **ray_mesh_with_bvh 真实形态** 6 槽/F6 while+return 复合复活/F7 kernel 侧
  early-return(限 cpu,见发现)/F8 kernel 嵌套 break+return/F9 复合槽 per-call reset
  (over-hoisting 纪律扩展到 T(0) 形态的验收)/F10 守卫内 for/F11 四复合站点/F12 新校验
  负例 ×4(arity/bare+value/嵌套 tuple/struct)/F13 守卫内复合+标量首绑交错/
  **F14 = 缺陷②验收形态**(pre-fix:`Name "t" is not defined`,post-fix 数值过)。
  上游 `test_return.py` **46/46**;union runner 17 门全量(venva7 = dev104 自建 wheel,
  requirements-test.txt 栈,pytest 8.4.2);基线复现探针 pre-fix wheel 双缺陷复现 →
  dev104 双绿翻转。**试点**(不碰 mujoqo 文件,只读其源码):
  `_probe/probe_a7m2_pilot_mujocoqd_forms.py`——ray_mesh_with_bvh(march 循环 2 处
  6 元组复合 return + fall-through)与 box_box(4 处 Tuple[vec8f,mat83f,mat83f]
  return,serialize 循环原地改槽)形态等价 kernel,early_return=True 写法
  **cpu+metal ALL PASS**,锚 = numpy 孪生;真实站点 M2 化可行(正式移植归 P1 线)。
- **附带发现(既有核心问题,双 wheel 0de907cd vs dev104 一致,非本线回归,未修——
  非白名单,归编译器所有者线)**:kernel 循环内条件赋值的旗标作 runtime-if 条件时,
  守卫体内**复合工作**(整值存/元素存+读)错值或跨迭代泄漏
  (`_probe/_kcompound_diag.py` 5 形态 + `_probe/_k7metal_manual.py` 手工单出口等价
  kernel 在 pre-fix wheel metal 同样失败;`_kcompound_narrow2/3.py` 控制:纯数据条件
  下同形态 PASS)。影响:kernel 侧 early_return 的 __er_done 在函数级(循环外),故
  守卫区含复合工作被此 bug 挡(与 pass 输出无关);F7 因此限 cpu,F7/F8 逐元素 launch
  (kernel done 为 launch 级 SIMT-free 语义,k_guard 同口径)。func 侧不受影响
  (内联后旗标落调用方循环内)。精确最小触发器(条目常量性/内联敏感)留待独立归因线。
- **footprint**:early_return_transformer.py +150/−45 左右(单文件,含校验与拒绝);
  probe_early_return_forms.py +648(F 矩阵);runner +22(门 17 段+头注),TOTAL 16→17;
  ledger 本条目 + INDEX 行;_probe 新件 7(证据/诊断/试点,含死尾 drop 与 run 切分
  的归因记录)。零 native、零旗标、cache key 无涉(offline_cache_util 无此字段,
  IR body hash 兜底);休眠机制无。
- **状态**:landed + verified(squash merged to robostudio-main as e79ae899,基点 1fd930dc;
  待协调人 squash)。已知账(P1 线/他线):sleep.py `_sleep_cycle` return-in-loop 假绿
  (惰性编译未触发);18 处炸弹与 58 处手工存量迁移策略;kernel+复合+旗标既有 bug
  的独立归因与修复;单索引行赋值 `mat[i] = vec` 在部分站点被拒(试点用元素级赋值绕开,
  真实 box_box 同拼写可编译,差异待查——归编译器线顺带核对)。

## qd 原生 BVH — 泛型 AABB 空间查询原语(mujocoqd/bvh)
- **设计**:`quadrants/bvh.py`(新,893 行)——warp native LBVH(bvh.cu/bvh.h)算法级
  直译,零 native/零 codegen/零既有文件改动。构建:morton3<1024> 键 → (i32 morton,
  index) Batcher 奇偶排序(host 驱动 parallel_sort schedule)→ key deltas →
  build_leaves → build_hierarchy(delta 分裂+parity tie-break+原子完成计数;完成链
  **串行**保确定)→ mark_packed_leaf_nodes(leaf_size 静音)。遍历:bvh_query_init +
  bvh_query_{aabb,ray,sphere}_next——warp bvh_query_next_impl 骨架,三谓词展开,
  手工单出口,**标量 i32 返回**(兼容无 M2 复合槽的运行时);迭代器状态在调用方 2-D
  ndarray 行(stack[slot,depth]/state[slot,4];qd 无 kernel 内定长数组,探针 P1 实锤)。
  BVHPackedNodeHalf 位域(i:31|b:1)经 bit_cast 打进 vec4 w 通道(与 warp 字节同构,
  NaN payload 往返 cpu+metal 钉死)。refit = bvh_refit_kernel(结构不动 bounds 上提)。
  host Bvh 句柄(num_items 显式口应对 qd 零宽 1-填充)+ 三个 batch 查询 kernel。
- **数字**:gate 18(tests/robostudio/gate_bvh.py,27 断言/臂):**cpu 27/27、metal
  27/27 PASS**(venv-consolidated dev25+g3297e86fe);新运行时(0de907cd wheel)cpu 亦
  27/27。oracle 三层:结构不变量(排列/自查询必中/packed-leaf range 位级划分 [0,n))+
  **numpy 孪生谓词**(48 aabb/48 ray/48 sphere 候选集合逐一相等)+ **warp wp.Bvh A/A**
  (同界 48+48 集合相等,warp 借用为 cpu oracle,对 metal 臂亦有效)。ray 手工:any-hit
  截断、迭代器收窄求最近、球查询;refit 随动;空树/n=1/leaf_size=64/3 次构建逐位确定。
  上游 test_return.py **46/46**。union TOTAL 17→18,gate 18 双臂 PASS;既有 gate
  1-6/13 全过,gate 7-12/14-17 import 期失败 = 母盘 venv 陈旧(dev25 缺
  spirv_vectorize/host_visible/reassociate init 旗标、_cpu_paths、sort_segmented、
  early_return kwarg,各有专名证据;M2 条目记载 17 门全量当时跑在 venva7=dev104
  自建 wheel 上,本机现已无),非本线回归(bvh 分支零既有文件改动)。
- **footprint**:bvh.py +893;gate_bvh.py +437;runner 注册块 +33、头注 +2(TOTAL
  17→18);ledger 本条目 + INDEX 行;_probe/bvh_local_array_probe.py +330(勘察,
  任务书 step-1 另令,不计入本预算)。合计 1369 行 ≤1400。**sort 直调不可行记录**:
  此 wheel 可组合基数排序无 cpu 通路(GPU 楼梯 arm64 死于 block.sync)、
  sort_stage 仅收 field(template() 拒 ndarray + snode 访问)——以 ndarray 形态重述
  同一奇偶网络 + parallel_sort schedule,sort 本体未动。
- **perf 追加验收**(用户追加;一次性/稳态分离口径,_probe/bvh_perf_probe.py:
  1k/10k/50k × uniform/cluster,查询 512/5000 双档批量,5 次中位,wp-cpu 与 qd-cpu
  同进程交错 reps,qd-metal 独进程先过 _gpu_busy.sh)。**构建 = 一次性成本**(中位
  ms,1k/10k/50k):wp-cpu 0.2/2.6/15.1,qd-cpu 3.8/6.6/11.7(小 n 吃 ~55–136 次
  per-stage launch+sync 固定税,50k 反超 wp 的 SAH),qd-metal 19.7/44.1/97.9
  (per-stage launch 延迟主导,非 GPU 计算)。**查询 = 稳态计算**(5000 条批量档
  per-query 中位 ns,uniform/cluster 全场景范围):qd-cpu 22–1843,wp-cpu 75–10582
  (同数量级,qd-cpu 一致 4–6× 快;512 小批量档两者 0.5–1.0× 基本打平——固定路径
  未摊薄,大批量档 qd 吞吐规模化更好),qd-metal 含实测后端地板(389.7 µs/launch,
  任何 kernel 都有,非 BVH 特有)117–23376 / 扣地板 12–22615(≈qd-cpu 的 1.5–8×,
  小场景整批时间≈地板本身)。命中总数三方断言相等(f32 孪生精确;numpy f64 参考
  50k ray 1 例刃边差 51964 vs 51965,容差内;同协议双跑漂移 ~20%,±4× 方差纪律内)。
- **限制**:单 group(groups 不实现,键上 32 位预留);奇偶排序不稳定(平键顺序网络
  相关;LBVH 对任意平键排列有效,候选集与树形无关,gate 确认);构建/refit 完成链串行
  (确定性优先);ray 谓词保持 warp 原版 slab(0×inf=NaN 边界语义一致,robust 变体未引);
  capsule 谓词未实现;单 kernel 可组合形态待基数排序 cpu 通路进 wheel。状态:landed on robostudio-main(squash 76541b81)。
## CPU 头声明状态 per-iteration 重建(mujocoqd/cpu-head-state-per-iteration)
- **设计**:qd CPU 并行 kernel 的头声明标量(pre-loop 声明)被 offload 跨任务提升为
  launch-shared global temp,非 serialize 顶层循环的写∧读构成无同步 RMW = 静默数值
  竞态(mujoqo _flex_edges 字段实锤,27042 槽位 ~7700 随机错;探针
  `_probe/head_state_probe.py` head 形态单调 1..16)。修复 = offload.cpp pre-Offloader
  变换 `rebuild_head_state_per_iteration`(arch 无关 IR,全后端共用,顺带拆 GPU 侧
  同型雷):判据 = **写∧读 ∧ 非序列化 kernel 顶层循环**(range_for/struct_for/
  mesh_for)——克隆 alloca + 循环前无环前缀 store 进循环体入口,循环子树引用全重接;
  alias(pointer alloca)簇 union 后整簇重建;`is_shared`(SharedArray block 级共享窗,
  block collectives 在 func 顶层声明、inline 后落 kernel root)显式排除;写-only 形态
  (test_ret_4k 类)不重建。**语义二元性(ADJUDICATED R3,2026-10-18)**:非 serialize
  顶层循环 = per-program 语义(头状态写∧读 → 每迭代重建);serialize=True = 显式顺序
  任务语义(头状态跨迭代携带,pre-fix 行为不变)——后者是语言级声明契约,消费方 =
  _cpu_paths.py 全家族("one thread, program order, so the loop-carried accumulator
  is exact";计数排序直方图/256-digit 前缀和逐循环显式 serialize)。IR 层"每迭代重置"
  与"顺序携带"两形态同形不可静态区分,唯一判据 = serialize 标志;集成线摘 mujoco
  serialize workaround 后对应 kernel 自动落 per-program 修复面。权威文本 =
  docs/cpu_head_state_development_spec.md 状态段(含 dev_notes 搬运指示)。
- **数字**:探针三形态 ×8 launches:head 全 1 / body 全 1 / serialize 确定性 1..16;
  gate 19 六形态 ×20 launches 全过;test_return.py **46/46**(test_ret_4k 由写-only
  排除救回:首版判据 44/46);AD 电池双轮对照 436 个 test_adstack 失败为分支既有
  (pre/post 失败清单逐条相同,零新增),其余 AD 文件 214/214 双轮全绿;union runner
  19 门全量绿(18 留 BVH 线)。双轮归因:分支基线轮(rebuild 前的 offload.cpp 重编
  wheel)probe_subgroup/probe_algorithms/probe_metal_i64 全 PASS;含缺陷判据轮
  gates 10/16 CPU 臂红(select/sort_radix 全挂、t10 sort INT64_MIN)→ 两条排除
  (is_shared / serialize)逐一救回。
- **footprint**:offload.cpp +260/−12 左右(单文件;pass + 排除 + 注释);新件
  `_probe/head_state_probe.py`(复现/语义钉)、`tests/robostudio/
  gate_cpu_head_state.py`(gate 19,六形态);runner 追加门 19 段+头注,TOTAL 17→19
  (18 槽留 BVH 线);规格状态段+ledger 本条目。零旗标、cache key 无涉(注:离线缓存
  对同版本戳不同二进制会吐陈旧 kernel——本轮实证,缓存键线在案)。
- **限制**:①写-only + 循环内读的未来形态判据会漏,须响亮拒绝或 lint 补位,不许静默;
  ②未标 serialize 的顺序携带写法(头 RMW 跨迭代)在 ≤512×threads 迭代时 pre-fix 靠
  512-chunk 偶然串行,修复后转确定性 per-iteration——顺序算法必须显式 serialize
  (algorithms 已是此口径);③同版本戳不同二进制的 offline cache 陈旧命中(实证,
  gate/电池前清 ~/.cache/quadrants/qdcache;归缓存键线);④metal 端到端冻结中
  (arch 无关变换,metal 臂经其余 gate 间接覆盖);⑤mujoqo 侧 10+ 序列化 workaround
  的移除回归归集成线。

## A32 — early_return:static-if-return 位于 runtime return 后的 SyntaxError 根修(mujocoqd/a32-static-return-order,e0f35bdd)
- **设计**:early_return 单出口重写跑在 IR 构建 walk 之前的 Python AST 层,而
  `qd.static` 条件到 walk 时才求值(call_transformer)——重写器拿不到 TRUE/FALSE。
  原实现对 static 子树"原样保留",首个 runtime return 之后的 static-if-return 被
  塞进 `if __er_done == 0:` runtime 守卫 run,裸 return 落进非静态 if → 前端
  "Return inside non-static if/for" 响亮拒绝(solver linesearch max_condim=4
  dispatch 唯一宿主,286e62f workaround 绕行)。修 = transform_block 对含裸 return
  的 qd.static 节点就地对子树套用本 pass 自身机制,**条件值无关 + 位点无关**
  (TRUE:槽赋值执行+后继守卫死;FALSE:walk 期剪除;前 return 位点靠新 done-store
  注册为 return-site,顺带闭合嵌套 runtime-if 内 static return 的漏守卫缺口);
  无 return 的静态子树字节不变。
- **数字**:AST 级单元矩阵 14/14(mujoqo `_probe/a32_transformer_unittest.py`);
  三态 oracle 探针(`_probe/a32_static_if_after_er.py`)dev109=REPRODUCED / 修复
  wheel=FIXED(含嵌套/静态循环/FALSE 分支/factory 回归臂);upstream test_return
  **46/46**;union runner 19/19(gate 17 early-return compound F1-F14 绿)。
- **footprint**:`early_return_transformer.py` 单文件 +55/−2;`docs/
  a32_static_return_order_development_spec.md`;探针三件在 mujoqo `_probe/`。
- **限制**:①286e62f port 侧 workaround 的还原归 mujoqo 独立 commit(未做,可选);
  ②gate 17 是该 pass 的常驻回归面,禁回退。

## A34 — cast f32→u1 双后端 C 真值化(mujocoqd/a34-bool-cast,91351dbd)
- **设计**:`bool(<float>)` 走 `Expr::cast(x,u1)`;LLVM cast_value u1 分支
  CreateIsNotNull(=ICmpNE)对 float 非法(assertions-on wheel 必崩 assert
  AssertOK,Release=非法 IR 静默 miscompile 风险),SPIR-V 响亮拒(cast@906)。
  修 = real 操作数发**显式无序比较**:LLVM `CreateFCmpUNE(input, ConstantFP 0.0)`,
  SPIR-V `OpFUnordNotEqual(vs float 零立即数)`;int/ptr 保持 CreateIsNotNull。
  前端单点改写(F 案)被定性矩阵否决:qd 浮点 `!=` 的 NaN 语义双后端不一致
  (cpu ordered / metal 经 MSL 塌缩 unordered),复用 `!=` 会复制 cpu 的错语义;
  该分歧独立观察项另立。
- **数字**:双后端探针矩阵全过——int2bool 零回归(cpu+metal);flt2bool cpu
  **abort 134→PASS**、metal 响亮拒→PASS;nan2bool 双后端 **[F,T,T,T] C 语义
  (NaN→True)**;ifflt 双后端保持响亮拒。test_return **46/46**;union runner
  19/19(合并前口径)。
- **footprint**:`codegen_llvm.cpp` +10/−1;`spirv_ir_builder.cpp` +3;**gate 20**
  (`probe_bool_cast.py` cpu+metal 硬断言,含 NaN 案例与隐式真值响亮拒检查;
  runner TOTAL 19→20,metal 臂 wait_gpu_idle);mujoqo 探针两件(三案例 +
  定性矩阵)。
- **限制**:①浮点 `!=` 的 NaN 双后端分歧保持原样(观察项,修则 cpu 侧对齐 C=unordered);
  ②port 侧 support.py 的 `!= 0.0` workaround 可还原 `bool()` 原形(mujoqo 独立
  commit,未做,可选);③隐式真值 `if <real>:` 维持响亮拒(设计如此)。

## A37 — Function::store_dests 生命周期根修 + gate14 x3 共识(mujocoqd/a37-compile-uaf,345dd750)
- **设计**:非确定性 SIGSEGV(compile_kernel → __dynamic_cast 野指针;21 份 crash
  report 同栈异址;宿主 = 含 FuncCallStmt(real_func) 的 kernel;暴露率随进程内存
  布局浮动,缓存/transformer 内容无关)。符号化崩链 = cfg_optimization →
  ControlFlowGraph::store_to_load_forwarding → reaching_definition_analysis →
  control_flow_graph.cpp:370 对野指针 dyn_cast。根因:`Function::store_dests`
  (指向 real_func IR 的持久裸指针集合)在 gather_func_store_dests 中**只增不清**,
  跨 IR 阶段/跨 top-level 编译累积过期指针,经 `FuncCallStmt::func->store_dests →
  reach_gen` 进入分析被 dyn_cast。修 = SCC pop 处 `store_dests.clear()` 后再
  insert(唯一消费者即 CFG 分析)。附带:union runner gate 14(UAF 宿主探针)改
  **3 连跑一致性**(单跑绿=侥幸,实证一轮 19/19 同窗口 21 进程崩、复跑 18/19);
  docs/qd_compiler_dev_notes.md §8 收三线机理。
- **数字**:census(mujoqo `_probe/a37_uaf_census.py`,高暴露 uv-python)修复前
  **6/6 SIGSEGV → 修复后 N=10 STABLE**;test_return **46/46**;union runner 19/19
  (gate 14 x3 真绿)。
- **footprint**:`gather_func_store_dests.cpp` +9/−1;runner gate 14 块 x3 化;
  `docs/qd_compiler_dev_notes.md` +31;mujoqo 探针三件(census/asan_run/st1)。
- **限制**:①崩溃因果诚实挂账——暴露窗随机器状态漂移(pre-fix 曾 6/6 崩 → 验证
  时 0/16 全不崩),"旧崩新稳"对照未当场完成,A37 保持 open,census oracle 在案,
  暴露窗复现时补判定;②ASAN 在 macOS python 扩展主机不可行(linker-signed 忽略
  DYLD_INSERT;重签名后 runtime 自检败,harness 留档);③trash_bin 全程无 flush
  = A24 "IR 驻留不释放"机理(本线只记账未修);④符号化 recipe = build 树 link.txt
  去 `-Wl,-S` 手工重链(dev_notes §8.5)。
- **扩展(2026-10-19,本 squash)**:①同族第二例 UAF 根因定案——scalarize 循环
  语句所有权(familyr worktree tip 的陈旧 A42 中间版缺 v2/v3 所有权 insert;落地
  squash 57391b9f 为正确形态,SSOT 无此洞;因果对照 familyr 布局 pre 3/3 崩 →
  post 5/5 正确);归因配方 `MallocStackLogging=full` + `malloc_history -fullStacks`
  (lite 只记 VM 事件,不可语句级归因)+ 重链 `-Wl,-map` 符号化 alloc/free 栈,
  收入 qd-compiler skill 修复模式。②`TypeCheck::visit(Local/GlobalStore)` 增
  null store-operand ret_type 守卫(ErrorEmitter 出具 IR+源行)——语句生命周期
  回归的常设响亮化层。③验证:dev123+gc0b58c09 wheel int2bool 5/5、test_return
  46/46、union runner **21/21**(gate 14 x3 含)。④限制:SSOT 自身未表现过此洞
  (守卫为预防层);ownership 审计 pass(IR operand 存活性断言,估 ~50 行)评估
  为可选拍板项,未动手——malloc_history 只解归因不解检测,错值形态曾无崩溃签名。

## a46-recursion-guard — @qd.func 递归环响亮检测(ledger A46,mujocoqd/a46-recursion-guard)
- **设计**:前端内联加 thread-local in-progress func 栈(func 对象身份),重入 =
  递归环 → `QuadrantsSyntaxError: Recursive @qd.func call detected (inline
  cycle): <全环链>`;挂点两处(func.py 普通 func 编译 with 包裹、real_func
  do_compile 的延迟 func_body 内)。Kernel 根不加(环必经 func;多 materialize
  key 重入为顺序非嵌套)。
- **数字**:A46 复现从静默 SIGKILL(25s 2.76GB)变秒级命名错误(直接 `f <- f`
  与互递归 `a <- b <- a` 双验);test_return 46/46;union 21/21 × {默认,
  QD_IR_AUDIT=1};ruff 归零。
- **footprint**:_func_base.py +30、func.py +3/−3、spec 文档。零默认行为变化。
- **背景**:A24 Phase 0 附带发现;历史"runtime-loop 2.5h 不收敛"疑似同机制。

## audit-lint — IR 所有权审计启用层 + lint 本地化最小支持(mujocoqd/audit-lint)
- **设计**:①复用既有 IRVerifier(operand 地址可见集断言 + parent 回指针),不加新
  pass:`verify_if_debug` 增 `QD_IR_AUDIT=1` env 强制门(Release 轮子/门禁可开审计;
  env 不设 = 逐位现状);offload_to_executable 的 kernel 路径 scalarize 分支首句、
  force_scalarize_matrix 分支、func 路径 scalarize 分支及 make_block_local 内部
  scalarize 后各补一记 verify。②**Stmt sized operator delete 审计隔离**:
  QD_IR_AUDIT=1 下释放的语句 0xDD 投毒不归还——地址永不复用,把 verifier 的地址集
  成员检查从"复用假阴性"补成确定性悬垂检测(实证:无隔离时同一 bug 审计静默漏报;
  有隔离时悬垂读撞 0xDDDD… 毒签名)。③审计发现 = 按地址命名的 QD_ERROR(刻意不
  deref 悬垂 op——原断言的 op->type() 参数本身会踩毒)→ Python 级 RuntimeError;
  非审计 debug 路径保留原断言。④lint:根 CMakeLists `CMAKE_EXPORT_COMPILE_COMMANDS`
  (compile_commands.json 落 build 树);pre-commit ruff 钩子本已配置,本机 uv tool
  装 ruff 实跑通。
- **数字**:抓真 bug 实证——worktree 临时复刻 familyr 陈旧 scalarize(去所有权
  insert),`QD_IR_AUDIT=1` 编 int2bool → `IR broken (audit): stmt 167 (…) has
  dangling operand 0x… - freed-but-referenced statement (ledger A37)`(Python
  RuntimeError);同轮 audit-off = 静默 SIGSEGV(旧行为对照)。健康面:int2bool
  双模式 PASS、test_return 46/46(audit-on)、clang-tidy 改动 3 文件 0 warnings、
  ruff 首跑 7 findings(全 --fixable,已由 ruff-import-sort 线清零)、union runner
  21/21(默认)与 **21/21(QD_IR_AUDIT=1 全量补验,零误报)**。
- **footprint**:verify.cpp +22/−1;ir.h +6;statements.cpp +19;
  compile_to_offloads.cpp +6;make_block_local.cpp +1;CMakeLists +1;spec 文档
  (docs/audit_lint_development_spec.md)。默认语义零变化。
- **限制**:①隔离以空间换检测——audit 模式语句释放不归还,超大 kernel(A24 家族)
  audit-on 编译内存会放大,勿开;②地址集审计不覆盖"跨 IR 阶段外部注册表陈旧指针"
  (A37 store_dests 类,指针不在本树)——该类靠 store_dests.clear() 与探针;
  ③QD_UNREACHABLE 在 NDEBUG 下落空(QD_ASSERT 非致命),故审计路径显式命名错误。

## Family R — real_func × 复合值/控制流全方向修复 + A42 scalarize 泛化(mujocoqd/familyr-realfunc-composite)
- **设计**:两笔编译器修复 + 一笔分析鲁棒性修复,同一 worktree 线落地:
  ①**Family R builder 解析**(func.py):`@qd.func` 内联叶在 real_func 体编译
  (嵌套 do_compile)时,`get_tree_and_ctx(ast_builder=current_kernel.ast_builder())`
  解析到外层 kernel 的 builder,叶的控制流语句进入永不编译的死树,普通赋值
  跟随 compiling_callable 进 real_func 自身树——叶被劈成两棵树,guard if 全丢、
  体直排 last-wins(A41 分发乱值/A44 体腐坏/A33 跨界错值的统一根因)。修 =
  plain 内联路径 builder 解析改从 compiling_callable 取(嵌套时正确指向
  real_func 自身;非嵌套场景两 builder 同对象,零行为变化)。
  ②**A42 scalarize 泛化**(scalarize.cpp):张量 val 非 MatrixInitStmt 的 store
  不再 assert——临时 alloca + LocalStore 停泊 + per-element MatrixPtr/LocalLoad
  (v1 漏插 dest ptr 的 SIGSEGV 族与 v2 move 后空 origin 两轮迭代已修)。
  ③**alias_analysis 保守 uncertain**:同 ndarray 不同索引数(全元素 vs 链式
  分量两种合法降级形态)不再 assert,返回 uncertain。
- **背景需求**:A40-A44 族(bvh-interop 线 ray 链 2F 阻塞 + A33 转正 +
  ray_boundary P4 乱值 + box4b 体腐坏);mujoqo 账本 A40-A44 全量状态。
- **数字**:20 行最小复现(fr_min_er_leaf)dev128 5/5 MISMATCH → dev145
  PASS;3-way 控制矩阵(直调/func 分发/real_func 分发)PASS;ray_boundary
  P1–P4 全过(P4 从 -1.0/-0.061341 乱值翻正 0.102987/0.020101 与 P2/P3 一致);
  box4b 从 -1.0 翻正 0.17595585≈box4a;A33 探针 FIXED;A42 V 矩阵五变体全绿
  (V2/V2c 从 scalarize@55 断言/MatrixPtr@132 拒绝翻绿);test_return 46/46;
  union runner **21/21**(含 gate 14 x3 新钉值、gate 20、gate 21 首跑)。
  dev145(修复)/dev148(+typeid debug)双 wheel 复验。
- **footprint**:func.py +15/−1;scalarize.cpp +32/−5;alias_analysis.cpp +7;
  statements.cpp(A42DBG 追踪,历史保留);gate 21 探针 + runner 接线
  (TOTAL 20→21);dev-notes §8;任务书/评估文档。零旗标。
- **限制**:①真根因 taichi 局部矩阵语义(矩阵值分量读产生 tensor 形状)——
  scalarize 泛化 = 正确修复层,前端无需改动(A42 原评估的"前端类型化修复"
  方向撤销);②alias 索引数断言保守 uncertain 可能轻微增加保守性(正确性
  优先);③FRDBG/A42DBG 插桩已回退留史;④A37 线 UAF(a19 宿主)x3 防闪烁
  维持,根因 lldb 深挖另行立项。
