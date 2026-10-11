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

# mujocoqd/compiler-line 条目(2026-10-11;**已合入 robostudio-main**,merge 33e7dabf --no-ff)

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
