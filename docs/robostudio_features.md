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
