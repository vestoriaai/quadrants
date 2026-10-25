# A69 — Metal 库编译接通 `fast_math` 配置（MTLCompileOptions fastMathEnabled/mathMode）

> 2026-10-25。分支 `mujocoqd/a69-metal-fastmath`（worktree quadrants-a69wt），基点
> robostudio-main b5db014f6。来源：mujoqo metal-battery 线（port/metal-battery-clean）
> 需求清单 A69。

## 问题

`quadrants/rhi/metal/metal_device.mm::MetalDevice::get_mtl_library` 编译 MSL 时
`MTLCompileOptions` 只在设备支持 atomic_float 时才构造且仅设 `languageVersion`，
`fastMathEnabled` 取 Metal 默认 **YES**（FMA 收缩、重结合、假定无 NaN/Inf、近似
超越函数全开）。而 `CompileConfig::fast_math`（`qd.init(fast_math=...)`，默认 True）
只被 LLVM 系后端（cpu/cuda/amdgpu）消费，对 SPIR-V→MSL 路径无任何作用：
**用户设 `fast_math=False` 时 metal 仍在 fast-math 下跑**，CPU/metal 位级 parity 不可控。

port 侧实锤（mujoqo 需求清单 A69）：
- `util_misc.wrap_circle` 精确相切用例：`end[3]*sqrad - radius*end[2]*sqrt1` 纯 f32 一致
  舍入应为 0，metal 得 4.06e-10 → `length_circle` 的 cross 符号由噪声决定 → wlen 翻成
  2πr（port 侧已加 D19 守卫绕行）。
- `collision_gjk` box_box_max（-0.036362633 vs -0.03636224，places=7）/ max2（+1.39e-6 vs
  -4.94e-5 符号翻）metal 专属 F：A47 位级平局族，CPU 靠 f64 `_linear_combine`（A55 metal 退
  f32）+ qd LLVM reassoc 修复（7a077591）才过；metal fast-math 重结合下补偿求和亦不可靠，
  port 侧无修法。

## 判决（设计）

- **语义**：`fast_math` 是 Program 级旗标（一进程一 `qd.init`），Metal 设备由
  `MetalProgramImpl::materialize_runtime` 在拿到 `config` 时创建 → **device 级**
  `MetalDevice::set_fast_math(bool)`（默认 `true` = 与当前行为逐位一致），
  `get_mtl_library` 据此构造 `MTLCompileOptions`：
  - macOS 15+/iOS 18+：`mathMode = fast ? MTLMathModeFast : MTLMathModeSafe`
    （`fastMathEnabled` 已 deprecated；Safe = IEEE 严格，与 LLVM 无 fast-math 旗标的
    CPU 代码生成同口径——Taichi codegen 发 plain fmul/fadd，无收缩）；
  - 旧系统：`fastMathEnabled = fast`。
  - options 对象改为**无条件构造**（原先无 atomic_float 能力时传 nil）。
- **不做**：per-kernel 旗标（RHI `PipelineSourceDesc` 跨后端改动，无需求）；
  Relaxed 档位（无 parity 价值，待有 perf 需求再评）。
- **缓存**：qd 离线缓存键已含 `config.fast_math`（`offline_cache_util.cpp:76`），
  缓存产物为 SPIR-V、MTLLibrary 每进程从 SPIR-V 重编 → 翻旗标无 stale 命中；
  Metal 无持久 pipeline cache（`create_compute_pipeline` 每次编译）。raster 路径
  （`create_raster_pipeline` 的 vert/frag 库）同走 `get_mtl_library` → 同受控。
- **默认值不变**：`fast_math` 默认 True → 现有用户零行为变化；mujoqo port 侧
  conftest 自行决定 metal 下默认用 `fast_math=False`（parity 优先，A24 cfg 旋钮先例）。

## 验证方案（含反证判据）

1. **union gates**：`zsh tests/robostudio/run_robostudio_gates.sh <venv>` 全过 + 上游
   `pytest tests/python/test_return.py` 46/46。**反证**：任一门变红即方案错或副作用
   （fast_math=True 默认路径必须逐位不变——gate 7/9 口径）。
2. **新 gate（本线）**：`tests/robostudio/test_metal_fastmath_flag.py`——metal 下
   `qd.init(fast_math=False)` 时 `(-1)*(r*r) + r*r*1.0`（wrap_circle 相切形态）精确为 0，
   且 `fast_math=True` 时行为与旧轮一致（允许非零）；cpu 两档均 0（LLVM 不收缩）。
   **反证**：precise 档仍非零 → Safe 模式未生效（options 未传到位 / 版本分支走错）。
3. **port 侧验收（mujoqo-mbwt，克隆 venv 装新轮）**：
   - `_probe/mb_metal_fastmath_probe.py` / `mb_wrap_tangent_probe.py`：metal `sol11.y` → 0；
   - collision_gjk `test_box_box_max/max2` metal 转绿（**主判据**）；horizon 维持 CPU 同挂
     （A47 既有，不属本线）；
   - 全量 metal battery（28 模块）零新增 F；cpu 面不受影响（LLVM 路径未动）；
   - render metal 86P + golden（若可跑）在容差内。
   **反证**：max/max2 仍挂 → A47 平局不止 fast-math 一因（如 f64 不可用的 A55 余项），
   本线仍以 gate 2 + 零回归收口，账目改写归因。
4. **perf 对照**：同机交错 min-of-N（≥6）：forward_test 墙钟 + `_probe/metalov_bench_min.py`
   （若 venv 可用）fast vs safe；预期 safe 慢 0–30%，结论进 ledger（数字，非感觉）。

## footprint 预算

- 代码：`metal_device.{h,mm}` +~20 行，`metal_program.cpp` +1 行，新 gate 测试 1 文件；
  零 IR/codegen 改动。
- 构建：worktree `build.py wheel`（sccache 热缓存 ~5 min）；验证 venv = APFS 克隆
  `_port.venv`（`cp -Rc`）装新轮，**共享 venv 不动**（2026-10-25 装轮纪律）。
- 运行时：Safe 模式潜在 GPU 吞吐下降（待测）；默认 fast 档零变化。

## DoD

1. gates 全过（union + test_return 46/46 + 新 gate）；
2. 新轮 banner commit 字段 = 本分支 sha；
3. port 侧 max/max2 metal 判定（绿 或 归因改写）+ 全量 metal battery 零新增 F；
4. perf 数字入 ledger；`docs/robostudio_features.md` 四字段条目 + INDEX 行（squash sha 由
   协调人填）；mujoqo 需求清单 A69 闭合。
5. 不 merge/push；分支 + wheel 路径交协调人 squash。

## 白名单

- `quadrants/rhi/metal/metal_device.h`、`quadrants/rhi/metal/metal_device.mm`
- `quadrants/runtime/program_impls/metal/metal_program.cpp`
- `tests/robostudio/test_metal_fastmath_flag.py`（新）、`tests/robostudio/run_robostudio_gates.sh`
  （加一门）
- `docs/a69_metal_fastmath_development_spec.md`（本件）、`docs/robostudio_features.md`（ledger）
- mujoqo 侧（另一 worktree，另行 commit）：`mujoco_qd/_src/conftest.py`（`QD_FAST_MATH` 旋钮）、
  需求清单/TODO。

## 状态(2026-10-25 收口,待 squash)

- **实施**:642b3135e(device 级旗标 + gate 22)→ c8523b6b1(具体类型指针)→ **664dd6c17**(strict 档
  `#pragma METAL fp contract(off)` 注入 + gate 22 改 wrap_circle 完整形态/numpy f32 孪生判据)。
- **判决修正**:原判"Safe = IEEE 严格、与 LLVM 无旗标路径同口径"**只对了一半**——Safe 修除法
  (fast 最大 2 ulp → 0),但 Metal 的 FP 收缩是独立开关,`MTLMathModeSafe` 下 `a*b-c*d` 仍 7234/20000
  失配(最大 1664 ulp);必须源级 `#pragma METAL fp contract(off)`。sqrt 两档均 ≤1 ulp 非正确舍入
  (Apple GPU 库属性,不在 MTLCompileOptions 可控面)→ 记限制。
- **验证**:gate 22 strict 与 f32 孪生逐位同(新轮 0/40;旧轮 dev152 FAIL = 回归信号有效);union
  21/21 存量 + gate 22;gate 19 基线同挂(既有,非本线);test_return 46/46;banner 664dd6c1。
- **反证判据触发(验证方案 §3 的"若 max/max2 仍挂")**:box_box_max 转绿,**max2 仍挂** → 归因改写:
  max2 真因 = A55(metal 无 f64,`_linear_combine` 退 f32),fast-math 非主因;另 **cylinder_cylinder_contact
  在 strict 下新挂**(7.3e-8 @ places=7,FMA 单次舍入在近抵消项上反而更准)——strict ≠ 更接近 f64 参考,
  metal f32 GJK 的 7 位小数断言是刀锋,处置归 A47/A55 口径。util_misc wrap0 在 strict 下暴露 C 同款
  退化(重合切点 → 闭区间 is_intersect 判相交),port 侧 D20 守卫后三档全绿。
- **perf**:humanoid×64 100 步 min-of-6 ×3 轮交错:fast 13.86/13.86/9.73 ms vs strict 14.16/14.51/8.93 ms,
  差异在非独占机器噪声内,无显著代价。
- **默认**:qd 侧默认不变(fast_math=True);mujoqo conftest 按 arch 自选(metal strict / cpu 默认),
  全量 strict battery 结果见 mujoqo TODO metal-battery 线条目。
