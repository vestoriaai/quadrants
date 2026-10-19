# Family R 任务书 — real_func × 复合值/控制流(A41+A44+A33;worktree `quadrants-familyr`)

- 线名:`mujocoqd/familyr-realfunc-composite`;基线 robostudio-main @ 99ba7419
  (含 A32/A34/A37 合并)。账:A33(候选转正)/A41/A44 + 新发现 N-1(见下)。
- 评估成文:mujoqo `docs/qd-A40-A44-缺陷族-需求评估.md`(§3 Family R,review 通过,
  2026-10-18 用户开工令)。

## 已确证(本轮归因产出,2026-10-18 深夜)

### 最小复现(全新,比在案探针小一个数量级)

`_probe/fr_min_er_leaf.py`(mujoqo):**一个 `@qd.func(early_return=True)` 叶 +
一个 `@qd.real_func` 直接 return 该叶** 即复现:
- 直调 kernel:OK;经 `@qd.func` 分发:OK;**经 `@qd.real_func`:错值**
  (两个 guard 体都被无条件执行,最后一个胜出)。
- 隔离进程首编译仍复现(非 fastcache/顺序假象);触发要素 = **real_func 调用者**
  × **early_return 重写过的叶**(直调/func 分发不触发;分支数无关,2 分支即可,
  6 分支纯 func 叶不触发)。

### 证据链

1. **编译产物对照**(QD_DUMP_IR=1 的最终 LLVM):工作形态(直调/func 分发)guard
   链 = **7 条 br i1 + 16 条 i1 store**(M2 守卫链完好);via-real_func 形态 =
   **1 条 br + 10 条 store**(守卫链塌缩成直排,last-wins)。
2. **函数 stage 的 Initial IR**(print_ir 抓取):disp 的 CHI **从刚建完起就没有
   if**——两个 guard 的条件(`not done`、`pnt[0] > -100` 比较)完全没有进 IR,
   guard 体无条件直排;这排除了"优化器删分支",**病灶在 Python AST → CHI 构建段**
   (real_func 体 walk)。
3. **Python 层 tracer**(monkeypatch build_If):build_If 以正确条件被调用、
   test.ptr 为运行时 Expr、方法走完正常路径(begin_frontend_if 已调用)——
   **Python 层两形态行为一致**;分歧发生在 begin_frontend_if 的发射物与
   real_func IR 物化之间(builder 上下文/流归属)。
4. **K 的 CHI 干净**:外层 kernel 无叶 guard 泄漏(排除"泄漏进 caller IR"假设)。

### 工作假设(下一步验证)

real_func 体编译(`_transform_as_kernel` 路径,`FunctionDefTransformer:577`)下,
**内联 func 叶的 `begin_frontend_if` 发射进了与体语句不同的 builder/块上下文**
(条件语句流丢失;体赋值语句流存活——故症状 = "条件全丢、体直排")。注意
A16(loop 变体,登记在案)与此同域:**func→real_func 内联的控制流(loop/if)在
real_func 体构建中静默损坏**——A16 只登记过 loop,本线补 if 变体,两者大概率
同一修复点。

### 下一步(开发序)

1. lldb + 重链 DWARF .so(dev_notes §8.5 recipe)在 `begin_frontend_if`/
   `insert` 上断点,对照 direct/via_r 两形态的 builder 身份与落点块;
   或临时 C++ 打印(real_func 体编译期,单文件增量重编已验证 ~10–20 分钟)。
2. 修复落点按定位结果:预期在 real_func 体构建的 builder 上下文交接
   (`Func.__call__` real_function 路径 / `_func_base.get_tree_and_ctx` 的
   ast_builder 归属 / C++ ASTBuilder 的块栈)。
3. A44 的 plain-tuple-func 变体(box4b)与 A33 返回方向在修复后回归验证
   (同族不同形态,可能需要第二处修复)。

## 验证方案 / DoD

- 探针矩阵全绿:`fr_min_er_leaf.py`、`fr_min_er_ctrl.py`(三形态对照)、
  `fr_iso_via_r.py`、`ray_boundary_probe.py` P1–P4、`box4_probe.py`、A33 五探针链。
- 新 gate:`probe_realfunc_composite.py`(CPU-only)进 union runner(TOTAL→21):
  矩阵 = 直调/func 分发/real_func 分发 × early_return 叶/普通叶。
- test_return 46/46;union runner 全绿;port 侧绕行退役验证
  (`_ray_geom_mesh` 还原分发、box4b 形态还原,独立 port commit)。
- bvh-interop 遗留 2F(rc 路径 hfield/mesh BVH)翻绿。

## footprint 预算 / 白名单

- 预期修复 = real_func 体构建的 builder 上下文交接(预计 ≤50 行);gate 20 行 zsh +
  探针;禁顺手重构。
- 白名单:`python/quadrants/lang/_func_base.py` 或 `kernel_impl.py` 或 C++ 前端
  builder(以定位为准,≤2 文件);`tests/robostudio/run_robostudio_gates.sh`;
  本任务书;ledger。

## 归因里程碑 2(2026-10-19 深夜,树级证据闭合)

- **调试 wheel + 组合抓取**(QD_FR_DEBUG 门控 begin_frontend_if/expr_alloca 追踪
  + print_ir 同跑):guard 的 begin_frontend_if 落在 **tree root=0xacc008c00**;
  而 disp 实际编译的树 = **root=0xacc008500**(compile_function 打印,相邻的
  两棵兄弟树)。**guard 的 FrontendIfStmt 进入了从未编译的死树;disp 的编译树
  里只有被拍平的 guard 体赋值** = 直排 last-wins 错值的完整解释。
- **附带裁定**:①A44 的 plain-tuple-func 最小收缩(单 tuple func + real_func 体)
  **不复现**(body(3)=10 ✓)——box4b 需更多配料,待主修复后迭代;②A43 整体
  **翻案为假登记**(dev112 串行 + numpy 孪生 want 下四变体全绿;原探针踩顶层
  for 并行语义 + want 错,详见 mujoqo 账本 A43 翻案小节)。
- **下一步(精确到行)**:①在 Func.__call__ plain 路径的
  `ast_builder=current_kernel.ast_builder()` 处打印 current_kernel 的身份
  (嵌套 do_compile 时它指向外层 kernel 还是当前 real_func——本轮 GTX trace
  显示 leaf 的 builder id == do_compile 时 kernel 的 builder id,但
  current_kernel 名字提取失败未确证指向);②确认后修 = 嵌套 real_func 编译
  必须把 current_kernel/builder 归属切到当前 real_func(或叶的 ctx 直通其
  builder),并核对 scope 栈随 builder 切换的配对;③修复后跑完整验证方案。
- **A16 并系确认**:本缺陷 = A16(loop 变体)的 if 变体——func→real_func 内联
  控制流静默损坏的统一根因方向成立。

## 归因里程碑 3(2026-10-19 凌晨):A37 线 UAF 全面暴露阻塞验证 — 新最简复现入库

- **回归 battery 在 dev126/dev128 布局下大面积 SIGSEGV**(int2bool/flt2bool/nan2bool/
  V1/V2d 全崩 139;A43/A34 修正符号化崩溃栈 = 前端 walk 中
  `DataType::ptr_removed()` 解引用近空指针 0x8 —— 非编译期 assert、非 A37 旧
  dyn_cast 签名,是同一语句生命周期 UAF 族的另一表现)。
- **判别实验闭环**:dev112 布局 5/5 过;dev128 布局 5/5 崩;两 wheel 对
  int2bool(无 func 调用、无张量 store、20 行内核)的代码路径**等价** →
  崩溃与 wheel 内容补丁无关,纯 .so 布局决定的 UAF 暴露。**A37 线的
  "崩溃因果待暴露窗"已兑现为常态化暴露,阻塞 dev126+ 一切 wheel 级验证。**
- **价值**:int2bool(无 real_func、无 func 调用、无张量)成为 A37 线
  **迄今最简 UAF 复现**(旧最简 = a19 全套件)。A37 线修复入口更新:
  前端 walk 中某 stmt 的 ret_type 为空即被 ptr_removed 解引用——
  该空 ret_type 语句从何而来 = 语句生命周期根因的直入口。
- Family R 修复本身在 dev118 布局的验证矩阵(最小复现/控制矩阵/P1-P4/box4/
  A33 链/test_return)全部绿 ✓ 不受影响;dev126+ 布局的验证待 A37 线落地。

## 定性更正(2026-10-19 凌晨,dev134 复验)

- **int2bool/dev126-127 崩溃族 = patch v1 漏插 bug 本身**(dest MatrixPtrStmt 未入块
  → TypeCheck 读已释放语句),**非 A37 线 UAF 暴露常态化**——"dev126+ 布局阻塞"
  的前判作废:patch v2 补插后 dev134 上 int2bool 3/3 绿、union runner **20/20**
  (gate 14 x3 新钉值 + gate 20 首跑)。A37 线回归原状态:仅 a19 宿主 x3 防闪烁
  在案(gate 14 已内建),无新增阻塞。
- **A42 残余失败精确定位**:V2/V2c = 前端**值侧链式下标**(`v[0,0][0]`:张量值
  语句的分量读)降为 `MatrixPtrStmt(origin=值语句)` → 构造期 QD_ERROR
  (statements.cpp:132 origin 类拒绝)。修复方向 = 前端值侧分量读改 GetElement
  或 alloca 停泊(make_tensor_access 的非 lvalue 停泊逻辑已存在,值侧
  IndexExpression 路径未走它)。下一 fix 线入口明确。

## 收口(2026-10-19 深夜)

- Family R 修复(func.py builder 解析)在 dev145/dev148 双 wheel 验证:
  最小复现/控制矩阵/P1-P4/box4b/A33 链/test_return 46/46/union 21/21 全绿。
- A42:scalarize 泛化(v1→v3)落地,V2/V2c 翻绿(见 A42 里程碑 3);
  alias 索引数断言以保守 uncertain 修复(94f99959)。
- A37 线定性更正:dev126/127 崩溃族 = patch v1 漏插(已修),非 UAF 暴露。
- bvh 2F:根因定位完成并移交 bvh-interop 线(最小复现 + mj 仲裁,
  `_probe/bvh_geomid_min.py`)——BVH(rc) 路径错过 mesh 相交穿落 plane,
  brute 与 mj 逐位一致;修复入口 = ray_mesh_with_bvh / bvh_query_ray_next
  的 mesh-local 变换与遍历语义。
