# qd_compiler_dev_notes.md — 编译器源码开发经验汇编(源码地图 / 修复模式 / 调试工具箱)

> 日期:2026-10-11。定位:给在 quadrants 源码上做开发的工程师(包括未来的自己)
> 一份"从哪下手、怎么定位、修完怎么守"的实操手册。素材来源:
> ①robostudio-main 五个已落地修复族的完整谱系(ledger + consolidation);
> ②mujoco-qd 移植 P0 验证期新发现的三个待查缺陷(real_func SPIR-V / tile
> runtime-loop / build_Return 元组注解,复现脚本见文末);③qd-render-poc 与
> parallel-accel 两条 kernel 线的反向经验(前端语义契约部分)。
> 操作性构建/安装/门禁配方见 `docs/build_robostudio.md`,分支谱系见
> `docs/consolidation_robostudio_main.md`,功能台账见 `docs/robostudio_features.md`。

---

## 1. 源码地图(改动最常落的文件,全部经实战验证)

### 1.1 Python 前端(`python/quadrants/lang/`)

| 文件 | 职责 | 已验证的改动 |
|---|---|---|
| `ast/ast_transformer.py` | AST→IR 主变换器;`build_If`/`build_Return`/`build_Subscript` 等 per-node 方法 | S3:build_If 改 `scope_guard=False` 让 static 分支绑定落外层作用域(2d9a99ca);**待查:A4 build_Return:646 对元组注解 func 直接 `return_data[i]=...` 撞 tuple 不可变** |
| `ast/ast_transformer_utils.py` | 作用域/语句构建辅助 | S2:`register_loop_var` + `check_assign_not_loop_var`(循环变量赋值守卫,1d645464);build_stmts 签名加 scope_guard 形参 |
| `kernel_impl.py` | `@qd.kernel`/`@qd.real_func` 装饰器实现;real_func = 非内联真调用 lower(:39-96) | **待查:A1 real_func 在 Metal/SPIR-V 全路径崩(`spirv_ir_builder query_value: Value "tmp" does not yet exist`),CPU 正常——疑非内联调用在 SPIR-V 的参数编组/生命周期** |
| `kernel.py` | Kernel 类、launch、fastcache、graph=True、capture walk | C0S/union:cache-miss 捕获走查让位 S-guard 精确收口(60003393) |
| `impl.py` | `subscript` 等 IR 收口点 | S-guard:field 进 IR 唯一收口 `_check_not_stale`;C0S 配套清 materialized_kernels |
| `field.py` / `matrix.py` | Field/MatrixField 宿主对象 | S-guard:generation 戳 + 全宿主路径拒绝(b778ab20,64 行+4 测试) |
| `misc.py` | `loop_config`/`block_dim` 等 | (perf 待查:C1 block_dim 对串行内层循环无观测影响) |
| `simt/_tile.py` | Tile16x16/32x32 寄存器瓦片(工厂 `_make_tile_class`,subgroup shuffle 通信) | **待查:A2 tile 在运行时嵌套循环 ≥3 次外层迭代产生错误数值,静态展开同序列正确——疑寄存器行生命周期/φ 节点** |
| `_funcs.py` / `_funcs_sym_eig_general.py` | svd/eig/solve/randn 等 | solve 3×3 数值与 numpy 一致(mujoco-qd 探针验证) |

### 1.2 C++ 后端 / runtime(`quadrants/`)

| 文件 | 职责 | 已验证的改动 |
|---|---|---|
| `compilation_manager/kernel_compilation_manager.{cpp,h}` | 编译产物缓存(盘上 key、cache_epoch_) | C0S:cache_epoch_ 混入 key,随 SNodeTree 生命周期递增(6ea5ec4b) |
| `program/program.cpp` | Program 生命周期、add/destroy_snode_tree | C0S:epoch 递增钩子 |
| `python/quadrants/_snode/fields_builder.py` | field 出生登记 | C0S:`_qd_owner_runtime` 记录出生 runtime |
| `codegen/spirv/spirv_ir_builder.cpp` | SPIR-V IR 构建(Metal/Vulkan 共用) | **A1 崩溃点(query_value@1062)**;另见 f64 拒绝(get_primitive_type:335,硬件事实) |
| `rhi/metal/metal_device.{h,mm}` | Metal 设备层 | readback R2 直 map;Blit-pool:`MetalMemory::mapped_ptr` 加 QD_BLIT_DEBUG 诊断 |
| `runtime/gfx/runtime.{h,cpp}` | GfxRuntime、ext-arr 镜像池 | Blit-pool 根修:镜像无条件 host-visible(f4c57935);R3a 64MiB 尺寸键 first-fit 池 |
| `python/dlpack_funcs.cpp` | DLPack 导出 | u8/i8/u16/i16/u32/u64 类型矩阵补全(4b2e3c77) |
| `python/quadrants/_ndarray.py` | ndarray 宿主侧 | R3b per-call commit 模型文档化 |

## 2. 修复模式手册(五案提炼,每案 = 症状/根因/落点/可复用模式)

### S2 循环变量赋值守卫
- 症状:并行归约内核循环里写累加变量 → 未捕获 C++ 异常进程 abort(exit 134);另有晦涩 dest-type assert。
- 根因:LoopIndexStmt 被局部赋值 lower 成 `LocalStoreStmt(LoopIndexStmt)`;增广赋值形态直接裸异常。
- 模式:**在 AST 前端把非法形态变成响亮 SyntaxError**(`register_loop_var` 登记 + 赋值检查),ndrange/grouped(alloca-backed)豁免。45 行纯 Python。教训:坏的报错路径比缺功能更贵——未捕获 C++ 异常杀死进程让上游归因走了弯路。

### S3 static-if 分支名 join
- 症状:`if qd.static(...)` 分支内绑定,join 后使用报 `Name "y" is not defined`。
- 根因:build_stmts 的 per-block scope 吞掉编译期绑定;static 分支语义上应落外层作用域。
- 模式:**区分编译期与运行期的作用域语义**——static 分支 `scope_guard=False` 在外层构建;运行时 if 的 alloca 仍留分支内。48 行纯 Python。附带翻案:bool 字面量/qd.min/max/sqrt 的历史"坑清单"多为该 bug 的显形,修复后应回头拆除规避代码。

### C0S 缓存纪元
- 症状:多 init/多 SNodeTree 家族下旧缓存复用 → 脏 field 绑定 → OOB/SIGSEGV。
- 根因:盘上缓存 key 不含 SNodeTree 生命周期;产物烘焙了布局字节偏移。
- 模式:**缓存 key 必须覆盖一切影响产物语义的生命周期状态**——`cache_epoch_` 混 key + 单 tree 常规程序永不 bump(保盘上 key 稳定)。配套 python 侧清 materialized_kernels。

### S-guard 陈旧 field 响亮拒绝
- 症状(夜探 Inv2 定案):`qd.init#2` finalize Program#1 并重置进程级 HostMemoryPool,存活 Field 带裸 SNode 指针 = use-after-free;历史"还能跑"是池内存未覆写的运气。
- 模式:**把"碰巧能跑"升级为"响亮拒绝"要诚实标注语义变更**——generation 戳 + 唯一收口点检查(`impl.subscript`)+ 粗粒度防御走查让位精确收口(次序冲突的 union 处理,60003393)。64 行+4 测试。契约变更(one-init-per-process)必须写进文档与 battery。

### Blit-pool 同质化
- 症状:粘性、按题确定、重试不可恢复的 `Failed to map ext arr data buffer for host_to_device blit`(cavity 189 题中 189 集中簇)。
- 根因:access==NONE 死参数拿到 Private 镜像 → 入池(池只按 size 键控)→ first-fit 发给需要 blit 的活参数 → `map()` 对 Private 永远 invalid_usage → 失败条目回池再被抓 = 毒化循环。
- 模式:**对象池的键控必须覆盖用途差异**(可 map 性是用途,不是尺寸);修法反而更简单——镜像无条件 host-visible。配套 `QD_BLIT_DEBUG` 诊断区分 can_map=false 与 contents nil、失败消息带 arg_id/size/alloc_id。**识别特征:随机 flake(~2.25%,backoff-retry 治)vs 粘性簇(确定性复现,retry 不救)的判别是处置分叉。**

## 3. 待查缺陷(mujoco-qd P0 新发现,均带可运行复现)

> 复现脚本在 `/Users/mini0/Code/mujoqo/_probe/`。**注意基线**:以下均在
> dev7+g7aaaa7d73 wheel 上发现,早于 S2/S3/S-guard/blit-pool 落地;M0 应先在
> dev25+ wheel 重跑确认仍然存在(尤其 A2,理论受 S2 影响概率低但未验)。

### A1 real_func Metal 全路径崩溃(高优)
- 复现:`_probe/p0_realfunc.py metal` → 0/5(同脚本 cpu 4/5)。最小用例:无循环、单 if 早退、标量返回。
- 报错:`spirv_ir_builder.cpp:query_value@1062 Value "tmp6" does not yet exist`。
- 已排除:语法(real_func 构建期可过)、CPU 路径(同代码全绿)、循环结构(无循环也崩)。
- 入手建议:`kernel_impl.py:39-96` 的非内联 lower 与 `spirv_ir_builder` 的函数参数/返回编组;上游 test_return.py 的 real_func 族标 `arch=[cpu,cuda,amdgpu]` 排除 metal,说明 SPIR-V 路径从未被走过。

### A2 Tile16x16 运行时嵌套循环 ≥3 次外层迭代数值错误(高优,正确性地雷)
- 复现:`_probe/p0_block_chol.py metal` → m=16/32 PASS(3e-7),m=48 第 3 块行 2e-2;**同一指令序列静态展开 m=48 九块全 PASS(1e-7)**(对照 numpy fp64 排除精度)。
- 疑点:`simt/_tile.py` 的寄存器行(`r0..rN`)在 runtime 循环回边上的生命周期/φ 处理;或 shuffle 重排与循环展开优化的交互。
- 危害:静默错数值、无报错;任何用户在 runtime 循环里用 tile 都会中招。短期缓解:文档 + 编译期检查(runtime 循环含 tile op 时警告/拒绝)。

### A4 带元组注解的 @qd.func 编译崩溃(低,已绕过)
- 复现:对 `-> tuple[qd.f32, qd.f32]` 的 @qd.func 任何 `return` → `ast_transformer.py:646 build_Return: 'tuple' object does not support item assignment`。无注解的元组字面量返回正常(两后端验证)。
- 修复方向:build_Return 的 return_data 用可变容器或局部副本;上游 test_return.py 同类测试也只标 cpu/cuda/amdgpu,疑同根源。

## 4. 调试工具箱(实战有效)

1. **QD_BLIT_DEBUG**(metal_device.mm):区分 can_map=false 与 contents nil——blit 类问题第一步。
2. **gate 先行**:修复落树前先写最小探针进 `tests/robostudio/`,union runner(`run_robostudio_gates.sh`)一条命令回归全族;新缺陷照此办理(建议 mujoco-qd 探针注册为独立 gate 家族)。
3. **双向红标记实验**(qd-render-poc §4.6):断言"环境坏了/缓存坏了"之前,注入显眼标记(如 `bi=137`)验证改动真的生效、还原后精确恢复——宿主侧脚本静默 no-op 是高频假信号。
4. **最小 repro 逐层剥离**:A1 的定位路径 = 先剥循环 → 再剥元组 → 再剥嵌套 func,每层 CPU/Metal 双跑。`python -c` 内不能定义 kernel,探针一律落盘文件。
5. **判别探针先修**(S2/S3 翻案教训):两个"上游误编译"实为探针 bug + 并行归约竞态。定罪上游前独立验算(OR 验算、f64-true 探针、逐位对照)。
6. **性能归因先分固定税与速率**(readback 方法论):launch/封送/桥接是固定税,memcpy/带宽是速率;对照"另一条已证可达的路"定验收线。

## 5. 前端语义契约(kernel 作者视角,文档化义务)

以下为两线实战沉淀的语言面契约,应反映进 user_guide(部分已隐含):

- kernel/@qd.func 运行时 if/for/while 内 **return 不支持**(单出口风格);real_func 可但见 A1;
- 运行时循环变量不可赋值(S2 守卫,响亮报错);ndrange/grouped 目标可赋值;
- static-if 分支绑定 join 外层(S3 修复后行为);**运行时 if 分支内新变量不可跨分支使用**(顶层先声明);
- kernel 参数不可重新赋值;func 矩阵/struct 按值;func 可收 ndarray 但**公共逻辑收 ndarray 参数时注意模板爆炸**(qd-render-poc 实践:受控复制);
- PEP 563(`from __future__ import annotations`)下装饰期签名解析只认模块级名字(`eval_str=True` 对 `func.__globals__`)——dtype 工厂模式须模块级重绑注解别名;
- Metal 面硬约束:无 f64(SPIR-V 拒绝)、无 i64/u64 buffer/原子、无 print/assert、越界检查静默禁用;
- 编译律:`qd.static(range(N))` 硬展开随 N 平方级编译爆炸(T25:927 球 >6000s→0.2s),tile 运行时循环另有 A2 未决——**tile + 循环的组合当前只信静态展开**。

## 6. 工作流纪律(分支既定规矩,新改动照办)

1. feature commit 与 `docs/robostudio_features.md` 账本**同 merge 更新**,不一致 = 合并不完整;
2. 每个 family 落地必须带 gate 进 `tests/robostudio/`,union runner 要求全绿;
3. 上游关系 = 直修不提 PR(T20 裁决);overlay 树已退役,禁止再打 overlay;
4. wheel 全部从本分支构建(配方 `docs/build_robostudio.md`,RelWithDebInfo + `-DQD_WITH_CUDA:BOOL=OFF`——**CUDA 从未验证,对外引用数字须带此边界**);
5. 对外引用任何数字带验证域(metal macOS arm64 + cpu linux aarch64 容器)。

## 7. C2 向量化战役新地雷/新模式(2026-10-11,mujocoqd/spirv-vectorize)

1. **SPIRV-Cross 对 vec4 型 PhysicalStorageBuffer 指针的 OpPtrAccessChain 按 1(字节)缩放**
   (实测;标量 float 指针走数组索引按 4 正确缩放)。vec4 访问必须显式算字节地址 +
   OpConvertUToPtr;别依赖 PtrAccessChain 的元素缩放。MSL 形态:
   `*(device float4*)(args._mN._m1 + ulong(idx << 4))` ✓。
2. **Apple MSL fast-math 会重结合"裸露的累加器链"**:`(acc - p0) - p1 - p2 - p3` 与
   `acc + (-p0) + (-p1)…`(以及 `acc + x*c` 常数因子形态)都会被重排,实测 1–2 ulp 漂移;
   而**循环携带链**(runtime 界的内层循环)与逐元素向量运算(saxpy)保持程序序。
   对策 = 把 lane 归约放进 runtime 界内层循环(OpVectorExtractDynamic 动态 lane),
   编译器无法证明 trip count → 不展开 → 序保持 → bitwise。纯 add/mul-over-load 链
   (物理热点形状)实测静态展开也 bitwise,可走快路径。
3. **前端把 `s -= X` 降低为 `s = s + neg(X)`**(显式 UnaryOp neg 节点 + add)——做归约
   分析时别假设 binop=sub 存在;neg 节点会出现在乘积之上。
4. **Metal ndarray 镜像来自子分配池,data_ptr 不保证 16B 对齐**:vec4 访问的对齐判定必须
   用**绝对地址**(data_ptr 低 4 位 + 元素偏移)算 residue,只看元素索引会在部分分配布局下
   产生硬件 misaligned float4 慢路径(实测症状:同进程内 4× 吞吐抖动)。
5. **实测墙:bench_physics 形状在 M4 上是 DRAM 字宽瓶颈**(dot 2load/el 与 4×flops 版同速、
   半字节版半速;~63 GB/s ≈ 同进程 copy 带宽)。codegen 改动若不减字节搬运量,大尺寸无
   收益;小尺寸吃 dispatch 税。测 perf 前先跑 copy 定墙,再同轮 A/B(当日机器态抖动 ±4×,
   单轮定论必事故)。
6. **逐位 A/B 的正确姿势**:one-init-per-process ⇒ 双臂必须子进程;hash 对照 ndarray
   bytes;覆盖 nv ∈ {1,3,5,23,24,25}(tiny/remainder/跨 arg class 不一致/对齐 prologue)。
   参考实现:tests/robostudio/probe_spirv_vectorize.py。
7. **快路径 lane 析因陷阱(c2-emitter-fixes ①)**:还原累加 DAG 的叶子若含**循环不变
   uniform**(被 cache-loop-invariant-global-vars 上提出循环的 load / 体内 coeff-0 load /
   标量 kernel arg),Apple MSL fast-math 会把四条 lane 积和析因重排为
   `(x0+x1+x2+x3)*c`——§7.2 "acc + x*c 常数因子形态"的 uniform 叶子特化,实测
   `s += x[j]*c` 漂 ~2 ulp。对策 = fast 资格收紧到"体内 stride-1 load 叶子",其余回落
   order-barrier lane 环(参考 vec_analyze fast_dag)。
8. **GlobalStoreStmt 的 ret_type 恒为 unknown(c2-emitter-fixes ③)**:store 不产值,
   `stmt->element_type()` 对它无意义;dtype 判定要走 ext-ptr pointee
   (`ep->element_type().ptr_removed()`)。误用会以 "non-f32 memory access" 的误导性
   理由拒掉**所有**含体内 store 的可向量化循环(症状:只有还原循环向量化)。
9. **C2 三遍发射的名字作用域(c2-emitter-fixes ②)**:同一循环体被 scalar prologue /
   vector main / scalar remainder 走三遍,每遍体内语句重发射并按 raw_name 重注册——
   每遍都必须包 push/pop_value_scope(prologue 与 vector walk 均已包),漏包则
   `register_value "tmpNN already exists"`。循环后按名查询的只有 alloca 与任务级值,
   注册于 push 之前,不受 pop 影响。
