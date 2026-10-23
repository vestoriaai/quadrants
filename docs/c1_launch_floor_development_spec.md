# C1 launch floor 开发任务书(qd 侧 C1a + C1.5)

2026-10-22 立项。上游依据:mujoqo `docs/qd-launchtiled-fusion-需求评估.md`
(§0-§15,多轮第一性原理 review,用户逐轮质询后定稿)。本任务书覆盖 **C1a
(Python 命中路径短路)+ C1.5(小 work 自适应串行)**;C1b(plan)/C2(回放)
另立任务书,本线的接口决策须预留其消费者位置。

## 1. 判决(问题定义)

- **C1a**:qd 每次 kernel launch 的 Python 命中路径 ~15.6µs(trivial 双参数
  实测),其中 ~13.3µs 在 `quadrants/lang/kernel.py:694 launch_kernel` 体内
  (cProfile 实证);warp 同语义 3.3µs(context.py:10913 invoke 用类型元组键
  缓存 ctypes Struct + 函数指针直调)。成本随参数量线性(求解核 15-30 参
  53µs/次)。mujoqo S6 一步 133 次 launch,Python 残余 ~1.3ms。
- **C1.5**:qd kernel CPU 并行域发射的 barrier 开销主导小 work 执行——同一
  kernel 串行 vs 并行实测 27 元素 2.53µs vs 16.9µs(6.7×)、4096 元素 7.05µs
  vs 23.5µs(3.3×)(mujoqo _probe/perf_serial_vs_parallel.py)。mujoqo S6
  C++ 段 2.0ms 的大头是 barrier 而非计算(mj C 同步全链 16µs 交叉印证)。

## 2. 方案

### C1a:launch 命中路径短路

- C0 行级归因先行(cProfile 行级 + time.perf_counter 分段),产出
  **分支-键覆盖表**:launch_kernel + Kernel.__call__ 每个内部分支
  (args_hash 构建 / mutable-ndarray guard 链 / populate_launch_ctx_from_cache /
  launch_ctx.copy / ensure_compiled / fuse_args / current_cfg / optional-None
  跳过 / all-field shortcut / graph_do_while / checkpoint 块)→ 各自耗时
  → 由键的哪个维度决定。该表是实现的直接依据,也是 C2 任务书输入。
- 归因数据决定短路点组合(预期候选,以实测为准):
  ①args_hash 构建:guard 判定为"本 kernel 无 struct ndarray 参数"时跳过
  逐参 id() 元组(kernel 级静态事实,AST/首次解析时已知);
  ②populate 命中后的 `launch_ctx.copy(cached)` 换成预绑定闭包直呼;
  ③Kernel.__call__ 层的 ensure_compiled/fuse_args/current_cfg 在
  "已编译+无 fuse"常见形态短路。
- **红线**:stale guard 语义原样保留——非 stable_members 的 dataclass struct
  嵌 ndarray 形态永远走现有 guard 链(慢路径是语义权威);`stable_members=True`
  承诺形态可跳 guard。死亡回调(clearing callback)必须保留并同样挂到任何
  新缓存上(id 复用 = 错值,硬要求)。offline cache 键不变。
- 反证判据:任何分叉矩阵探针失败(类型切换/struct 重绑/数组死亡后重分配/
  config 切换/多线程)即该短路点回退。

### C1.5:小 work 自适应串行(CPU 专属)

- runtime 侧:host launch 时按 total work(< 阈值)自动走串行执行路径
  (等价 `qd.loop_config(serialize=True)` 的 runtime 化)。
- 阈值探针矩阵先行:kernel 计算密度分档(纯搬运/轻算术/重算术)× work 档
  (1e1/1e2/1e3/1e4/1e5)× 串并行,产出阈值表;默认阈值保守(只覆盖明确
  串优区间),暴露 `QD_AUTO_SERIAL`(默认开,CPU;metal 直通不适用)与
  `QD_AUTO_SERIAL_THRESHOLD` 可调。
- 浮点求和顺序:同 kernel 顶层 `for __t in range(n)` 串行化改变调度不改
  单线程结果语义(现有 serialize 路径已被 genesis-world/移植代码使用);
  gate 逐项跑确认零数值漂移(联立 A47 位级账,任何 diff 单独归因)。
- 反证判据:任一 gate 数值 diff(非 A47 已知集)或 battery 新增失败即回退
  阈值或关闭默认。

## 3. footprint 预算

- python/quadrants/lang/kernel.py(launch_kernel/__call__ 热路径)+ 可能的
  _template_mapper_hotpath.py;新文件仅限 `_launch_plan.py`(若闭包化需要);
- C1.5:runtime host launch 入口(python/quadrants/lang/impl.py 或 kernel_impl.py
  的发射侧)+ 阈值配置;
- 白名单外文件零接触;C++ 侧本任务书**不改**(C++ 归因若显示必要,另立)。

## 4. DoD(验收)

1. **性能**:trivial 双参 launch ≤8µs(实测基线 15.6);30 参 kernel ≤15µs
   (基线 53,参数量线性项压平);C1.5:27 元素核发射+执行 ≤6µs(基线 16.9);
   mujoqo S6 step ≤2.0ms(现 3.32,容许 C1a/C1.5 单独交付的中间水位 2.6);
2. **正确性门禁**:qd union runner 21 门全过;`QD_IR_AUDIT=1` 触达面;
   tests/python battery 失败集与 dev134 基线逐条一致(零新增);分叉矩阵
   探针(5 维)全过;ruff + clang-tidy(改动 .cpp 若有);
3. **跨平台**:metal green 子集(history/ray/sensor)前后全绿;metal battery
   失败集零新增;C1.5 对 metal 直通无影响;
4. **下游**:重烤 wheel → mujoqo 两个 venv 装入 → 全量 cpu `_src/*_test.py`
   失败集 = 既有挂账(step24/ray-tie/gjk-horizon)零新增;benchmark3way
   S1-S6 1000 步 parity 与基线逐位一致;S6 水位达 DoD-1。
5. **留痕**:本 spec 的归因/阈值数据、探针(进 mujoqo _probe/ 或 quadrants
   _probe/),ledger 条目(docs/robostudio_features.md 四字段)。

## 5. 工作量与顺序

| 步骤 | 内容 | 估算 |
|---|---|---|
| S0 | C0 行级归因 + 分支-键覆盖表 + 阈值探针矩阵 | 0.5-1 天 |
| S1 | C1a 短路实现 + 分叉矩阵探针 | 1-2 天 |
| S2 | C1.5 auto-serial + 阈值标定 | 1-2 天 |
| S3 | gates 全套(battery/metal 子集/union)+ wheel + mujoqo 下游全量 | 1 天 |
| 合计 | C1a+C1.5 | **3.5-6 天** |
| (后续线) | C1b plan 2-3 天;C2 回放 3-5 天(另立任务书) | — |

## 6. 风险

1. 13.3µs 中 C++ 边界占比未知——若 C++ 侧 ≥8µs,C1a 天花板 ~50%,需在 S0
   后重估(C++ 侧另立);
2. guard 链短路对 stable_members 语义依赖用户标注的正确性——误标注 = 错值,
   文档必须醒目(与现有 `stable_members` 语义一致,非新增概念);
3. auto-serial 与 host 多线程的交互:线程池被并发 kernel 共享时阈值决策需
   线程安全(决策是只读比较,风险低,验证覆盖);
4. C1a 热路径是全库必经——任何回归都是全局性;缓解 = 分叉矩阵 + battery
   对照 + mujoqo 全量,任何一处不过即回退该短路点。

## 7. S0 归因修正(2026-10-23,开工后首轮实测——推翻 §1 部分判据)

源码插桩(kernel.py launch_kernel 体内分段计时)推翻 cProfile 误读:

| 形态 | wall | C++ 段(prog.launch_kernel) | Python 全部 |
|---|---|---|---|
| 并行(kernel 默认) | 17.6µs | **13.33µs** | 4.3µs |
| 串行(loop_config serialize) | **2.64µs** | **0.19µs** | 2.45µs |

- 13.3µs 不是"Python 体"(cProfile 把 nanobind 调用时间计入调用者 tottime),
  而是 **ThreadPool wake-all + master_cv 等待的 barrier**(ThreadPool::run,
  mutex/condvar 往返);
- **C1a 天花板 ~16%**,降级为顺手项;**C1.5 升为主力**;
- C1.5 落地层:`ThreadPool::run` 开头串行退化(splits ≤ QD_AUTO_SERIAL_
  THRESHOLD,默认 64;QD_AUTO_SERIAL=0 关闭)——单点 C++ 改动全库生效;
- 语义安全论证:task_fn 每 task 自足(TLS 栈上局部、prologue/epilogue 每
  task 完整、range-for 无跨 task 同步);block-tier(simt.block)自建
  std::thread 不走本池,不受影响。

**Gate 结果(2026-10-23,dev136+gb2b318c1 wheel)**:
- floor:27 元素核 5.43µs(开)vs 16.25µs(关)——3×;
- mujoqo 下游:solver_test 63P/1S;S6 1000 步 **2.95→1.965ms**;benchmark
  S1-S6 mjc 锚定语义指标与基线逐位同值(零回归);
- 确定性:同环境 run-to-run 逐位一致;AS=0/AS=1 1001 步逐位一致(C1.5 零
  数值影响);dev134 vs dev136 跨 wheel 在 step41 起 4.6e-05 差(工具链
  rebase 位级差异,A47 族,语义口径验收);
- qd gates:return 26P;threading/serial/offload_cross/global_thread_idx
  12P/1S;kernel_coverage 等 183P/79S/2F——2F(test_atomic_add_global_i32/
  f32)在 dev134 同挂,既有挂账(TODO battery 分类③),零新增;
- 待办:全量 battery(协调人收尾跑)+ metal 子集。
