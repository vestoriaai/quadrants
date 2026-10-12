---
name: qd-compiler
description: quadrants(Taichi fork)编译器/运行时源码开发的能力速查与纪律:源码地图、A 系列已知 bug 家族、修复模式、构建/gate/烤 wheel 流程。改 codegen、修编译器 bug、排查"同一代码 cpu 过 metal 错"类问题、或给 quadrants 加 API 时使用。
---

# quadrants 编译器开发速查

本 skill 是索引 + 高频结论;机理细节查权威文档,不要凭记忆扩展。

## 权威文档

1. `docs/qd_compiler_dev_notes.md` — 源码地图、修复模式、§6/§7 地雷与案例
2. `docs/robostudio_features.md` — ledger:每条已合入线的设计/数字/footprint(附录 A 系列证据索引)
3. `docs/build_robostudio.md` — 构建/烤 wheel/venv 角色
4. mujoqo 侧 `docs/quadrants-需求清单.md` — A1-A15 bug 账(带复现指针)

## 源码地图(高频触点)

- **SPIR-V 后端(metal/vulkan)**:`quadrants/codegen/spirv/spirv_codegen.cpp`(C2 向量化段 vec_analyze/try_emit_vectorized_range_for/vec_emit_*;C2-emitter 修复段 env_lookup/dtype-pointee/push-pop scope)、`spirv_ir_builder.{h,cpp}`、`detail/spirv_codegen.h`
- **CPU/LLVM**:`quadrants/codegen/llvm/codegen_llvm.cpp`(SharedArray cpu 分支)、`quadrants/codegen/cpu/`、`runtime.cpp`(cpu_block_parallel_for/代际屏障)
- **前端**:`python/quadrants/lang/ast/ast_transformers/`(early_return_transformer = M2)、`kernel_impl.py`、`_ndarray.py`
- **API 面**:`program/compile_config.h`(旗标)+ `python/export_lang.cpp`(导出)+ `analysis/offline_cache_util.cpp`(**新旗标必须进 cache key**)+ `python/quadrants/lang/impl.py`(qd.copy 等原生便利函数)
- **algorithms**:`python/quadrants/algorithms/`(cpu 路径 `_cpu_paths.py`;serialize 逐循环铁律见地雷)

## A 系列 bug 家族速查(修复/规避已在主线,详见需求清单)

- **A5/A11/A12(Apple MSL loop-carried 族)**:条件 RMW 用 `+=`;原子写后消费放下一个 launch;串行嵌套循环 store 用 atomic。归因 = 独立 MSL harness(`_probe/a5_runner.m` 模式)。
- **A10(自家 codegen)**:嵌套 loop-carried 循环**一律行序(i-outer)**,列序双后端 nan——codegen 侧待修(下一 native 线首项)。
- **A2**:Tile 跨 runtime 回边错值(钉死)→ 静态展开;A1:real_func SPIR-V 响亮拒绝(降级设计);A4:tuple 注解已修;A6:func 内 for 内联=并行 → serialize 逐循环;A7:early_return 复合返回手工;A9/A14:fuse_args 工厂崩 → 内联;A13:链式写=误报(已实证);A15:alias assert=上下文特异(smooth wrap 场景)。
- **C2 emitter 三缺陷已修**(env_lookup/dtype-pointee/scope),既有 know-how 在 dev_notes §7.7-7.9。

## 修复模式(经验证)

1. **先最小复现进 `_probe/`(断言生效),再动手**;dual-wheel(dev36 vs 当前)归因先行。
2. env 缺条目 → `env_lookup` 按名复用任务级值(SSA def-before-use 保证安全);**不许 try/except 吞**。
3. dtype 判定走 ext-ptr pointee(load/store 通载),不是 stmt ret_type(store 无值)。
4. 递归 walk 的注册用 push/pop_value_scope 配对(prologue 既有模式)。
5. 旗标 OFF 路径必须逐位不变(union gate 7/9 是硬门);新旗标三件套 = compile_config.h + export_lang.cpp + offline_cache_util.cpp。
6. batch 改写:全 parse 过再统一 write(带病落盘 = 返工);探针必须断言。

## 流程

- 构建/wheel:`MAKEFLAGS=-j8 NO_PROXY="*" .venv-build/bin/python build.py wheel`(sccache;python-only 分钟级)→ 装**自建验证 venv**(venv 归属见 mujoqo memory;基线 venv3=dev36 冻结,用于 dual-wheel 归因)。
- 门禁:`zsh tests/robostudio/run_robostudio_gates.sh <venv>`(门数以 runner 头注为准)+ `pytest tests/python/test_return.py`(46/46)。gate-4 perf 钉值 WARN-only(机器方差 ±4×,perf 结论交错中位,`_gpu_busy.sh` 查空闲)。
- 新 native 旗标/大改:ledger 条目(设计/数字/footprint)+ `--no-ff` 合并由协调人执行;上游 test_return 无回归是底线。
- Metal 无 f64/i64/u64/print/assert;3-arg range 不支持;静态展开平方爆炸(有界);池指针 4B 对齐;S-guard 一进程一 init。
