# A24 Phase 1 cfg 阈值让位线任务书

> 依据:mujoqo docs/qd-A24-A46-开发文档.md §1.4(定位修订:内存墙已消失,
> 本线 = 防御 cfg-ON 大 kernel 的编译时间/swap 爆炸——实测套件级 ON 跑把用户
> 磁盘经 swap 吃掉 10+GB)。

## 机制
`CompileConfig::cfg_auto_skip_threshold`(默认 150000,0 = 禁用):full_simplify
的 cfg pass 调用点先数语句,超阈值 → QD_WARN 点名 kernel 并跳过。cfg pass 数据
结构随 IR 规模超线性,大 kernel 上主导编译时间与内存;跳过仅降低优化强度,语义
不变。旗标三件套齐:compile_config.h/cpp + export_lang.cpp + offline_cache_util
cache key(跳过改变 codegen,必须入键)。

## 验收(全过)
- 单元三臂:threshold=50 → WARN 命中 + 结果正确(478.783386);threshold=0 →
  无 WARN 同结果;默认 150000 → 无 WARN 同结果(三臂逐位一致)。
- T1-T5:union 21/21 × {默认,audit}、test_return 46/46、ruff 归零、clang-tidy
  4 改动文件 0 repo warnings。
