# A37 线任务书 — compile_kernel 内非确定性 UAF(dyn_cast 读悬垂指针)

- 线名:`mujocoqd/a37-compile-uaf`(worktree `~/Code/quadrants-a37wt`,从
  robostudio-main HEAD 03ad496b 切出)。
- 账:mujoqo `docs/quadrants-需求清单.md` A37;评估成文 mujoqo
  `docs/qd-A37-编译器UAF-需求评估.md`(第一性原理评估通过,2026-10-18 用户预授权开工)。
- 探针:mujoqo `_probe/a37_uaf_census.py`(census 包装器,基线 6/6 SIGSEGV @ 高暴露
  uv-python;DoD oracle = 修复后 N=10 STABLE)。

## 判决

按评估文档方案 A 执行:**定位先行(ASAN/-g 调试轮),定点修生命周期**,不做防御性
重构(去 RTTI=C 案,否决)、不做 port 侧绕行(D 案,否决)。

阶段化(阶段门):
1. **阶段 1(-g 调试构建)**:`CMAKE_ARGS="-DCMAKE_BUILD_TYPE=RelWithDebInfo
   -DCMAKE_CXX_FLAGS=-g\ -O1 -DQD_WITH_CUDA:BOOL=OFF"`(构建 log 的 C++ Flags 行必须
   出现 `-g`——RelWithDebInfo 不粘性已知坑,先验证再继续)→ census 6×:
   - 若崩(预期布局仍暴露或部分暴露)→ lldb 符号化 bt(调用链 11 帧匿名 → 全名)+
     被转型对象的类型(fault 现场 x0/x1/x2/dt)→ 判定点 pass → 进阶段 3;
   - 若不崩(布局变,失去复现)→ 阶段 2。
2. **阶段 2(ASAN 构建)**:同上加 `-fsanitize=address`(ASAN quarantine 使 UAF 与
   布局无关地必报);macOS python 扩展加载 ASAN 运行时需
   `DYLD_INSERT_LIBRARIES=<clang resource>/lib/darwin/libclang_rt.asan_arm64e.dylib`
   (按实际 resource dir 取)→ census → ASAN 报告给出 free 栈/use 栈三要素。
3. **阶段 3(定点修)**:按三要素(悬垂节点类/释放点/使用点)修所有权或时序。
   候选形态(先验排序):改写 pass 持有的节点指针跨释放事件 / scope 退出顺序 /
   错误路径清理释放仍被引用节点(A21/A22 都在错误路径)。修法原则:**最小所有权修复**,
   禁顺手重构 RTTI 分发(C 案边界)。

**反证判据**(任一出现即方案错,回滚重评):
1. ASAN 报告显示的"释放"来自第三方/LLVM 静态库(未插桩,三要素不全)→ 转方案 B
   (watchpoint 溯源,+0.5-1d);
2. 定位点显示缺陷在移植侧 kernel(非编译器)→ 重归因,A37 改判;
3. 修复后 census 仍有崩(多释放点)→ 扩大审计,不掩盖。

## 验证方案

1. census:`_probe/a37_uaf_census.py -n 10` @ 高暴露 uv-python venv → **STABLE(0 崩)**;
   低暴露构建(homebrew-python)复核 10×0 崩。
2. upstream `pytest tests/python/test_return.py` 46/46;union runner 全绿(19/19,
   gate 14 此时是真绿)。
3. gate 14 去闪烁(独立小 commit):runner 内 gate 14 改为 N=3 连跑一致性(任一
   SEGV=FAIL)——先于修复落地,修复后仍保留(防回归面)。
4. 全量 cpu 重跑基线不回退(A32/A34 形态不回归)。

## footprint 预算

定位阶段零代码;修复预计 <50 行 C++(单点所有权/时序);gate runner 改动 ~10 行
zsh。调试构建产物不进常规 venv、不分发。禁顺手重构。

## DoD

- [ ] census N=10 STABLE(高暴露)+ 低暴露复核;
- [ ] test_return 46/46;union runner 全绿;gate 14 去闪烁落地;
- [ ] 全量 cpu 重跑基线不回退;
- [ ] 线 commit(英文 subject,无 co-author)+ ledger 条目(A37 根因+修复+数字)。

## 白名单

编译器修复目标文件(阶段 3 定位后确定,≤2 个 .cpp/.h);`tests/robostudio/
run_robostudio_gates.sh`(gate 14 去闪烁,独立 commit);本任务书;ledger 条目。
其余发现只报不改。
