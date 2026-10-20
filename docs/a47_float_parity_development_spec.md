# A47 位级浮点 parity 线任务书(mujocoqd/a47-float-parity)

日期:2026-10-20 · 基线:robostudio-main @ 8ae3bfe4 · worktree:~/Code/quadrants-a47wt
账本:mujoqo docs/quadrants-需求清单.md A47;总序:mujoqo docs/TODO.md 八段五线后的
qd 遗留修复程序(A47 → flex SIGSEGV → A31 → A30 残留 → A45/A26)。

## 1. 需求

collision_gjk_test 残留 3F(同族,GJK/EPA 位级浮点奇偶差):
- **box_box_max**:上游(warp)EPA 16 迭代收敛,qd 17+ → dist 漂移 4.2e-7(places=7 断言挂);
- **box_box_max2**:qd GJK 判分离(+1.3928e-6 > tol 1e-6 → ncon=1),上游判穿透
  (-4.9375e-5 → EPA → ncon=2);断言 want dist=-4.9374998e-05(MJC 64-bit);
- **box_box_horizon**:溢出位未置(qd EPA 路径略异,未耗尽 5 棱 horizon 缓冲)。
验收:三测试转绿(qd 与上游 mujoco_warp 的 f32 逐位 parity 恢复);全量门禁不回归。

## 2. 归因现状(本会话首轮;复现件已断言生效)

- **3F 复现**:dev122+g8ae3bfe4,`pytest collision_gjk_test.py -k "box_box_max or
  box_box_horizon"` → 3 failed,22s。
- **机制锁定 = `config.fast_math`(默认 true)的 FMF**:codegen_llvm.cpp:3181
  `initialize_context` 在 fast_math 下给 IRBuilder 设 `reassoc ninf nsz afn`
  (无 contract)。最小复现 `_probe/a47_reassoc_repro.py`(gjk.py:437 的重心三项和
  `x = l0*s2 + l1*s3 + l2*s4`,半 ulp 消抵值):
  - fast_math=OFF:qd == numpy L2R == **warp 参照**(L2R 逐位,双 case 验证);
  - fast_math=ON(默认):qd 偏离(异常形态见下);
  - 单语句形态才触发,语句化(t0/t1/t2 分名)后两 arm 都正确;
  - llc -O2 对 dump 的 ON-IR 输出 `A + (B + C)` 重排(确认 codegen 级重结合)。
- **parity 目标钉死**:warp CPU 对同表达式 = 严格 L2R(无重结合/无 FMA),
  `_probe/a47_warp_ref.py`(/tmp 副本待迁 _probe/)。
- **未解异常(当前深潜点)**:anomalous 值组(l0=0,l1=l2=1,s2=1,s3=s4=2⁻²⁴)下
  任何 fp 正确求值都 = 2u;qd-ON 却稳定返回 u(3/3,单项存活)——**代数不可能,
  疑似 JIT 路径与 llc 不等价**或更深缺陷;QD_DUMP_IR 的 IR 与 llc 复算均正确。
  下一步:lldb attach sleeping 进程反汇编 ORC JIT 的真实机器码;`llvm_context.cpp`
  侧查 kernel 模块是否有额外 opt 步骤。
- **排除项**:qd 自身 IR/from_ast 正确(三 fmul 按序,无 FMA);端口逐句排查已排除
  (bvh 线:_S2D/_linear_combine/_gjk_support/缓冲区分配一致)。

## 3. 修复设计(候选,待异常闭合后定稿)

- **候选 A(对齐 parity,首选)**:CPU 后端 fast_math 不再设 `reassoc`(保留
  ninf/nsz/afn)——重结合是 parity 分叉的直接来源,warp CPU 无重结合;影响面 =
  全部 CPU kernel 的浮点结合序(逐位变化),需 union 全量数值门 + gate 4 perf 钉值
  复验。GPU/其他后端不动(afn 先例:codegen_llvm.cpp:605 按 arch 分域)。
- **候选 B(config 化)**:fast_math 拆出 `fp_reassoc` 子旗标(默认 off 对齐
  warp),三件套入 cache key。footprint 更大,仅当 A 引发数值回归时启用。
- **候选 C(算法侧)**:collision_gjk.py 在敏感性位点语句化(强制求值序)——
  port 侧 workaround,不作 qd 修复。

## 4. 测试标准(T1-T6 + 本线专项)

- 专项:三个失败测试转绿;`a47_reassoc_repro.py` ON arm == warp == L2R;
- T1 union(默认)21/21;T2 union(QD_IR_AUDIT=1)21/21;T3 test_return 46/46;
- T4 ruff 归零;T5 clang-tidy 改动文件 0 repo warnings;
- T6 数值面:若浮点结合序全量变化,抽验 collision/solver 族既有断言。

## 5. 工作量

归因闭合 0.5 天;候选 A 实现 + 门禁 0.5 天;回归处置缓冲 0.5-1 天。
