# A34 线任务书 — cast f32→u1 走 CreateIsNotNull(浮点非法 ICmp)→ abort / 非法 IR

- 线名:`mujocoqd/a34-bool-cast`(worktree `~/Code/quadrants-a34wt`,从
  robostudio-main HEAD 03ad496b 切出)。
- 账:mujoqo `docs/quadrants-需求清单.md` A34;评估成文 mujoqo
  `docs/qd-A30-A34-缺陷修复线-需求评估.md` §1(review 通过,2026-10-18 用户开工令)。
- 复现件:mujoqo `_probe/a30_bool_cast_probe.py`(三案例)+ `_probe/
  a34_bool_cast_semantics_probe.py`(定性矩阵,2026-10-18 新增)。

## 判决

**方案 B(双后端 cast_value→u1 修复)**,理由(review 定案 + 定性矩阵推翻 F 案先验):

- 调用链:`bool(x)` → `impl.qd_bool` → `__qd_bool__` → `ops.cast(u1)` → 前端
  `Expr::cast` → 各后端 cast_value。原评估倾向 F(前端单点改写);但定性矩阵实测
  **qd 浮点 `!=` 的 NaN 语义双后端不一致**(cpu=ordered/NaN→False,metal=unordered/
  NaN→True,SPIR-V 发 OpFOrdNotEqual 经 SPIRV-Cross 到 MSL 塌缩为 C 语义)——前端
  F 案复用 `!=` 会在 cpu 复刻 NaN 错值,否则须新造 IR op(footprint 超预算)→ **B 案**:
  两后端各自在 cast_value→u1 处发**显式 unordered 比较**,语义一次定义(C:`x != 0.0`,
  NaN→true),int/ptr 路径不动。
- LLVM(codegen_llvm.cpp cast_value u1 分支):real 操作数 →
  `CreateFCmpUNE(input, ConstantFP::get(type, 0.0))`(vector 形态由 ConstantFP 广播);
  int/ptr → 保持 `CreateIsNotNull`(合法,零回归)。
- SPIR-V(spirv_ir_builder.cpp cast() u1 分支):新增 real 分支 →
  `OpFUnordNotEqual(value, float_immediate_number(stype, 0))`(原 loud reject
  "do not support type cast from f32 to u1" 移除);int/ptr 路径不变。
- 附带登记(非本线范围):qd 浮点 `!=` 的 cpu/metal NaN 语义分歧(cpu ordered vs
  metal MSL 塌缩)已实测,另立观察项;`ifflt` 案例实测前端对 `if <real>:` 响亮拒绝
  (QuadrantsTypeError)→ 隐式真值无隐藏路径,无需处理。

**反证判据**(任一出现即方案错,回滚重评):
1. 修复后 `flt2bool`/`nan2bool`(cpu)仍 abort 或值错;
2. int2bool(合法路径)回归;
3. 修复后 cpu 侧 NaN→False(说明发成了 ordered);
4. test_return 46/46 或 union runner 任一门回归。

## 验证方案(定性矩阵基线,2026-10-18 dev109 实测)

| 案例 | cpu 现状 | metal 现状 | 修复后 want(双后端) |
|---|---|---|---|
| int2bool | PASS | PASS | PASS 不变 |
| flt2bool | **abort 134**(ICmp AssertOK) | loud reject(spirv cast@906) | PASS,值 [F,T,T,T] |
| fltcmp(`!=`) | PASS 但 NaN→False(ordered) | PASS NaN→True | 不在本线范围(观察项) |
| nan2bool | abort 134 | loud reject | PASS,NaN→**True**(C 语义) |
| ifflt | 响亮拒(QuadrantsTypeError) | 响亮拒 | 不变(响亮拒) |
| veccmp_nan | PASS NaN→False | PASS NaN→True | bool() vector-cast 形态罕见,非registered 面 |

1. 烤 wheel(全量 native,LLVM 22.1.0 缓存已恢复)→ 装**自建验证 venv**,清 qdcache。
2. 探针三案例 + 定性矩阵(cpu+metal)全绿:`nan2bool` 显示 `NaN->True(UNORDERED/C-sem)`,
   `flt2bool` 值断言过。
3. upstream `pytest tests/python/test_return.py` 46/46;union runner 19/19。
4. support.py 还原 `bool()` 原形 = mujoqo 侧独立 commit(非本线)。

## footprint 预算

native 双文件 ~25 行:codegen_llvm.cpp +9/−1;spirv_ir_builder.cpp +4/−0。
禁顺手重构;int/ptr 路径逐位不变。

## DoD

- [ ] 探针矩阵按上表翻绿(cpu+metal);
- [ ] test_return 46/46;union runner 19/19;
- [ ] 线 commit(英文 subject,无 co-author)+ ledger 条目草案交协调人;
- [ ] LLVM `!=` NaN 分歧观察项入 mujoqo 账本(本线附带产出,另立条目)。

## 白名单

`quadrants/codegen/llvm/codegen_llvm.cpp`;`quadrants/codegen/spirv/spirv_ir_builder.cpp`;
`docs/a34_bool_cast_development_spec.md`;`docs/robostudio_features.md`(收尾)。
其余文件发现"问题"只报不改。
