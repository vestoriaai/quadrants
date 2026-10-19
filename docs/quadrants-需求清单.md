
#### A42 开发线里程碑 2 + 定性更正(2026-10-19 凌晨,dev134 wheel)

- **dev126/127 崩溃族定性更正**:A42 patch v1 漏插 dest MatrixPtrStmt →
  TypeCheck 读已释放语句 = battery 大面积 SIGSEGV 的真因(**非** A37 线 UAF
  暴露)。patch v2(补插 + element 读取经临时 alloca)后 dev134 全绿:
  int2bool 3/3、union runner **20/20**(gate 14 x3 新钉值 + gate 20 首跑)。
  "A37 线 UAF 暴露常态化阻塞 dev126+" 前判作废;A37 线回归原状态(仅 a19
  宿主 x3 防闪烁)。
- **A42 残余失败精确定位**:V2/V2c = 前端**值侧链式下标**(`v[0,0][0]`:张量值
  语句的分量读)降为 `MatrixPtrStmt(origin=值语句)` → 构造期 origin 类拒绝
  (statements.cpp:132)。修复方向 = 前端值侧分量读改 GetElement 或 alloca
  停泊(make_tensor_access 已有非 lvalue 停泊逻辑,值侧 IndexExpression
  路径未走它)。**下一 fix 线入口明确,预计小改动**。

#### A42 开发线里程碑 3(2026-10-19 凌晨,dev142 wheel)

- **patch v3(空 origin 修复)验证**:SIGSEGV 崩溃族与 MatrixPtr 空 origin 拒绝
  双双消失;battery 除 V2/V2c 外全绿(V1/V2d/V3/A34 三案例/FamilyR/A43)。
- **V2/V2c 递进到第三层**:`alias_analysis.cpp:127 ptr1->indices.size() ==
  ptr2->indices.size()`——temp-alloca + per-element load 路径触碰
  alias_analysis 的 MatrixPtr 契约(同一 base 的 MatrixPtr 索引数须一致)。
  下一层修复方向:①val 表达式树递归 scalarize(操作数逐元素化,不引入
  temp alloca);②或 alloca 形状改为标量数组逐元素 store。**需 lldb/单步
  在 alias_analysis 现场确认契约细节后定**。
- **根修定性**:A42 的真根因 = 前端把 `v[0,0][0]`(ndarray 元素 vec3 值的
  分量读)物化为 TensorType[1] 值树而非标量——前端组件读类型化修复
  (值侧链式下标)才能根治;scalarize 侧为防御性兜底。
