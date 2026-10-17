# 开发规格:qd 原生 BVH(泛型 AABB 空间查询原语)

需求来源:mujoqo/docs/qd-原生BVH-需求评估.md(v3,已 review 通过)。本文件是 qd 侧实施规格。
分支:mujocoqd/bvh(worktree /Users/mini0/Code/quadrants-bvhwt)。合并:squash 进 robostudio-main,
同步更新 docs/robostudio_features.md(ledger)与本文件。

## 目标

为 qd 提供 warp-parity 的空间查询能力(warp 的 wp.Bvh/wp.Mesh/bvh_query_* 对应物):
泛型 AABB BVH——构建 kernel 流水线、遍历查询 func、refit kernel、host 句柄。
**纯 qd kernel + qd 数组:零 native 改动、零 codegen 改动**。首消费方 mujoco_qd 的
ray/mesh 路径(应用层自行包装三角语义,不在本规格内)。

## 照抄来源(warp native,算法级直译)

只读参照(venv 内 warp 安装):
- `<venv>/lib/python3.12/site-packages/warp/native/bvh.h`(762 行):
  - `BVHPackedNodeHalf`:打包节点(vec3 bounds + w 通道 = child index / leaf 标志位)
  - `bvh_query_next_impl`(584-655 行):共享栈式遍历骨架——**逐行照抄的对象**
  - `bvh_query_test` 各谓词(RAY=slab test / AABB / SPHERE)
  - `bvh_query_ray` 构造(552 行)
- `<venv>/lib/python3.12/site-packages/warp/native/bvh.cu`(888 行):GPU 构建流水线
  - `compute_morton_codes`(184)→ radix sort(可用 qd sort 替代)→ `build_leaves`(228)
    → `build_hierarchy`(261)→ `mark_packed_leaf_nodes`(402);`bvh_refit_kernel`(42)
  - morton3 编码在 bvh.h(288)
- 数据布局与 warp 保持同构(便于对拍)。

## 模块布局

`quadrants/bvh.py`(新文件,~600-900 行预计):

1. **host 句柄 `Bvh`**
   - 字段:lower/upper(输入 AABB,vec3 qd 数组)、node_lowers/node_uppers(vec4 打包)、
     primitive_indices(int)、root(int 标量数组)、bvh 总数
   - 方法:`build()`(发射构建流水线)、`refit()`(发射 refit kernel)
   - 构造入参:lower/upper qd 数组(泛型——调用方自备 AABB;mujoco_qd 的 mesh 侧负责
     从三角网算 AABB)
2. **构建 kernel 流水线**(照抄 bvh.cu)
   - `compute_morton_codes`:归一化到总 bounds,morton3<1024> 编码(bvh.h:288 直译),
     64 位 key = (group<<32)|morton(group 暂恒 0,groups 分组留接口不实现——warp 有
     groups 但 mujoco_qd 首需不需要)
   - 排序:复用 qd 现成 sort 原语(见"约束")对 (key, index) 排序
   - `build_leaves` / `build_hierarchy`:bvh.cu:228/261 直译(差分构造 delta 分裂)
   - `mark_packed_leaf_nodes`:叶聚合(leaf_size 上限)
   - 验收:同输入与 warp 构建(如可达)对比遍历结果集合一致;不可达则查询语义自测
3. **遍历查询 func(照抄 bvh.h 骨架)**
   - `bvh_query_next_impl` 直译为 `@qd.func`;谓词 RAY/AABB/SPHERE 三种(qd.func 参数化
     或三份展开——qd 无模板,选三份展开或 kind 参数分支,以 lint/编译干净为准)
   - 栈:优先 qd 本地定长数组(深度 64,验证 qd 支持;不行则 per-thread scratch 数组,
     形态参照 mujoco_qd/_src/sensor.py weld_geom_list——由调用方传入 scratch)
   - API 形态对齐 warp 语义:`bvh_query(bvh_id 或数组组, ...)` 迭代器式(next 一对)
     或回调式,以 qd.func 可组合为准;**必须有可在任意用户 kernel 内调用的形态**
4. **refit kernel**(照抄 bvh.cu:42):自底向上 bounds 更新(不动结构)

## 验证方案(本质口径)

1. **构建正确性 oracle**:不依赖"看起来对"——构建产物做三层自检:
  (a)结构不变量:节点数/叶覆盖/primitive 指数恰为排列(每个 primitive 恰出现一次);
  (b)每个 primitive 独立 aabb 查询自身必命中(上句的运行期表达);
  (c)与 warp native 语义对拍(oracle = 测试 venv 的 wp.Mesh,遍历结果集合相等,
  不比顺序;warp native 只作参照,产物不依赖它)。
2. **遍历语义独立参考**:谓词的真值用 numpy 直接实现(slab test/盒重叠/球距)逐
  primitive 暴力对照——期望集合从独立实现导出,不得从被测 kernel 的行为反推
  (mujoqo 纪律:断言生效 ≠ 断言正确)。
3. **确定性**:同进程三次构建+遍历结果逐位一致(cpu 多线程默认档)。
4. **后端口径**:cpu 先行全绿;metal 臂结果单列,失败先对号 A10-A12(行序律/
  `+=` 形式/kernel 边界可见性),不混入移植错误。
5. **性能口径**:本规格不设性能门;若测,先 `bash tests/robostudio/_gpu_busy.sh`,
  机器方差 ±4×,结论用交错中位。

## 约束

- 纯 qd kernel/funf + qd 数组;禁 native/C++/codegen 改动
- CPU 先行验证;Metal 臂跑通为准(BVH kernel 是普通 qd kernel,理论上双后端;metal
  失败先对号 A10-A12 地雷,见 mujoqo qd-kernel skill)
- 排序复用 qd 内部 sort 原语(`quadrants/_kernels.py` sort_stage 系;先勘察其 API
  形态再决定直调或包装;不得为此改 sort 本身)
- 写/改任何 kernel 前先跑 `python3 _probe/qd_lint.py <paths>`(mujoqo 侧工具,规则
  R1-R21;分批 commit 前 lint 清零)
- 已知地雷(qd-kernel skill 18-22 条):并行游标头声明泄漏、early_return 守卫块作用域、
  0 宽 padding、切片视图 IO——遍历/构建 kernel 全部按这些约束写

## 测试与门禁

- 新 gate:`tests/robostudio/gate_bvh.py`(gate 18):
  1. 构建正确性:随机/退化 AABB 集(含重复中心、单元素、n=1/n=0 边界)构建后,
     对每个 primitive 独立遍历(aabb query 自身)必命中自身
  2. 遍历语义 A/A:同输入 AABB 集,wp.Mesh(测试 venv 的 warp,CPU)查询结果集合
     == qd BVH 查询结果集合(ray 谓词:命中 primitive 集合一致;不比顺序)
  3. ray 语义:已知场景(两盒一球,手工期望)ray 最近/任意命中断言
  4. refit:移动若干 AABB 后 refit,查询结果随动
  5. 边界:n=0(空 BVH 查询返回 False)、leaf_size 边界
- 注册进 `tests/robostudio/run_robostudio_gates.sh`(gate 18,cpu 臂必过;metal 臂
  跑通为佳,失败须归因 A10-A12 并在 ledger 注明)
- `tests/robostudio/` 下新增 probe/gate 文件命名沿用现有惯例

## 纪律(硬性)

- 白名单:`quadrants/bvh.py`(新)、`tests/robostudio/gate_bvh.py`(新)、
  `tests/robostudio/run_robostudio_gates.sh`(追加 gate 18)、`docs/bvh_development_spec.md`
  (本文件,状态更新)、`docs/robostudio_features.md`(ledger 条目)。其余一概不动。
- 小步 commit:每完成一个独立件(构建流水线/遍历/refit/gate)一个 commit;
  message 前缀 `feat(bvh):` / `test(bvh):` / `docs(bvh):`,尾缀
  `Co-Authored-By: Claude Code <noreply@anthropic.com>`
- 开工前 `pwd` 确认在 worktree;全程路径相对 worktree;不得碰主树 /Users/mini0/Code/quadrants
- 里程碑即报(构建流水线绿/遍历绿/gate 绿);报数字不报感觉

## 完成定义(DoD)

1. `quadrants/bvh.py` 全部 lint 清零(mujoqo 侧 `python3 _probe/qd_lint.py` 对该文件)
2. `gate_bvh.py` 全断言过(cpu);metal 臂结果记录
3. `zsh tests/robostudio/run_robostudio_gates.sh` 全绿(既有 gate 1-17 无回归)
   **+ 上游 `pytest tests/python/test_return.py` 46/46**(qd CLAUDE.md 门禁硬性项)
4. ledger 条目已写,四字段齐:**设计/数字/footprint/限制**(见下预算)
5. 报告:commit 列表、gate 输出摘要、遗留项

## Footprint 预算(qd CLAUDE.md 一等指标)

- 新增:`quadrants/bvh.py`(预计 ≤900 行)+ `tests/robostudio/gate_bvh.py`(预计
  ≤400 行)+ runner 追加 ~10 行 + ledger ~10 行;**总增量 ≤1400 行,零 native、
  零既有文件语义改动**(runner 只追加注册块)。
- 触及核心文件:无(sort 经公共 API 调用;若发现必须改 sort 本体,停下报告,
  那是 OWNER ADJUDICATION)。
- 超预算即停:实现中发现显著超出(如遍历必须改 codegen),写报告交协调人裁决,
  不自行扩面。

不做(后续线):wheel 重烤与 mujoco_qd 接线(协调人另派)。

## 状态(2026-10-17 实施)

**landed on mujocoqd/bvh,待协调人 squash 进 robostudio-main。**

- commits:87d75bcf(勘察探针 P1-P7)→ bfd5b39c(quadrants/bvh.py:构建流水线+遍历+refit,
  882 行)→ 1afd8ab9(gate 18 + runner 注册)→ 本条 docs。
- DoD 核对:
  1. lint:bvh.py 与 gate_bvh.py 均清零(qd_lint R1-R21;gate 内 20 处系 @wp.kernel
     oracle kernel 被工具误判为 qd 函数的假阳性,已用 wpk 别名规避——工具假设
     mujoco_qd 文件无 wp kernel,此处是 gate 文件特例,规则本身无需改)。
  2. gate_bvh.py:**cpu 27/27 PASS,metal 27/27 PASS**(venv-consolidated 运行时
     dev25+g3297e86fe);前向兼容:新运行时(0de907cd wheel)cpu 亦 27/27。
  3. 上游 test_return.py **46/46**;union runner TOTAL 17→18,gate 18 双臂 PASS;
     既有 gate 1-6/13 全过,gate 7-12/14-17 **import 期失败 = 母盘 venv 陈旧**
     (dev25 缺 spirv_vectorize/host_visible/reassociate init 旗标、_cpu_paths、
     sort_segmented、early_return kwarg——各失败均有专名证据;M2 账记载 17 门
     全量当时跑在 venva7=dev104 自建 wheel 上,该 venv 本机现已不存在)。
     非本线回归:bvh 分支零既有文件改动。
  4. ledger 四字段条目已写(docs/robostudio_features.md,INDEX 行已续)。
- 遗留(后续线):单 group(groups 不实现);capsule 谓词未引;排序不稳定(平键
  顺序网络相关,查询候选集与树形无关);构建/refit 完成链串行(确定性优先);
  单 kernel 可组合形态待运行时基数排序 cpu 通路进 wheel;wheel 重烤与
  mujoco_qd 接线(协调人另派)。
