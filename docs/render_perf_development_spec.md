# render-perf 开发任务书（qd 线）

> 2026-10-24。分支 `mujocoqd/render-perf`（worktree quadrants-rpwt，基点
> robostudio-main@1ac60f6d）。承接 mujoqo readback 分解（render-开发计划 Phase 4
> 附加段）：e2e 11.46ms = submit 3.18 + GPU 6.98 + readback 0.4；两个杠杆按
> 本线归因后择优实施。**先归因后修**——归因结果可能推翻杠杆假设，届时停下
> 更新本任务书再动手。

## 2. 杠杆 B（先做，便宜）：host 提交 3.18ms — C1a fast lane 覆盖检查

- **背景**：C1a Kernel fast lane（qd 49adfdac）绑定闭包直达 C++（2.89→1.24µs），
  五情况失效防御 fall-through，其中 struct-ndarray args 走 guard 链慢路。
  render megakernel 144 参数（大量 ndarray）疑似落入慢路；3.18ms 可能大头是
  每 render() 的逐参数 guard 检查 + 装配。
- **步骤**：
  1. mujoqo 侧探针：对 `_render_megakernel` launch 打点（kernel.py fast lane
     入口计数/计时，QD_LAUNCH_FAST=0 对照），确认是否走慢路 + 慢路内时间占比；
  2. 若确认：归因哪一防御条件触发（五情况逐一排除），评估放宽（如
     guard 链的预计算/缓存——同 kernel 同 args 结构的 guard 结果 memo）；
     设计上不得削弱五情况防御的正确性。
- **预期**：submit 3.18→~1ms（poc upload 0.79ms 为参照）；不达预期则如实记录。
- **反证判据**：QD_LAUNCH_FAST=0 对照数字不劣化；五情况探针（c1a_lane_check）
  不回退。

## 3. 杠杆 A：GPU kernel 6.98ms — 分段计时归因（shadow BVH 二遍 / 多世界间接）

- **背景**：poc 同口径 kernel 3.51ms vs RenderContext 版 6.98ms。候选：
  shadow ray 二次 BVH 遍历、多世界间接寻址、AA/顶点法线等功能面差异。
- **步骤**：
  1. mujoqo 侧变量隔离探针（bench 件 `render_p3_bench.py` 改造）：矩阵
     {shadows on/off} × {nworld 1/64} × {单/多相机}，拿各配置 kernel 时间
     （显式 sync 分解口径）；
  2. shadow 归因若成立：对照 poc 的 shadow 实现 diff（shadow ray 路径
     BVH 遍历形态、early-exit、stack 复用）——移植级差异逐条列清单，
     逐条改+复测（shadow off 时若 7.0→3.5 则 shadow 即全部账）；
  3. 多世界间接：nworld=64 vs 1 的 kernel 时间差应≈0（同射线量）——若显著
     为正，归因间接层并评估；
  4. 每条优化 golden 12 + render_test 100P 零回退（正确性优先）。
- **反证判据**：shadows=off 与 on 的差值即 shadow 账上界；优化后若
  shadows=on 仍 >5ms，剩余账如实单列，不强修。
- **footprint**：mujoqo render.py kernel 侧改动逐条 commit；qd 本体零改动
  预期（若归因指向 qd codegen，另行停等登记）。

## 4. DoD

1. 64envs×84×84 shadows=on e2e：目标 ≤9.5ms（两个杠杆吃满的保守预期），
   门槛 15ms 保持 PASS 且余量扩大；实测数字如实记录；
2. mujoqo render 全量 100P/2S/0F + golden 12/12 不回退（每次 kernel 改动后复验）；
3. qd 侧（若动）union gates 21/21；
4. 归因数据 + 优化前后对照表进 mujoqo render-开发计划.md；
5. ledger 四字段条目（有 qd 侧改动时）。

## 执行结果（2026-10-24，方案 1/3 双证伪）

| 方案 | 结果 |
|---|---|
| 3（composable radix sort） | **❌ metal 上 GPU 时间 2.2× 恶化**（64envs shadows=on gpu_wait 4.72→12.46ms），结果正确（golden 逐位同值）。16 个 ~128 项小数组上 radix 的每-phase 固定 GPU 开销远超 Batcher 网络 |
| 1（去 per-stage sync） | **❌ metal 上总账 2× 恶化**（submit+gpu_wait 7.73→15.7-16.5ms），结果正确（m13_shadow golden 1.731 逐位同值）。per-stage sync 在 qd metal 后端**不是纯浪费**——疑似 launch 攒批/sync 触发 command buffer commit-and-wait 的提交节拍，去掉触发批量化惩罚（机制待查） |
| 2（build 移出帧循环） | 唯一未证伪路径。bvh.py:50 裁定"refit 不能跨拼接所以选重建"——重开需 per-group refit 循环或 refit 跨拼接实现；另一 sub-option = bounds dirty-check（probe 反复渲染同一帧仍全量 build = bounds 未变也重建）。独立评估 |

对照基线复验：dev142 轮原版 bvh 同轮 7.36ms / dev145(A53+A46) 轮原版 bvh 7.73ms——A53/A46 **无** shadows 性能回归（初期 16ms 疑云 = bvh 改动所致，已归因排除）。

判据：两方案在 metal 上均为净恶化，**不交付**；bvh.py 改动已全部回退，验证 venv 恢复原版。方案 2（dirty-check/refit 恢复）是剩余唯一可行路径，工程量中等（需动 refit 跨拼接限制），独立评估后再立项。

## 重要更正（2026-10-24 深夜）：方案 1/3"证伪"撤回——原测量被 GPU 共享噪声污染

min-of-N 交错重测（`_probe/render_perf_min_comparison.py`，三变体交错 × 2 轮 ×
min-of-9，64envs shadows=on 的 sync_drain 段）：

| 变体 | r0 min | r1 min |
|---|---|---|
| pristine（448 sync Batcher） | 4.99ms | 5.23ms |
| nosync（3 sync） | 8.56ms | 5.57ms |
| radix（composable chain） | 5.69ms | 5.50ms |

三变体 min 值在 5.0-5.6ms 带内（<12%）——**无显著差异**。且同配置单轮
sync_drain 漂移 4.78→10.54（±2×）实测（本机 GPU 为共享资源，有并行会话）。
结论修正：

1. 此前"方案 1/3 双证伪"（12-16ms vs 7.1ms）的数字**落在噪声峰内，不可判读**，
   证伪撤回；同时此前"去 sync 是低垂果实"的判定同样未经抗噪验证，一并撤回。
2. "metal 提交节拍"强怀疑**无数据支撑，撤回**——no-sync 与 pristine 在交错
   min 口径下不可区分。机制假设（hazard tracking 等）留待 Metal Instruments
   GPU profile 定论（Apple 文档确认 tracked 模式存在 runtime 同步屏障，但无
   官方定量数字）。
3. 本机 render perf 的 ms 级测量在 GPU 共享态下噪声带 ±2×——**任何后续 perf
   判定必须 min-of-N（≥12）+ 交错对照 + GPU 独占窗口**。Phase 3 的 11.46ms
   门槛数字同为 median 口径，同样带噪声（min 口径更低）。
4. 结构性事实仍成立（不受噪声影响）：985 launch/render 的构成普查、144 参数
   0 struct-ndarray、sort_stage 448。减少 launch 数的方向（dirty-check/合批）
   理论收益仍在，定量收益需上述抗噪口径重测。
5. radix 变体实现（脚本可再生，本 commit 未含 bvh.py 改动）结果正确，留作
   GPU 独占窗口重测的候选；当前不交付。
