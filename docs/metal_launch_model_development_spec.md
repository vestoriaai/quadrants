# metal-launch-model 开发任务书(mujocoqd/metal-launch-model,2026-10-25)

> 上游依据:mujoqo `docs/metal-开发计划.md`(战线 A,§2/§10/§11)与 `docs/metal-launch-overhead-需求评估.md`。本任务书覆盖 **A1(阀门非阻塞化)**;A2(args ring 池)按 mujoqo 计划 §11.5 推迟(F6 判据),C-ICB 已删。

## 判决(问题)

`GfxRuntime::submit_current_cmdlist_if_timeout` 每 ~33 次 launch 强制 `synchronize()`(flush + `waitUntilCompleted` 全队列)——MoltenVK encoder tracker 防御,对原生 Metal 无必要。实测(Apple M4):每次排空 0.5-1.3ms 即便 GPU 工作 trivial;mujoqo humanoid step 126 launch/步 ≈4 次排空;heavy 核连发 host 完全串行(326µs/launch host 阻塞)。

## 方案

Metal 专属(`config->arch == Arch::metal`),`QD_METAL_RETAINED_FLUSH=0` 回退:
1. `flush()`:提交 cmdlist 后释放停车的 args/ret buffer——标准 MTLCommandBuffer 对已编码命令引用的资源 retain 到完成,放掉的只是 qd 的引用;ext-array 镜像仍停车(R3a 池复用需 GPU 空闲)。
2. 阀门:`flush()` + 清零 代替 `synchronize()`;镜像停车 >1024 作远端兜底 `synchronize()`。
3. `MetalStream::submit`:提交时从队首回收 `status >= Completed` 的 cmdbuf(FIFO 有序完成;含 Error,与 `command_sync` 不查状态一致)。
Vulkan 路径不变。

## 验证方案(含反证判据)

- 结构:mujoqo `_probe/metalov_step_census.py` in-call 排空/步 3.4→≤2(剩余为 port 侧观察点);`metalov_launch_floor.py` heavy ×128 连发 total ≈ 纯 GPU 时间(overlap=yes)。**反证**:排空数不降 / heavy 仍串行 → 阀门未绕过。
- 正确性:nworld=1 step 轨迹逐位一致(基线自身确定);nworld=64 落在基线 run-to-run 噪声包络内(metal 原子行序非确定);port metal 门禁 history/ray/passive/sensor/forward 失败集合与基线**逐名相同**(零新增)。
- qd:union gate runner 21 门(A1 venv)。

## footprint 预算

runtime.cpp +44/−1、metal_device.mm +12;零头文件改动;无新 API。实际 56 行。

## DoD

- [x] 实现 + 构建(worktree 内 `build.py wheel`,external/ 子模块源码自主树复制)
- [x] 结构指标:排空 ×64 3.4→2.0、×1 2.2→0.5;heavy 连发 host 326→13µs,total 52.7→36.3ms(=纯 GPU);trivial floor 19→6.8µs
- [x] parity:×1 逐位一致;×64 包络内(偏差 1.46e-6 vs 基线噪声 1.79e-6)
- [x] port 门禁零新增(forward 14F 集合逐名同基线;passive 6F/sensor 11F 同基线——三者为 dev136→dev152 既有回归面,移交)
- [ ] union gates 21/21(运行中,结果见 ledger)
- [ ] squash 进 robostudio-main + 重烤 wheel(协调人)

## 白名单

`quadrants/runtime/gfx/runtime.cpp`、`quadrants/rhi/metal/metal_device.mm`、本任务书、ledger 条目。
