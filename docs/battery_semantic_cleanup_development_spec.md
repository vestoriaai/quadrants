# battery-semantic-cleanup 开发任务书（qd 线）

> 2026-10-24。分支 `mujocoqd/battery-semantic-cleanup`（worktree quadrants-batwt，
> 基点 robostudio-main@1ac60f6d）。承接 A58 鉴别分桶（mujoqo TODO 10-24 批次①）。
> 必读 skill：quadrants 侧 `.claude/skills/qd-compiler/SKILL.md`。

## 1. 范围与判决（归因已定，三项）

### 1.1 A53 — kernel print 缓冲未随 sync 落 fd（print×6 + overflow×8 = 14 实例）

- **现象**：`test_print.py`（capfd）与 `test_overflow.py`（capfd 断言 kernel 内
  print 的 "overflow detected" 消息）在 pytest capture 下全部空输出；`-s` 可见。
- **归因**：`shell.py:15` `pybuf_enabled = sys.stdout is not sys.__stdout__`——
  pytest capture 下 sys.stdout 被替换 → pybuf 启用，kernel print 进 C++ python
  print buffer；`_shell_pop_print` 在**每次 kernel call 后** pop 并 `print()` 到
  **被替换的 sys.stdout**（CaptureIO，fd 层不可见）→ capfd 永远看不到。且与
  taichi #1303 语义（"print's in kernel won't take effect until qd.sync()"）矛盾。
- **修复设计**：
  1. `shell.py` 的 `new_call` 不再 call 后 pop（恢复"sync 才生效"语义）；
  2. `impl.PyQuadrants.sync()` 末尾（惰性 import shell 的 pybuf_enabled 防环）：
     pybuf_enabled 时 `_qd_core.pop_python_print_buffer()` 内容写
     `sys.__stdout__` 并 flush——直接落 fd 层，capfd/capsys/Jupyter 三方都对；
     pybuf_enabled=False 时 C++ 直写 fd，不动。
- **反证判据**：若 sync 后 capfd 仍空 → 修错层；若 `-s`（无 capture）出现重复
  打印 → C++ 直写路径与 python 路径双发，需在 shell.py 侧收口。
- **风险**：缓冲积累（不 sync 长跑）——与 taichi 语义一致，可接受；内存上限
  由 C++ buffer 侧既有行为决定，不在本线扩。

### 1.2 A46 守卫升级 — 误伤静态终止递归惯用法（ast_refactor×2 arch）

- **现象**：`test_ast_refactor::test_func_in_python_func`——上游合法惯用法
  `@qd.func bar(x: qd.template())` + `if qd.static(x): mat = bar(x // 2)`
  （模板递归，编译期 x 递减到 base case）被 A46 `inline_cycle_guard`
  （_func_base.py:223）以 "Recursive @qd.func call detected" 拒绝。
- **修复设计**：guard 栈条目从 `(func, qualname)` 扩为 `(func, qualname,
  called_from_static)`；同 func 重入时，**仅当本环每一层调用点都在 qd.static
  条件分支内**才放行，并施加**同 func 重入深度上限（128）**——超限仍报
  A46 错误（防失控展开）。`called_from_static` 由调用点 AST 上下文提供
  （static-if/static-while 守卫区内的 call 置 True）——落点在
  `function_def_transformer`/call 展开处向 guard 传递；若上下文标志在该层
  不可得，降级方案 = 深度上限放宽为模板参数指纹（x 值逐层递减即放行），
  实施时按可达性取其一，两案都在任务书备案。
- **反证判据**：真无限递归（无 static 判据）必须在编译期秒级报错（A46 原目标），
  六臂探针 `_probe/` 复验；test_return 46/46 不回退。
- **footprint 预算**：_func_base.py ±40 行内 + call 展开点 ±15 行；超预算停下报告。

### 1.3 pyi_stubs ×2 — 纯环境缺失（错分型纠正）

- `pyright` 包不在 .venv-consolidated → 测试 subprocess 直接 ModuleNotFound。
  此前账面"需重生成"是误判。修复 = venv 装 pyright；若装后仍红（真实 stub
  过期）则按实际 diff 重生成 stub 并单独 commit。零 qd 本体改动。

## 2. 白名单

- `python/quadrants/lang/shell.py`、`python/quadrants/lang/impl.py`（sync 段）
- `python/quadrants/lang/_func_base.py` + call 展开点（A46，实施时定：预期
  `ast_transformer*`/`function_def_transformer*` 一处）
- `_probe/` 新探针；`docs/robostudio_features.md`（ledger，合并时同 commit）
- pyi_stubs：仅 venv 操作 +（若需）stub 生成物

## 3. DoD 与门禁

1. test_print + test_overflow + test_ast_refactor + test_pyi_stubs 全绿；
2. battery 全量不劣化（18F 基线只减不增；A53/A46 修复后预计 18→4-5F）；
3. union gates 21/21；test_return 46/46；
4. A53 反证：`-s` 无重复打印；A46 反证：无限递归探针仍秒级报错；
5. ledger 四字段条目（合并 commit 内）。
