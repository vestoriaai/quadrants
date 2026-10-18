# A32 线任务书 — early_return:static-if-return 位于 runtime return 之后的 SyntaxError

- 线名:`mujocoqd/a32-static-return-order`(worktree `~/Code/quadrants-a32wt`,
  从 robostudio-main HEAD 03ad496b 切出)。
- 账:mujoqo `docs/quadrants-需求清单.md` A32;评估成文 mujoqo
  `docs/qd-A30-A34-缺陷修复线-需求评估.md` §2(review 通过,2026-10-18 用户开工令)。
- 复现件:mujoqo `_probe/a32_static_if_after_er.py`(六臂,当前 wheel PASS=复现)。

## 判决

**修 qd 前端**(非 port 侧规避):early_return transformer 对 qd.static 控制节点持裸
return 的形态,应产出"无裸 return 残留"的单出口形态,且**条件值无关**(重写期不可求值
static 条件——重写在 IR 构建 walk 之前,static 求值在 walk 时)。

**方案(已 review 定案)**:transform_block 内统一规则——凡 static 节点(if: test 为
`qd.static(...)`;for: iter 为 `qd.static(...)`)子树含裸 `ast.Return`,就地对子树 body
套用本 pass 自身的 transform_block(槽赋值 + done 置位 + loop_depth 透传,break 规则
不变),节点保留 static test/iter。正确性论证:
- TRUE 分支:槽赋值执行、done=1,后继语句被守卫链拦截(walk 期消除)——等价于 return;
- FALSE 分支:walk 期整支剪除,槽赋值从不执行;
- 位点无关性:前 return 位点的 static 节点改写后含 done store,被 `_contains_done_store`
  识别为 return-site → 后继语句获守卫(修复嵌套形态:runtime-if 内 static return 原
  本外层不可见,后继语句漏守卫);顺带覆盖 static 循环、嵌套 static、tuple 多槽。

**反证判据**(若出现下列任一,方案判错,回滚重评):
1. 修复后 V_curse max4(上游形态)仍 SyntaxError 或数值错;
2. 对照臂(max3/max6)任一编译失败或数值错(改写破坏既有绿形态);
3. upstream test_return.py 任一用例回归(early_return 主面);
4. union runner 19 门任一变红(尤其 gate 17 early-return compound F1-F14)。

## 验证方案

1. mujoqo 探针升级为三态 oracle(REPRODUCED/FIXED/FAIL)+ 新边界臂(嵌套 static-in-
   runtime-if、static 循环含 return、static-FALSE 含 return、tuple 多槽 static return):
   旧 wheel 全部 REPRODUCED,新 wheel 全部 FIXED,FAIL=0。
2. upstream `pytest tests/python/test_return.py` 46/46(线 venv)。
3. union runner `zsh tests/robostudio/run_robostudio_gates.sh <线venv>` 19/19。
4. cpu 主验;A32 是纯前端 AST 变换,cpu+metal 行为一致(metal 臂由 union runner 双臂
   gate 覆盖,不单列)。

## footprint 预算

python-only,单文件 `early_return_transformer.py`:预计 +30–50 行(两 helper + 一处
分支 + docstring 更新),0 删除性改动。禁顺手重构;OFF/既有路径逐位语义不变(门禁保)。

## DoD

- [ ] 探针三态 oracle 全绿(新 wheel FIXED);
- [ ] test_return 46/46;union runner 19/19;
- [ ] A32 线 commit(英文 subject,无 co-author 尾注)+ ledger 条目草案交协调人;
- [ ] port 侧还原(286e62f workaround 撤销)归 mujoqo 侧独立 commit,不在本线范围。

## 白名单

`python/quadrants/lang/ast/ast_transformers/early_return_transformer.py`;
`docs/a32_static_return_order_development_spec.md`;`docs/robostudio_features.md`
(ledger 条目,收尾时)。其余文件发现"问题"只报不改。
