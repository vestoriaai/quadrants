# A46 递归守卫线任务书(mujocoqd/a46-recursion-guard)

> 依据:mujoqo docs/qd-A24-A46-开发文档.md §1.1;账本 A46
> (quadrants-需求清单.md)。基线:robostudio-main @ e9408978。

## 范围
@qd.func 前端内联加 thread-local in-progress 栈(func 对象为身份键),
重入 = 递归环 → QuadrantsSyntaxError 全环链命名
(`a <- b <- a`)。挂点两处:func.py 普通 func 编译路径(整段 with 包裹)、
real_func do_compile 的 func_body(transform 延迟执行,守卫随 transform)。
Kernel 根不加守卫(环必经 func;materialize 多 key 重入为顺序非嵌套)。

## 验收(T1-T6 通用门禁 + 本线专项)
- 专项:a46_func_recursion_repro.py 从静默 SIGKILL(25s 2.76GB)变为秒级
  QuadrantsSyntaxError 且消息含环链;非递归对照正常。
- 误报面:test_return 46/46、union 21/21 × {默认, audit}(gate 21 realfunc
  composite 覆盖 real_func 路径)、int2bool、qd.copy 等 impl 原生函数路径。

## footprint
_func_base.py +35(threading/contextlib import、tls、guard contextmanager);
func.py +3/-3(两挂点包裹)。零默认行为变化(非递归路径仅 push/pop)。
