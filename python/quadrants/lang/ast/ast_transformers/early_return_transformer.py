"""Early-return → single-exit rewrite (opt-in via `@qd.kernel`/`@qd.func(early_return=True)`).

Warp-style guard-clause code uses `return` inside runtime if/for/while, which the quadrants
front-end rejects ("Return inside non-static if/for is not supported"). This pass mechanically
converts every early return into single-exit form *before* the regular AST walk, preserving
semantics and inlining:

    def f(a: qd.f32) -> qd.f32:            def f(a: qd.f32) -> qd.f32:
        if a < 0:                              __er_ret0 = qd.cast(0, qd.f32)
            return -1.0                        __er_done = 0
        return 1.0                             if a < 0:
                                                   __er_ret0 = -1.0
                                                   __er_done = 1
                                               if __er_done == 0:
                                                   __er_ret0 = 1.0
                                               return __er_ret0

Rules (see docs/robostudio ledger, mujocoqd/compiler-line M2):
- Slot types come from the return annotation via `qd.cast(0, <dtype>)` init. Unannotated
  value-returning functions are rejected with a clear error (bare-`return` void functions
  are supported without slots).
- `return` inside loops additionally emits `break` — EXCEPT in a kernel's outermost
  (parallelized) loop, where break is forbidden by the front-end; there the remaining loop
  body is simply guarded and the loop runs out (semantically equal, SIMT-free early exit).
- Subtrees inside `qd.static(...)` control flow are left untouched: static returns are
  already legal and are compile-time.
- Name space `__er_ret{i}` / `__er_done` is verified collision-free against the function.
"""

import ast

from quadrants.lang.exception import QuadrantsSyntaxError

_RET = "__er_ret{}"
_DONE = "__er_done"


def _stamp_deep(node, loc):
    """Fill in missing location info on synthesized nodes (unparse/compile need it)."""
    for child in ast.walk(node):
        if isinstance(child, (ast.stmt, ast.expr)) and not hasattr(child, "lineno"):
            child.lineno = loc
            child.col_offset = 0
            child.end_lineno = loc
            child.end_col_offset = 0
    return node


def _contains_done_store(stmts):
    for n0 in stmts:
        for node in ast.walk(n0):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == _DONE:
                        return True
    return False


def _is_static_call(node):
    """True if the expression is (or contains) a `qd.static(...)` call."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr == "static":
                return True
            if isinstance(f, ast.Name) and f.id == "static":
                return True
    return False


class _Ctx:
    def __init__(self, nret, is_kernel, loc=1):
        self.nret = nret
        self.is_kernel = is_kernel
        self.loc = loc


class _Rewriter:
    def __init__(self, ctx: _Ctx):
        self.ctx = ctx

    def _ret_stmts(self, value, loop_depth):
        stmts = []
        if value is not None:
            if isinstance(value, ast.Tuple):
                for i, elt in enumerate(value.elts):
                    stmts.append(
                        ast.Assign(targets=[ast.Name(id=_RET.format(i), ctx=ast.Store())], value=elt)
                    )
            else:
                stmts.append(
                    ast.Assign(targets=[ast.Name(id=_RET.format(0), ctx=ast.Store())], value=value)
                )
        flag = ast.Assign(targets=[ast.Name(id=_DONE, ctx=ast.Store())], value=ast.Constant(value=1))
        stmts.append(flag)
        if loop_depth > 0 and self._break_allowed(loop_depth):
            stmts.append(ast.Break())
        return [_stamp_deep(s, self.ctx.loc) for s in stmts]

    def _break_allowed(self, loop_depth):
        # func: loops are nested after inlining — break is legal and keeps linesearch-style
        # early exits cheap. kernel: the outermost (parallelized) loop forbids break; there we
        # rely on guards only. Deeper kernel loops are serialized and may break.
        if not self.ctx.is_kernel:
            return True
        return loop_depth >= 2

    def _guard(self, stmts):
        test = ast.Compare(
            left=ast.Name(id=_DONE, ctx=ast.Load()),
            ops=[ast.Eq()],
            comparators=[ast.Constant(value=0)],
        )
        return _stamp_deep(ast.If(test=test, body=list(stmts), orelse=[]), self.ctx.loc)

    def transform_block(self, stmts, loop_depth):
        out = []
        seen_return = False
        for st in stmts:
            if seen_return:
                new_st = self.transform_stmt(st, loop_depth)
                body = new_st if isinstance(new_st, list) else [new_st]
                out.append(self._guard(body))
                continue
            new_st = self.transform_stmt(st, loop_depth)
            if isinstance(new_st, list):
                out.extend(new_st)
                has = _contains_done_store(new_st)
            else:
                out.append(new_st)
                has = _contains_done_store([new_st])
            if has:
                seen_return = True
        return out, seen_return

    def transform_stmt(self, st, loop_depth):
        if isinstance(st, ast.Return):
            return self._ret_stmts(st.value, loop_depth)

        if isinstance(st, ast.If):
            # qd.static branches are compile-time: returns there are already legal; leave untouched.
            if _is_static_call(st.test):
                return st
            body, _ = self.transform_block(st.body, loop_depth)
            orelse, _ = self.transform_block(st.orelse, loop_depth)
            return _stamp_deep(ast.If(test=st.test, body=body, orelse=orelse), getattr(st, "lineno", 1))

        if isinstance(st, (ast.For, ast.While)):
            # static loops (`for i in qd.static(range(n))`) are compile-time: leave untouched.
            if isinstance(st, ast.For) and _is_static_call(st.iter):
                return st
            sub = loop_depth + 1
            st.body, _ = self.transform_block(st.body, sub)
            st.orelse, _ = self.transform_block(st.orelse, loop_depth)
            return _stamp_deep(st, getattr(st, "lineno", 1))

        return st


def rewrite(node: ast.FunctionDef, is_kernel: bool, global_vars: dict) -> ast.FunctionDef:
    """Rewrite `node.body` in place into single-exit form. Returns the node."""
    # name-space collision check against the names this pass will introduce
    used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    if _DONE in used:
        raise QuadrantsSyntaxError(
            f"early_return=True: function {node.name!r} already uses the reserved name {_DONE!r}"
        )

    # slot count from the widest tuple return
    nret = 1
    has_value_return = False
    has_bare_return = False
    for n in ast.walk(node):
        if isinstance(n, ast.Return):
            if n.value is None:
                has_bare_return = True
            else:
                has_value_return = True
                if isinstance(n.value, ast.Tuple):
                    nret = max(nret, len(n.value.elts))

    if has_value_return:
        for i in range(nret):
            if _RET.format(i) in used:
                raise QuadrantsSyntaxError(
                    f"early_return=True: function {node.name!r} already uses the reserved name "
                    f"{_RET.format(i)!r}"
                )

    if has_value_return:
        if node.returns is None:
            raise QuadrantsSyntaxError(
                f"early_return=True: function {node.name!r} returns values but has no return "
                "annotation; add one (e.g. -> qd.f32 or -> tuple[qd.f32, qd.f32]) so the rewrite "
                "can type the return slots"
            )
        if nret == 1:
            # scalar annotation: keep as the single slot dtype source
            ann_elts = [node.returns]
        elif isinstance(node.returns, ast.Subscript) and isinstance(node.returns.slice, ast.Tuple):
            ann_elts = list(node.returns.slice.elts)
        elif isinstance(node.returns, ast.Tuple):
            ann_elts = list(node.returns.elts)
        else:
            raise QuadrantsSyntaxError(
                f"early_return=True: function {node.name!r} returns {nret}-tuples but the return "
                f"annotation is not a tuple type"
            )
        if len(ann_elts) < nret:
            raise QuadrantsSyntaxError(
                f"early_return=True: function {node.name!r} returns {nret}-tuples but its "
                f"annotation only describes {len(ann_elts)} slot(s)"
            )
    else:
        ann_elts = []

    ctx = _Ctx(nret, is_kernel)
    rewriter = _Rewriter(ctx)
    body, _ = rewriter.transform_block(node.body, 0)

    # prologue: typed slot initializers + flag
    # NOTE: init via `qd.cast(0, qd.i32)` rather than a bare `= 0` constant. After func
    # inlining, a constant prologue store is treated as loop-invariant by the optimizer and
    # hoisted out of the caller's loop, leaking `__er_done` across calls (observed: serial
    # loop, second call skipped its body). The call-shaped init stays put. Same reason the
    # slots use `qd.cast`. Upstream issue: inlined-func prologue store over-hoisting.
    prologue = [_stamp_deep(
        ast.Assign(
            targets=[ast.Name(id=_DONE, ctx=ast.Store())],
            value=ast.parse("qd.cast(0, qd.i32)", mode="eval").body,
        ),
        1,
    )]
    for i in range(nret if has_value_return else 0):
        elt = ann_elts[i]
        src = ast.unparse(elt)
        try:
            dtype_expr = ast.parse(src, mode="eval").body
        except SyntaxError as e:
            raise QuadrantsSyntaxError(
                f"early_return=True: cannot resolve return annotation element {src!r} of "
                f"{node.name!r}: {e}"
            )
        call = ast.parse(f"qd.cast(0, {src})", mode="eval").body
        prologue.append(
            _stamp_deep(ast.Assign(targets=[ast.Name(id=_RET.format(i), ctx=ast.Store())], value=call), 1)
        )

    if has_value_return:
        rets = [ast.Name(id=_RET.format(i), ctx=ast.Load()) for i in range(nret)]
        final_value = rets[0] if nret == 1 else ast.Tuple(elts=rets, ctx=ast.Load())
    else:
        final_value = None
    final = ast.Return(value=final_value)

    node.body = prologue + body + [final]
    ast.fix_missing_locations(node)
    return node


class EarlyReturnTransformer:
    """Pass handle matching the other `ast_transformers/*` classes (CheckpointTransformer etc.)."""

    rewrite = staticmethod(rewrite)
