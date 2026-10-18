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
- Slot types come from the return annotation. Primitive slots init via `qd.cast(0, <dtype>)`;
  compound slots (VectorType/MatrixType annotations, e.g. tuple[vec3, qd.f32]) init via
  `qd.cast(<T>(0), <T>.dtype)` — the `T(0)` broadcast constructor produces a correctly-shaped
  tensor (make_matrix_expr carries T's element dtype) and the cast keeps the init call-shaped
  and the element dtype explicit. Unannotated value-returning functions are rejected with a
  clear error (bare-`return` void functions are supported without slots).
- Statements after the first return-bearing statement of a block are grouped into runs:
  plain statements merge into ONE `if __er_done == 0:` guard (per-statement guards would
  strand first-bindings in sibling runtime-if blocks: names born in one block are retired
  at block exit and the sibling read fails with an actionable branch-scope NameError);
  a sibling return-site (a compound containing a return) keeps its own guard and starts a
  fresh run, because taking it sets `__er_done` and later statements in the same body
  must not run. Source-level dead code after a literal `return` at the same block level
  is dropped (it could never run); the literal return's slot stores stay in the current
  run so they can read names first-bound there.
- `return` inside loops additionally emits `break` — EXCEPT in a kernel's outermost
  (parallelized) loop, where break is forbidden by the front-end; there the remaining loop
  body is simply guarded and the loop runs out (semantically equal, SIMT-free early exit).
- Subtrees inside `qd.static(...)` control flow keep their static test/iter (the walk
  still evaluates them), but any bare `return` inside is rewritten with the same
  single-exit machinery. This is condition-value-agnostic by construction: the pass
  runs before the walk that evaluates qd.static, so it cannot know TRUE/FALSE — and
  doesn't need to. TRUE branch: the slot stores execute and everything after is
  guard-dead; FALSE branch: the walk prunes the branch and the stores never run.
  Leaving static returns untouched would embed a bare return inside runtime control
  flow — this pass's own `if __er_done == 0:` guard runs (a static-TRUE return after
  a runtime return; ledger A32), or an enclosing runtime if when the static site
  sits pre-return in the block — and trip the front-end's "Return inside non-static
  if/for" rejection either way. Rewritten pre-return sites register as return-sites
  through their new done-stores, so followers get the guards they need.
- Loud rejections: nested-tuple returns (tuple annotations containing a tuple) and
  struct/custom-class returns — for those shapes use a manual single-exit (flag-guard)
  function; mixed bare/value returns; return sites with inconsistent arity (a short return
  would silently leave the remaining slots at their init values).
- Name space `__er_ret{i}` / `__er_done` is verified collision-free against the function.
"""

import ast
import typing

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


def _contains_return(node):
    """True if the subtree still holds a bare `ast.Return` (untouched static sites keep them)."""
    for n in ast.walk(node):
        if isinstance(n, ast.Return):
            return True
    return False


def _is_static_site(node):
    """True for a compile-time control node transform_stmt leaves untouched (A32)."""
    if isinstance(node, ast.If):
        return _is_static_call(node.test)
    if isinstance(node, ast.For):
        return _is_static_call(node.iter)
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


def _resolve_slot_type(elt, node, global_vars):
    """Classify one return-annotation element: `("primitive", src)` or `("compound", src)`.

    compound = a VectorType/MatrixType instance (vec3, mat83, qd.types.vector(3, qd.f32), ...).
    Raises QuadrantsSyntaxError for shapes the single-exit rewrite cannot slot: nested-tuple
    annotations and struct/custom-class returns (both: "use a manual single-exit (flag-guard)
    function"), and elements not resolvable from the function's globals.
    """
    src = ast.unparse(elt)
    try:
        obj = eval(src, dict(global_vars or {}))  # pylint: disable=eval-used
    except Exception as e:  # noqa: BLE001
        raise QuadrantsSyntaxError(
            f"early_return=True: function {node.name!r}: cannot resolve return annotation element "
            f"{src!r} at rewrite time ({type(e).__name__}: {e}). Use a type expression resolvable "
            "from the function's globals (e.g. qd.f32 or a module-level vec/mat type)."
        ) from None
    origin = typing.get_origin(obj)
    if origin is tuple or (isinstance(obj, type) and issubclass(obj, tuple)):
        raise QuadrantsSyntaxError(
            f"early_return=True: function {node.name!r} returns {src!r}: nested-tuple returns are "
            "not supported by the single-exit rewrite. Flatten the tuple, or use a manual "
            "single-exit (flag-guard) function for this shape."
        )
    # late imports: lang/matrix.py sits above this module in the package graph
    from quadrants.lang.matrix import MatrixType

    if isinstance(obj, MatrixType):  # VectorType subclasses MatrixType
        return "compound", src
    from quadrants.lang.util import cook_dtype

    try:
        cook_dtype(obj)
    except Exception:  # noqa: BLE001
        raise QuadrantsSyntaxError(
            f"early_return=True: function {node.name!r} returns {src!r}: struct/custom-class "
            "returns are not supported by the single-exit rewrite. Use a manual single-exit "
            "(flag-guard) function for this shape."
        ) from None
    return "primitive", src


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

    def _rewrite_static_site(self, st, loop_depth):
        """Rewrite bare returns inside a `qd.static(...)` control node with the normal
        single-exit machinery (A32). Condition-value-agnostic by construction: this pass
        runs before the walk that evaluates qd.static, so it cannot (and need not) know
        TRUE/FALSE. TRUE branch: slot stores execute, everything after is guard-dead;
        FALSE branch: the walk prunes the branch, the stores never run. The node keeps
        its static test/iter so the walk still eliminates the dead branch. loop_depth is
        passed through unchanged - static control adds no runtime nesting, so break
        emission follows the enclosing runtime loop only."""
        st.body, _ = self.transform_block(st.body, loop_depth)
        if st.orelse:
            st.orelse, _ = self.transform_block(st.orelse, loop_depth)
        return _stamp_deep(st, getattr(st, "lineno", 1))

    def transform_block(self, stmts, loop_depth):
        out = []
        seen_return = False
        run = []  # plain post-return statements, merged into ONE guard
        for st in stmts:
            new_st = self.transform_stmt(st, loop_depth)
            if isinstance(new_st, list):
                # literal return at this level: everything after it is source-level dead.
                # First (unconditional) return stays at block level; a fall-through return
                # keeps its slot stores in the current run so they see names first-bound
                # there (the defect-2 fix).
                if seen_return:
                    run.extend(new_st)
                else:
                    out.extend(new_st)
                    seen_return = True
                break
            if _is_static_site(new_st) and _contains_return(new_st):
                # qd.static if/loop holding bare returns: left as-is they end up inside
                # runtime control flow at walk time - inside this block's own guard run
                # `if __er_done == 0:` when the site is post-return (ledger A32:
                # static-TRUE return after a runtime return), or inside the enclosing
                # runtime if when it sits pre-return. Rewrite unconditionally of
                # position; rewritten pre-return sites register as return-sites via
                # their new done-stores, which is exactly the followers-need-guards
                # semantics.
                new_st = self._rewrite_static_site(new_st, loop_depth)
            if seen_return:
                if _contains_done_store([new_st]):
                    # a sibling return-site (return nested in an if/for/while): it needs
                    # its OWN guard, and whatever follows needs a FRESH guard, because
                    # this statement may set __er_done when taken (a single merged body
                    # would keep executing past the taken return and clobber the slots)
                    if run:
                        out.append(self._guard(run))
                        run = []
                    out.append(self._guard([new_st]))
                else:
                    run.append(new_st)
            else:
                out.append(new_st)
                if _contains_done_store([new_st]):  # the first return-site of the block
                    seen_return = True
        if run:
            out.append(self._guard(run))
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

    # inventory of return sites (drives every validation below)
    value_sites = []  # (lineno, arity): arity = tuple width, or 1 for a scalar expression
    n_bare = 0
    for n in ast.walk(node):
        if isinstance(n, ast.Return):
            if n.value is None:
                n_bare += 1
            else:
                arity = len(n.value.elts) if isinstance(n.value, ast.Tuple) else 1
                value_sites.append((getattr(n, "lineno", 0), arity))

    has_value_return = bool(value_sites)
    has_bare_return = n_bare > 0

    if has_value_return and has_bare_return:
        raise QuadrantsSyntaxError(
            f"early_return=True: function {node.name!r} mixes bare `return` with `return <value>`; "
            "a bare return would silently leave the value slots at their initial values. Return "
            "the full slot set at every site, or make the function void (bare returns only)."
        )

    nret = 1
    if has_value_return:
        arities = {a for _, a in value_sites}
        if len(arities) > 1:
            sites = ", ".join(f"line {ln}: {a} value(s)" for ln, a in sorted(value_sites))
            raise QuadrantsSyntaxError(
                f"early_return=True: function {node.name!r} has inconsistent return arities "
                f"({sites}); every return site must return the same number of values, because a "
                "short return would silently leave the remaining slots at their init values"
            )
        nret = arities.pop()

    if has_value_return:
        if node.returns is None:
            raise QuadrantsSyntaxError(
                f"early_return=True: function {node.name!r} returns values but has no return "
                "annotation; add one (e.g. -> qd.f32 or -> tuple[qd.f32, qd.f32]) so the rewrite "
                "can type the return slots"
            )
        if nret == 1:
            if isinstance(node.returns, ast.Subscript) and isinstance(node.returns.slice, ast.Tuple):
                # `-> tuple[T]` spells a single slot as a 1-element tuple type
                ann_elts = list(node.returns.slice.elts)
            elif isinstance(node.returns, ast.Tuple):
                ann_elts = list(node.returns.elts)
            else:
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
        if len(ann_elts) != nret:
            raise QuadrantsSyntaxError(
                f"early_return=True: function {node.name!r} returns {nret} value(s) but its "
                f"annotation describes {len(ann_elts)} slot(s); a mismatch would silently "
                "leave the extra slots at their init values"
            )
    else:
        ann_elts = []

    if has_value_return:
        for i in range(nret):
            if _RET.format(i) in used:
                raise QuadrantsSyntaxError(
                    f"early_return=True: function {node.name!r} already uses the reserved name "
                    f"{_RET.format(i)!r}"
                )

    ctx = _Ctx(nret, is_kernel)
    rewriter = _Rewriter(ctx)
    body, _ = rewriter.transform_block(node.body, 0)

    # prologue: typed slot initializers + flag
    # NOTE: init via `qd.cast(0, qd.i32)` rather than a bare `= 0` constant. After func
    # inlining, a constant prologue store is treated as loop-invariant by the optimizer and
    # hoisted out of the caller's loop, leaking `__er_done` across calls (observed: serial
    # loop, second call skipped its body). The call-shaped init stays put. Same reason the
    # slots use `qd.cast`. Upstream issue: inlined-func prologue store over-hoisting.
    # Compound slots go through `qd.cast(T(0), T.dtype)`: T(0) is the broadcast constructor
    # (make_matrix_expr carries T's element dtype, so the slot is a correctly-shaped tensor)
    # and the cast keeps the init call-shaped with an explicit element dtype. (A plain
    # `qd.cast(0, T)` dies in cook_dtype: VectorType/MatrixType are not dtypes.)
    prologue = [_stamp_deep(
        ast.Assign(
            targets=[ast.Name(id=_DONE, ctx=ast.Store())],
            value=ast.parse("qd.cast(0, qd.i32)", mode="eval").body,
        ),
        1,
    )]
    for i in range(nret if has_value_return else 0):
        kind, src = _resolve_slot_type(ann_elts[i], node, global_vars)
        if kind == "compound":
            init_src = f"qd.cast({src}(0), {src}.dtype)"
        else:
            init_src = f"qd.cast(0, {src})"
        try:
            call = ast.parse(init_src, mode="eval").body
        except SyntaxError as e:
            raise QuadrantsSyntaxError(
                f"early_return=True: cannot synthesize slot initializer {init_src!r} for "
                f"{node.name!r}: {e}"
            )
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
