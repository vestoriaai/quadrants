#include "quadrants/transforms/auto_diff/auto_diff_common.h"
#include "quadrants/common/exceptions.h"
#include "quadrants/transforms/auto_diff/make_adjoint.h"

namespace quadrants::lang {

namespace {

// Generate the adjoint version of an independent block
// A65: first-touch classification of locals inside a loop body (program
// order). A local whose first access is a READ (incl. atomic read-modify-write)
// is a loop-carried accumulator; a local whose first access is a STORE is a
// per-iteration temporary. Descends into if/else bodies in program order.
class MakeAdjoint : public ADTransform {
 public:
  using ADTransform::visit;
  Block *current_block;
  Block *alloca_block;
  // Backup the forward pass (the forward pass might be modified during the MakeAdjoint) for search whether a
  // GlobalLoadStmt is inside a for-loop when allocating adjoint (see the function `adjoint`). Should be stored before
  // entering a for-loop body or an if-stmt, and restored after processing every statement in those two cases.
  Block *forward_backup;
  // IB root: stays constant across visitor recursion. Used when we need to allocate persistent storage that must
  // survive enclosing for-loop iterations (e.g. the dedicated ad-stacks that snapshot IfStmt conds in visit(IfStmt)).
  Block *ib_root;

  std::map<Stmt *, Stmt *> adjoint_stmt;

  // Stack capacity (config.ad_stack_size). The WhileStmt reverse visitor sizes its synthetic
  // iteration-trail stack with it and refuses to run in non-stack mode.
  int ad_stack_size_ = 0;

  explicit MakeAdjoint(Block *block, int ad_stack_size = 0) {
    current_block = nullptr;
    alloca_block = block;
    forward_backup = block;
    ib_root = block;
    ad_stack_size_ = ad_stack_size;
  }

  static void run(Block *block, int ad_stack_size = 0) {
    auto p = MakeAdjoint(block, ad_stack_size);
    block->accept(&p);
  }

  // Does `if_stmt`'s true/false body contain any AdStackPushStmt targeting `stack`? Recursive to catch pushes nested
  // inside further control flow (if-in-if, if-in-for). Used by visit(IfStmt) to gate cond-snapshotting. Must be narrow:
  // snapshotting every if-stmt would add an AdStackAllocaStmt per if, and determine_ad_stack_size cannot size stacks
  // whose push/pop pair is only reachable through branches its Bellman-Ford walk considers "unreached" - codegen then
  // aborts with "Adaptive autodiff stack's size should have been determined" and the extras also spam "Unused autodiff
  // stack should have been eliminated" for every untouched snap stack. Only when the body actually pushes onto the
  // cond's backing stack does BackupSSA's reverse-time clone of load_top read a post-body value rather than the forward
  // cond (the case this snapshot guards against); in every other case the clone is correct and a snapshot would be dead
  // weight.
  static bool block_pushes_to_stack(Block *block, Stmt *stack) {
    if (!block)
      return false;
    for (auto &stmt : block->statements) {
      if (auto *push = stmt->cast<AdStackPushStmt>()) {
        if (push->stack == stack)
          return true;
      }
      if (auto *inner_if = stmt->cast<IfStmt>()) {
        if (block_pushes_to_stack(inner_if->true_statements.get(), stack))
          return true;
        if (block_pushes_to_stack(inner_if->false_statements.get(), stack))
          return true;
      }
      if (auto *inner_for = stmt->cast<RangeForStmt>()) {
        if (block_pushes_to_stack(inner_for->body.get(), stack))
          return true;
      }
      if (auto *inner_for = stmt->cast<StructForStmt>()) {
        if (block_pushes_to_stack(inner_for->body.get(), stack))
          return true;
      }
    }
    return false;
  }

  static bool body_pushes_to_stack(IfStmt *if_stmt, Stmt *stack) {
    return block_pushes_to_stack(if_stmt->true_statements.get(), stack) ||
           block_pushes_to_stack(if_stmt->false_statements.get(), stack);
  }

  // TODO: current block might not be the right block to insert adjoint instructions!
  void visit(Block *block) override {
    std::vector<Stmt *> statements;
    // always make a copy since the list can be modified.
    for (auto &stmt : block->statements) {
      statements.push_back(stmt.get());
    }
    std::reverse(statements.begin(), statements.end());  // reverse-mode AD...
    for (auto stmt : statements) {
      current_block = block;
      stmt->accept(this);
    }
  }

  Stmt *insert_grad_stmt(std::unique_ptr<Stmt> &&stmt) override {
    auto ptr = stmt.get();
    current_block->insert(std::move(stmt), -1);
    return ptr;
  }

  // Accumulate [value] to the adjoint of [primal]
  void accumulate(Stmt *primal, Stmt *value) {
    auto alloca_ = adjoint(primal);
    if (!alloca_ || alloca_->is<ConstStmt>()) {
      return;  // primal may be int variable
    }
    if (alloca_->is<AdStackAllocaStmt>()) {
      auto alloca = alloca_->cast<AdStackAllocaStmt>();
      if (is_real(alloca->ret_type.get_element_type())) {
        insert<AdStackAccAdjointStmt>(alloca, load(value));
      }
    } else {
      QD_ASSERT(alloca_->is<AllocaStmt>());
      auto alloca = alloca_->as<AllocaStmt>();
      auto local_load = insert<LocalLoadStmt>(alloca);
      local_load->ret_type = alloca->ret_type.ptr_removed();
      insert<LocalStoreStmt>(alloca, add(local_load, value));
    }
  }

  Stmt *adjoint(Stmt *stmt) {
    DataType adjoint_dtype = stmt->ret_type.ptr_removed();
    if (adjoint_dtype->is<TensorType>()) {
      DataType prim_dtype = PrimitiveType::f32;
      if (is_real(adjoint_dtype.get_element_type())) {
        prim_dtype = adjoint_dtype.get_element_type();
      }
      adjoint_dtype =
          TypeFactory::get_instance().get_tensor_type(adjoint_dtype->as<TensorType>()->get_shape(), prim_dtype);
    } else if (stmt->is<MatrixPtrStmt>()) {
      // pass
    } else if (!is_real(adjoint_dtype) || stmt->is<ConstStmt>()) {
      return constant(0);
    }

    if (adjoint_stmt.find(stmt) == adjoint_stmt.end()) {
      // normal SSA cases

      // Create the alloca. Using the statement's own `ret_type` tends to fit better than the kernel-wide
      // `get_current_program().config.gradient_dt` default.
      auto alloca = Stmt::make<AllocaStmt>(adjoint_dtype);
      adjoint_stmt[stmt] = alloca.get();

      // We need to insert the alloca in the block of GlobalLoadStmt when the GlobalLoadStmt is not inside the
      // currently-processed range-for. Code sample (a and b require grad):
      //
      // Case 1 (GlobalLoadStmt is outside the for-loop, compute 5 times and accumulate once, alloca history value
      // is needed):
      // for i in range(5):
      //     p = a[i]
      //     q = b[i]
      //     for _ in range(5)
      //         q += p

      // Case 2 (GlobalLoadStmt is inside the for-loop, compute once and accumulate immediately, alloca history
      // value can be discarded):
      // for i in range(5):
      //     q = b[i]
      //     for _ in range(5)
      //         q += a[i]
      if (stmt->is<GlobalLoadStmt>() && forward_backup->locate(stmt->as<GlobalLoadStmt>()) == -1) {
        // Case 1: the GlobalLoadStmt lives in a block outside the currently-processed range-for iteration. Its adjoint
        // must persist across all iterations of the inner reversed loop, so the alloca cannot live in the current
        // alloca_block (which would be the inner reversed loop body). Walk up from the primal's enclosing block until
        // we hit one whose owning statement unconditionally dominates both the forward and the reverse code (a loop /
        // offloaded / kernel body, not an if / while body): visit(IfStmt) emits the reverse code into a brand new
        // sibling IfStmt, not back into the forward if-body, so an alloca placed inside the forward branch is
        // SSA-invalid from the reverse branch's point of view and gets DCE'd into silently-zero gradients.
        Block *target = stmt->as<GlobalLoadStmt>()->parent;
        while (target != nullptr) {
          Stmt *parent_stmt = target->parent_stmt();
          if (parent_stmt == nullptr || parent_stmt->is<RangeForStmt>() || parent_stmt->is<StructForStmt>() ||
              parent_stmt->is<OffloadedStmt>() || parent_stmt->is<MeshForStmt>()) {
            break;
          }
          target = parent_stmt->parent;
        }
        // Reaching a null target means the primal's enclosing-block chain is broken (an unparented block). Falling back
        // to alloca_block here would place the adjoint inside a branch that gets DCE'd on the reverse side (silently
        // zeroed gradients); hard-assert instead so malformed IR surfaces loudly.
        QD_ASSERT(target != nullptr);
        target->insert(std::move(alloca), 0);
      } else {
        alloca_block->insert(std::move(alloca), 0);
      }
    }
    return adjoint_stmt[stmt];
  }

  // For ops in `NonLinearOps::unary_collections` the reverse formula must NOT read the forward `stmt` directly - only
  // `stmt->operand` (adstack-backed), `adjoint(stmt)`, and constants. BackupSSA spills the forward stmt to a single
  // plain alloca overwritten each iteration, so reading `stmt` from a reversed dynamic loop would use the
  // last-iteration value regardless of which reverse iteration is running. This helper walks the value-tree at
  // IR-transform time and asserts the invariant. It covers the formula-reads-forward-stmt half of the per-op
  // classification check; the missing-from-unary_collections half is covered by a Python-side audit of the
  // unary_collections set.
  void accumulate_unary_operand_checked(UnaryOpStmt *stmt, Stmt *value) {
    if (NonLinearOps::unary_collections.find(stmt->op_type) != NonLinearOps::unary_collections.end()) {
      std::unordered_set<const Stmt *> visited;
      std::function<void(const Stmt *)> walk = [&](const Stmt *s) {
        if (!s || visited.count(s))
          return;
        visited.insert(s);
        QD_ASSERT_INFO(s != stmt,
                       "MakeAdjoint adjoint formula for UnaryOpType::{} reads the forward stmt directly. It "
                       "must read `stmt->operand` (adstack-backed) instead - see the tan/tanh/exp branches "
                       "for the recompute pattern.",
                       unary_op_type_name(stmt->op_type));
        for (auto *op : s->get_operands())
          walk(op);
      };
      walk(value);
    }
    accumulate(stmt->operand, value);
  }

  void visit(UnaryOpStmt *stmt) override {
    auto acc = [&](Stmt *value) { accumulate_unary_operand_checked(stmt, value); };
    if (stmt->op_type == UnaryOpType::floor || stmt->op_type == UnaryOpType::ceil) {
      // do nothing
    } else if (stmt->op_type == UnaryOpType::neg) {
      accumulate(stmt->operand, negate(adjoint(stmt)));
    } else if (stmt->op_type == UnaryOpType::abs) {
      acc(mul(adjoint(stmt), sgn(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::sin) {
      acc(mul(adjoint(stmt), cos(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::cos) {
      acc(negate(mul(adjoint(stmt), sin(stmt->operand))));
    } else if (stmt->op_type == UnaryOpType::tan) {
      // d/dx tan(x) = 1 + tan(x)^2. Recompute tan(operand) rather than reusing the forward value: the primal is
      // per-iteration inside dynamic loops but BackupSSA only spills forward values to a single plain alloca, so
      // reading the forward tan would use the last-iteration value in the reversed loop. The operand, in contrast,
      // rides the adstack through its LocalLoad, so a fresh tan on it is per-iteration correct.
      acc(mul(adjoint(stmt), add(constant(1, stmt->ret_type), sqr(tan(stmt->operand)))));
    } else if (stmt->op_type == UnaryOpType::tanh) {
      // Recompute tanh(operand) in the reverse pass instead of reusing the forward stmt value. In dynamic loops
      // BackupSSA spills the forward stmt to a single plain alloca overwritten each iteration, so the reversed loop
      // would read the last-iteration tanh for every backward step. The operand rides the adstack through LocalLoad, so
      // a fresh tanh on it is per-iteration correct. Trade-off: tanh is evaluated twice per iteration (once forward,
      // once backward).
      acc(mul(adjoint(stmt), sub(constant(1, stmt->ret_type), sqr(tanh(stmt->operand)))));
    } else if (stmt->op_type == UnaryOpType::asin) {
      acc(mul(adjoint(stmt),
              div(constant(1, stmt->ret_type), sqrt(sub(constant(1, stmt->ret_type), sqr(stmt->operand))))));
    } else if (stmt->op_type == UnaryOpType::acos) {
      acc(mul(adjoint(stmt),
              negate(div(constant(1, stmt->ret_type), sqrt(sub(constant(1, stmt->ret_type), sqr(stmt->operand)))))));
    } else if (stmt->op_type == UnaryOpType::exp) {
      // See the tanh case above: recompute exp on the adstack-backed operand so the reversed loop sees the
      // per-iteration value rather than the last-forward value spilled by BackupSSA. Same trade-off as tanh: exp is
      // evaluated twice per iteration (once forward, once backward).
      acc(mul(adjoint(stmt), exp(stmt->operand)));
    } else if (stmt->op_type == UnaryOpType::log) {
      // No recompute workaround needed: the reverse formula `1 / operand` reads `stmt->operand` directly (which is
      // adstack-backed via LocalLoad inside dynamic loops), not the forward `log(operand)` stmt value.
      acc(div(adjoint(stmt), stmt->operand));
    } else if (stmt->op_type == UnaryOpType::sqrt) {
      // No recompute workaround needed: the reverse formula reads `stmt->operand` (adstack-backed via LocalLoad inside
      // dynamic loops, gated on `unary_collections` membership) and recomputes `sqrt(operand)` from it, not the forward
      // `sqrt(operand)` stmt value. Structure mirrors log above.
      acc(mul(adjoint(stmt), div(constant(0.5f, stmt->ret_type), sqrt(stmt->operand))));
    } else if (stmt->op_type == UnaryOpType::rsqrt) {
      acc(mul(adjoint(stmt),
              mul(constant(-0.5f, stmt->ret_type), pow(rsqrt(stmt->operand), constant(3, stmt->ret_type)))));
    } else if (stmt->op_type == UnaryOpType::cast_value) {
      if (is_real(stmt->cast_type.get_element_type()) && is_real(stmt->operand->ret_type.get_element_type())) {
        accumulate(stmt->operand, adjoint(stmt));
      }
    } else if (stmt->op_type == UnaryOpType::logic_not) {
      // do nothing
    } else if (stmt->op_type == UnaryOpType::round) {
      // d(round(x)) = 0 almost everywhere (piecewise-constant): the adjoint
      // contributes nothing. Matches autodiff conventions for quantization.
    } else {
      QD_P(unary_op_type_name(stmt->op_type));
      QD_NOT_IMPLEMENTED;
    }
  }

  void visit(BinaryOpStmt *bin) override {
    if (bin->op_type == BinaryOpType::add) {
      accumulate(bin->lhs, adjoint(bin));
      accumulate(bin->rhs, adjoint(bin));
    } else if (bin->op_type == BinaryOpType::sub) {
      accumulate(bin->lhs, adjoint(bin));
      accumulate(bin->rhs, negate(adjoint(bin)));
    } else if (bin->op_type == BinaryOpType::mul) {
      // d (x * y) = y * dx + x * dy
      accumulate(bin->lhs, mul(adjoint(bin), bin->rhs));
      accumulate(bin->rhs, mul(adjoint(bin), bin->lhs));
    } else if (bin->op_type == BinaryOpType::mod) {
      // Do nothing
    } else if (bin->op_type == BinaryOpType::div) {
      accumulate(bin->lhs, div(adjoint(bin), bin->rhs));
      accumulate(bin->rhs, negate(div(mul(adjoint(bin), bin->lhs), mul(bin->rhs, bin->rhs))));
    } else if (bin->op_type == BinaryOpType::atan2) {
      auto numerator = add(sqr(bin->lhs), sqr(bin->rhs));
      accumulate(bin->lhs, div(mul(adjoint(bin), bin->rhs), numerator));
      accumulate(bin->rhs, negate(div(mul(adjoint(bin), bin->lhs), numerator)));
    } else if (bin->op_type == BinaryOpType::pow) {
      // d (x ^ y) = x ^ (y-1) * (y * dx + log(x) * x * dy)
      auto common_coeff = pow(bin->lhs, sub(bin->rhs, constant(1, bin->ret_type)));  // x ^ (y-1)
      accumulate(bin->lhs, mul(adjoint(bin), mul(bin->rhs, common_coeff)));
      accumulate(bin->rhs, mul(adjoint(bin), mul(log(bin->lhs), mul(bin->lhs, common_coeff))));
    } else if (bin->op_type == BinaryOpType::min || bin->op_type == BinaryOpType::max) {
      auto cmp = bin->op_type == BinaryOpType::min ? cmp_lt(bin->lhs, bin->rhs) : cmp_lt(bin->rhs, bin->lhs);
      auto zero = insert_const_for_grad(bin->ret_type, bin, 0);
      accumulate(bin->lhs, sel(cmp, adjoint(bin), zero));
      accumulate(bin->rhs, sel(cmp, zero, adjoint(bin)));
    } else if (bin->op_type == BinaryOpType::floordiv) {
      // do nothing
    } else if (is_comparison(bin->op_type) || is_bit_op(bin->op_type) || binary_is_logical(bin->op_type)) {
      // do nothing

    } else {
      QD_WARN("gradient of binary op {}\n{}", binary_op_type_name(bin->op_type), bin->get_tb());
      QD_NOT_IMPLEMENTED;
    }
  }

  void visit(TernaryOpStmt *stmt) override {
    QD_ASSERT(stmt->op_type == TernaryOpType::select);
    auto zero = insert_const_for_grad(stmt->ret_type, stmt, 0);
    accumulate(stmt->op2, insert<TernaryOpStmt>(TernaryOpType::select, stmt->op1, load(adjoint(stmt)), zero));
    accumulate(stmt->op3, insert<TernaryOpStmt>(TernaryOpType::select, stmt->op1, zero, load(adjoint(stmt))));
  }

  void visit(IfStmt *if_stmt) override {
    // Snapshot a stack-backed forward cond into a dedicated 1-push-per-if-execution ad-stack, but only when the cond's
    // backing stack is also pushed inside the if body (e.g. short-circuit lowering pushes the rhs of `&&` onto the same
    // stack that holds the cond). Without this, BackupSSA's clone of `if_stmt->cond` in the reverse block reads the
    // cond stack AFTER the body-pushes rather than the forward-time cond value - the reverse IfStmt flips, pop counts
    // drift, and gradients come out silently zero. A dedicated stack has exactly one push per forward if-execution, so
    // the reverse load_top matches the forward cond.
    //
    // Guarded by the body-push check because snapshotting indiscriminately adds AdStackAllocaStmts that go through
    // `determine_ad_stack_size` unused on every other if-stmt in the kernel - the adaptive-size pass emits "Unused
    // autodiff stack should have been eliminated" warnings and the codegen step then fails with "Adaptive autodiff
    // stack's size should have been determined".
    Stmt *reverse_cond = if_stmt->cond;
    AdStackAllocaStmt *snap_stack_ptr = nullptr;
    // Narrow guard: only the bare `AdStackLoadTopStmt` shape needs the explicit snapshot below. A compound cond (e.g.
    // `cmp_lt(load_top(x_stack) + 0.1, threshold)` from `if x + 0.1 < threshold` when `x` has been promoted to an
    // adstack by `ReplaceLocalVarWithStacks`) reaches this visitor as a BARE `AdStackLoadTopStmt` cond anyway - the cmp
    // / arithmetic value goes through `PromoteSSA2LocalVar`'s required-defs set (the IfStmt cond path adds the cond's
    // value-producing op), then `AdStackAllocaJudger::visit(IfStmt)` marks its alloca stack-needed because it feeds the
    // cond, then `ReplaceLocalVarWithStacks` promotes the alloca to an adstack. By the time control reaches here,
    // `if_stmt->cond` is `AdStackLoadTopStmt` of that snap-promoted adstack and the body of the if does NOT push to
    // that stack (the cmp value is pushed once just before the IfStmt, never inside), so the `body_pushes_to_stack`
    // guard below is false and we correctly skip the additional snap-stack. Per-iter cond values are preserved by the
    // alloca-promotion pipeline.
    //
    // The bare-`AdStackLoadTopStmt` case the snap-stack below handles is the OTHER shape: a load_top whose backing
    // stack IS pushed to inside the if body (e.g. short-circuit lowering of `&&` pushes the rhs onto the same stack
    // that holds the cond). Without the snap-stack, `BackupSSA::generic_visit`'s clone-branch for `AdStackLoadTopStmt`
    // emits a fresh `AdStackLoadTopStmt` at the reverse cursor, which then reads the post-body-push top and sees the
    // wrong cond value. The snap-stack here decouples the cond value from the body's pushes by capturing it once just
    // before the IfStmt and reading it back at the matching reverse cursor.
    //
    // Compound conds whose value-producing op is not adstack-promoted by the alloca-promotion pipeline fall through to
    // `BackupSSA::generic_visit`'s `load(op)` spill: a single alloca written immediately after the forward cond and
    // reloaded at reverse-cond construction. That captures the forward-time cond value correctly within one iteration
    // of an enclosing dynamic loop.
    if (if_stmt->cond->is<AdStackLoadTopStmt>()) {
      auto *cond_stack = if_stmt->cond->as<AdStackLoadTopStmt>()->stack->as<AdStackAllocaStmt>();
      if (body_pushes_to_stack(if_stmt, cond_stack)) {
        auto cond_type = if_stmt->cond->ret_type.ptr_removed();
        // Size the snap stack the same way as the cond stack it mirrors: one forward push per if-execution matched by
        // one reverse pop. Reusing cond_stack->max_size keeps the snap stack exempt from `determine_ad_stack_size` when
        // the cond stack itself was built with a fixed size, which holds when ReplaceLocalVarWithStacks ran with a
        // non-zero `ad_stack_size` (the only configuration supported for stack-based reverse AD).
        auto snap_stack = Stmt::make<AdStackAllocaStmt>(cond_type, cond_stack->max_size);
        snap_stack_ptr = snap_stack->as<AdStackAllocaStmt>();
        // Allocate at the IB root so the stack persists across enclosing for-loop iterations.
        ib_root->insert(std::move(snap_stack), 0);
        // Per-execution forward push of the cond value, just before the forward if-stmt. No initial zero push: the
        // reverse load_top always runs after a matching forward push, so leaving the stack empty at entry is both
        // correct and avoids a dead store that DSE would otherwise drop (and that `determine_ad_stack_size` would then
        // miscount).
        if_stmt->insert_before_me(Stmt::make<AdStackPushStmt>(snap_stack_ptr, if_stmt->cond));
        // Per-execution reverse load of the snapshotted cond, emitted in the current reverse block.
        reverse_cond = insert<AdStackLoadTopStmt>(snap_stack_ptr);
        reverse_cond->ret_type = cond_type;
      }
    }

    auto new_if = Stmt::make_typed<IfStmt>(reverse_cond);
    if (if_stmt->true_statements) {
      new_if->set_true_statements(std::make_unique<Block>());
      auto old_current_block = current_block;
      // Backup forward pass
      forward_backup = if_stmt->true_statements.get();

      current_block = new_if->true_statements.get();
      for (int i = if_stmt->true_statements->statements.size() - 1; i >= 0; i--) {
        if_stmt->true_statements->statements[i]->accept(this);
        // Restore forward pass
        forward_backup = if_stmt->true_statements.get();
      }

      current_block = old_current_block;
    }
    if (if_stmt->false_statements) {
      new_if->set_false_statements(std::make_unique<Block>());
      auto old_current_block = current_block;

      // Backup forward pass
      forward_backup = if_stmt->false_statements.get();

      current_block = new_if->false_statements.get();
      for (int i = if_stmt->false_statements->statements.size() - 1; i >= 0; i--) {
        if_stmt->false_statements->statements[i]->accept(this);
        // Restore forward pass
        forward_backup = if_stmt->false_statements.get();
      }
      current_block = old_current_block;
    }
    insert_grad_stmt(std::move(new_if));
    if (snap_stack_ptr) {
      // One pop per reverse if-execution, paired with the forward push above.
      insert<AdStackPopStmt>(snap_stack_ptr);
    }
  }

  // A65 专项②: scoped WhileStmt reverse — "linear control-replay" form.
  // Sound when: (1) every loop-carried alloca (stored inside the loop body,
  // allocated outside it) is NON-gradient (int/bool control state); (2) the
  // loop condition and in-loop control (break/continue) depend only on such
  // state; (3) gradient flows through the body via grad-array loads + atomic
  // accumulations (self-adjoint) and LINEAR value chains. The reverse = the
  // forward while STAYS in the IR (primal replay: per-iteration pushes of
  // promoted guard/carried values + a synthetic iteration-trail marker) and
  // the adjoint while terminates on the trail sentinel, with the body
  // visited by the standard per-statement adjoint machinery in REVERSE
  // statement order (consumer adjoints emit before producers - same rule as
  // visit(RangeForStmt); accumulations into adjoints are order-free). Each
  // replay iteration consumes exactly one executed iteration's push block in
  // LIFO order. Real-typed carried allocas need per-iteration primal stacks
  // (adstack-for-while) — future work; rejected loudly.
  void visit(WhileStmt *while_stmt) override {
    auto *body = while_stmt->body.get();

    // 1. collect allocas stored inside the loop body
    std::unordered_set<Stmt *> stored_inside;
    {
      class StoreCollector : public BasicStmtVisitor {
       public:
        using BasicStmtVisitor::visit;
        std::unordered_set<Stmt *> dests;
        void mark(Stmt *s) {
          if (auto *al = s->cast<AllocaStmt>())
            dests.insert(al);
          else if (auto *mp = s->cast<MatrixPtrStmt>()) {
            if (auto *al2 = mp->origin->cast<AllocaStmt>())
              dests.insert(al2);
          }
        }
        void visit(LocalStoreStmt *stmt) override {
          mark(stmt->dest);
        }
        void visit(LocalLoadStmt *stmt) override {
          mark(stmt->src);
        }
        void visit(AtomicOpStmt *stmt) override {
          mark(stmt->dest);
        }
      };
      StoreCollector sc;
      body->accept(&sc);
      stored_inside = std::move(sc.dests);
    }

    // 2. carried allocas = stored inside body, allocated outside it
    std::vector<AllocaStmt *> carried_allocas;
    for (auto *al : stored_inside) {
      bool outside = false;
      for (Block *b = al->parent; b != nullptr; b = b->parent_block()) {
        if (b == body) {
          outside = false;
          break;
        }
        if (b == while_stmt->parent) {
          outside = true;
          break;
        }
      }
      if (outside)
        carried_allocas.push_back(al->as<AllocaStmt>());
    }

    // 3. applicability: carried allocas must be non-real (int/bool)
    for (auto *al : carried_allocas) {
      if (is_real(al->ret_type.get_element_type())) {
        throw QuadrantsError(
            "[a65] WhileStmt reverse: a REAL-typed local variable is carried "
            "across loop iterations (needs per-iteration primal stacks - "
            "adstack-for-while, future work). Restructure the kernel or "
            "exclude it from autodiff. Alloca id=" +
            std::to_string(al->id));
      }
    }

    // 4. entry-state restore sources: nearest pre-loop LocalStore per carried
    //    alloca, scanned along the DOMINATING block chain (program order):
    //    container -> ... -> root. A while nested in an if-branch has its init
    //    stores in enclosing blocks - the direct-parent-only scan misses them.
    Block *container = while_stmt->parent;
    int while_pos = -1;
    for (int i = 0; i < (int)container->statements.size(); i++) {
      if (container->statements[i].get() == while_stmt) {
        while_pos = i;
        break;
      }
    }
    // build the chain [(block, scan_limit)] from root down to container
    std::vector<std::pair<Block *, int>> chain;
    {
      std::vector<std::pair<Block *, int>> rev;
      Block *b = container;
      int limit = while_pos;
      rev.push_back({b, limit});
      while (true) {
        Stmt *ps = b->parent_stmt();
        if (ps == nullptr)
          break;
        Block *pb = ps->parent;
        if (pb == nullptr)
          break;
        int ppos = -1;
        for (int i = 0; i < (int)pb->statements.size(); i++) {
          if (pb->statements[i].get() == ps) {
            ppos = i;
            break;
          }
        }
        if (ppos < 0)
          break;
        rev.push_back({pb, ppos});
        b = pb;
      }
      for (auto it = rev.rbegin(); it != rev.rend(); ++it)
        chain.push_back(*it);
    }
    // program-order scan along the chain; the LAST store before the while wins
    std::unordered_map<Stmt *, LocalStoreStmt *> init_stores;
    for (auto &entry : chain) {
      Block *b = entry.first;
      int scan_to = std::min(entry.second, (int)b->statements.size());
      for (int i = 0; i < scan_to; i++) {
        auto &s = b->statements[i];
        if (auto *ls = s->cast<LocalStoreStmt>()) {
          if (auto *al = ls->dest->cast<AllocaStmt>()) {
            if (stored_inside.count(al))
              init_stores[al] = ls;
          }
        }
      }
    }

    // 5. termination + emission. The adjoint while terminates via a synthetic
    //    iteration-trail stack: the forward while (kept in the IR, executing
    //    the primal replay) pushes an i32 1-marker immediately after its
    //    lowered termination-if - once per body-executing iteration - and the
    //    trail carries an initial 0 sentinel (mirroring
    //    ReplaceLocalVarWithStacks' initial push). The adjoint while keeps
    //    only the lowered head latch and peeks the trail at its top: LIFO
    //    order makes replay iteration k consume exactly the k-th-from-last
    //    executed iteration's push block (its marker via the walk-emitted
    //    trail-pop, its body pushes via the per-site adjoint pops), and the
    //    sentinel surfacing terminates the loop. This needs no assumption on
    //    the lowered guard shape beyond the existence of the termination-if,
    //    and cannot loop forever because the forward pass pushes exactly one
    //    marker per executed iteration.
    if (ad_stack_size_ <= 0) {
      throw QuadrantsError(
          "[a65] WhileStmt reverse requires the adstack (use_stack) pipeline");
    }
    // locate the lowered termination-if: an IfStmt in the forward body with a
    // WhileControlStmt break in either branch (the lowered form puts the
    // break's `while control nullptr, const-0` inside an IfStmt branch; the
    // head latch's WhileControlStmt sits at body top level, outside any if).
    Stmt *term_if = nullptr;
    {
      auto branch_has_break = [](Block *b) {
        if (b == nullptr)
          return false;
        for (auto &s : b->statements)
          if (s->is<WhileControlStmt>())
            return true;
        return false;
      };
      for (int i = 0; i < (int)body->statements.size(); i++) {
        auto *is = body->statements[i]->cast<IfStmt>();
        if (is == nullptr)
          continue;
        if (branch_has_break(is->true_statements.get()) ||
            branch_has_break(is->false_statements.get())) {
          term_if = is;
          break;
        }
      }
    }
    QD_ASSERT_INFO(term_if != nullptr,
                   "[a65] WhileStmt reverse: lowered while has no termination-if (no WhileControlStmt break "
                   "under any IfStmt branch)");

    // synthetic iteration-trail stack: 0 sentinel + one 1-marker per executed
    // iteration.
    auto trail = Stmt::make<AdStackAllocaStmt>(PrimitiveType::i32, ad_stack_size_);
    auto *trail_ptr = trail->as<AdStackAllocaStmt>();
    ib_root->insert(std::move(trail), 0);
    auto trail_zero = insert_const(PrimitiveType::i32, trail_ptr, 0);
    trail_zero->insert_after_me(Stmt::make<AdStackPushStmt>(trail_ptr, trail_zero));
    auto trail_one = insert_const(PrimitiveType::i32, term_if, 1);
    trail_one->insert_after_me(Stmt::make<AdStackPushStmt>(trail_ptr, trail_one));

    auto new_while = while_stmt->clone();
    auto *new_while_ptr = new_while->as<WhileStmt>();
    auto *adj_body = new_while_ptr->body.get();
    // keep only the head latch (everything up to and including the first
    // WhileControlStmt); the body proper is regenerated by the adjoint walk.
    int latch_end = -1;
    for (int i = 0; i < (int)adj_body->size(); i++) {
      if (adj_body->statements[i]->is<WhileControlStmt>()) {
        latch_end = i;
        break;
      }
    }
    QD_ASSERT_INFO(latch_end >= 0,
                   "[a65] WhileStmt reverse: lowered while has no WhileControlStmt latch");
    const int blen = adj_body->size();
    for (int i = latch_end + 1; i < blen; i++)
      adj_body->erase(latch_end + 1);
    // trail termination head: peek the trail; break when the 0 sentinel has
    // surfaced (all executed iterations consumed).
    auto trail_peek = adj_body->insert(Stmt::make<AdStackLoadTopStmt>(trail_ptr), adj_body->size());
    trail_peek->ret_type = PrimitiveType::i32;
    auto sentinel = adj_body->insert(
        Stmt::make_typed<ConstStmt>(TypedConstant(PrimitiveType::i32, 0)), adj_body->size());
    auto reached_sentinel = adj_body->insert(
        Stmt::make_typed<BinaryOpStmt>(BinaryOpType::cmp_eq, trail_peek, sentinel), adj_body->size());
    reached_sentinel->ret_type = PrimitiveType::u1;
    auto exit_zero = adj_body->insert(
        Stmt::make_typed<ConstStmt>(TypedConstant(PrimitiveType::i32, 0)), adj_body->size());
    auto break_if = Stmt::make_typed<IfStmt>(reached_sentinel);
    break_if->set_true_statements(std::make_unique<Block>());
    break_if->true_statements->insert(Stmt::make<WhileControlStmt>(nullptr, exit_zero), 0);
    adj_body->insert(std::move(break_if), adj_body->size());

    for (auto &kv : init_stores) {
      auto cloned_store = kv.second->clone();
      insert_grad_stmt(std::move(cloned_store));
    }
    insert_grad_stmt(std::move(new_while));

    auto old_alloca_block = alloca_block;
    auto old_current_block = current_block;
    auto old_forward_backup = forward_backup;
    forward_backup = body;
    // iterate a COPY of the statement pointers: nested WhileStmt visitors
    // erase from body->statements during the walk (live-list iteration would
    // invalidate the iterator). Reverse statement order - same rationale as
    // visit(RangeForStmt): consumer adjoints must emit before producer
    // adjoints so the chain-rule reads of an adjoint slot see the full
    // accumulated contribution. The walk snapshot is taken AFTER the trail
    // marker push was inserted above, so the trail's adjoint pop is emitted
    // by the standard visit(AdStackPushStmt) machinery - it lands after the
    // body adjoints in each replay iteration (reverse statement order),
    // right before the next head peek.
    std::vector<Stmt *> statements;
    for (auto &stmt : body->statements)
      statements.push_back(stmt.get());
    std::reverse(statements.begin(), statements.end());
    for (auto stmt : statements) {
      alloca_block = adj_body;
      current_block = adj_body;
      stmt->accept(this);
      forward_backup = body;
    }
    alloca_block = old_alloca_block;
    current_block = old_current_block;
    forward_backup = old_forward_backup;

    // A68: the forward while MUST stay in the IR tree (same protocol as
    // visit(RangeForStmt), which never erases for_stmt). The adjoint emissions
    // above legitimately share forward-body statements as operands - promoted
    // guard/value stacks (pushes live in the forward body), IfStmt conds
    // (reverse_cond = if_stmt->cond), and snap-stack pushes inserted before
    // forward ifs. Erasing the forward while left those operands dangling:
    // erase() marks statements erased and the later DIE pass frees them, so
    // references decayed into use-after-free (observed as a GlobalLoadStmt
    // read of an unrelated stack operand in scalarize). With the forward
    // while kept, the reverse kernel runs forward-while pushes (per-iteration
    // guard + carried-value stacks), the entry-restores above rewind the
    // carried allocas, and the adjoint while replays LIFO (iterations N..1),
    // each replay iteration popping exactly one iteration's push block -
    // matching the range-for protocol.
  }

  // A65 专项②: break/continue in control-replay whiles need no adjoint
  // emission of their own - the cloned while's condition re-evaluates the
  // same non-gradient control state per replay iteration, so the break/continue
  // points match the forward trace exactly. (Linear bodies only; see
  // visit(WhileStmt) soundness notes.)
  void visit(WhileControlStmt *stmt) override {
    // do nothing (no gradient through non-gradient control flow)
  }

  void visit(ContinueStmt *stmt) override {
    // do nothing
  }

  void visit(RangeForStmt *for_stmt) override {
    auto new_for = for_stmt->clone();
    auto new_for_ptr = new_for->as<RangeForStmt>();
    new_for_ptr->reversed = !new_for_ptr->reversed;
    insert_grad_stmt(std::move(new_for));
    const int len = new_for_ptr->body->size();

    for (int i = 0; i < len; i++) {
      new_for_ptr->body->erase(0);
    }

    std::vector<Stmt *> statements;
    // always make a copy since the list can be modified.
    for (auto &stmt : for_stmt->body->statements) {
      statements.push_back(stmt.get());
    }
    std::reverse(statements.begin(), statements.end());  // reverse-mode AD...
    auto old_alloca_block = alloca_block;
    auto old_current_block = current_block;
    auto old_forward_backup = forward_backup;  // store the block which is not inside the current IB,
                                               // such as outer most loop
    // Backup the forward pass
    forward_backup = for_stmt->body.get();
    for (auto stmt : statements) {
      alloca_block = new_for_ptr->body.get();
      current_block = new_for_ptr->body.get();
      stmt->accept(this);
      // Restore the forward pass
      forward_backup = for_stmt->body.get();
    }
    // Restore current_block: if this RangeForStmt is visited from within another compound stmt (notably visit(IfStmt)),
    // that outer visitor continues iterating its own body in reverse after we return and emit further reverse stmts.
    // Without the restore, those emissions would land in the reversed-for's body instead of the outer block, producing
    // silently-wrong gradients whenever a runtime-guarded if wraps a for-loop with loop-carried variables (the reverse
    // loop body would over-pop the adstack and emit the x.grad accumulation on every iteration instead of once).
    current_block = old_current_block;
    forward_backup = old_forward_backup;
    alloca_block = old_alloca_block;
  }

  void visit(StructForStmt *for_stmt) override {
    // Save/restore mirrors visit(RangeForStmt) above. Rationale: visit(Block) inside `body->accept(this)` sets
    // current_block = for_stmt->body at the start of every iteration, so on return current_block points at the
    // struct-for's body. An enclosing compound visitor (e.g. visit(IfStmt)) that resumes iterating its children in
    // reverse after this StructForStmt needs current_block and alloca_block to still be its own, not this for's;
    // otherwise subsequent reverse emissions land inside the struct-for body and any adjoint alloca lives in a block
    // the enclosing if-branch cannot reach. forward_backup must be saved too because `visit(IfStmt)` mutates it
    // without restoring, so a nested if inside the struct-for body leaves it pointing at the if-branch block, which
    // then survives past this visitor and mis-routes `adjoint()` on GlobalLoadStmts for later siblings at the
    // enclosing scope.
    auto old_alloca_block = alloca_block;
    auto old_current_block = current_block;
    auto old_forward_backup = forward_backup;
    alloca_block = for_stmt->body.get();
    for_stmt->body->accept(this);
    current_block = old_current_block;
    alloca_block = old_alloca_block;
    forward_backup = old_forward_backup;
  }

  // Equivalent to AdStackLoadTopStmt when no stack is needed
  void visit(LocalLoadStmt *stmt) override {
    // QD_ASSERT(!needs_grad(stmt->ret_type));
    if (is_real(stmt->ret_type.get_element_type()))
      accumulate(stmt->src, load(adjoint(stmt)));
  }

  // Equivalent to AdStackPushStmt when no stack is needed
  void visit(LocalStoreStmt *stmt) override {
    // A65 note: loop-carried head accumulators are rejected at the FRONTEND
    // (reverse_segments CarriedDestFinder: head-assigned local written inside
    // the top-level loop, incl. atomic-dest form). First-touch classification
    // here is unreliable - store-to-load forwarding + DCE remove init stores,
    // making per-iteration temporaries look read-first (false positives on
    // e.g. _small_cholesky_factorize_solve_block).
    accumulate(stmt->val, load(adjoint(stmt->dest)));

    // Clear the adjoint of the dest after local store, because LocalStoreStmt overwrites the dest:
    // 1. If the alloca is inside a loop, the adjoint of this alloca for this iteration must be cleared once the
    //    iteration completes.
    // 2. If the alloca serves as the dest of multiple LocalStoreStmts, only the last LocalStoreStmt counts.
    auto dest_type = stmt->dest->ret_type.ptr_removed();
    if (is_real(dest_type.get_element_type())) {
      auto dtype = dest_type;
      auto zero = insert_const_for_grad(dtype, stmt, 0);
      insert<LocalStoreStmt>(adjoint(stmt->dest), zero);
    }
  }

  void visit(AdStackLoadTopStmt *stmt) override {
    if (is_real(stmt->ret_type.get_element_type()))
      insert<AdStackAccAdjointStmt>(stmt->stack, load(adjoint(stmt)));
  }

  void visit(AdStackPushStmt *stmt) override {
    accumulate(stmt->v, insert<AdStackLoadTopAdjStmt>(stmt->stack));
    insert<AdStackPopStmt>(stmt->stack);
  }

  void visit(GlobalLoadStmt *stmt) override {
    // issue global store to adjoint

    if (stmt->src->is<ExternalPtrStmt>() ||
        (stmt->src->is<MatrixPtrStmt>() && stmt->src->as<MatrixPtrStmt>()->origin->is<ExternalPtrStmt>())) {
      ExternalPtrStmt *src = nullptr;
      bool is_ptr_offset = false;
      if (stmt->src->is<MatrixPtrStmt>()) {
        src = stmt->src->as<MatrixPtrStmt>()->origin->as<ExternalPtrStmt>();
        is_ptr_offset = true;
      } else {
        src = stmt->src->as<ExternalPtrStmt>();
      }
      auto arg = src->base_ptr->as<ArgLoadStmt>();
      if (arg->ret_type.ptr_removed()->as<StructType>()->elements().size() > TypeFactory::GRAD_PTR_POS_IN_NDARRAY) {
        QD_ASSERT_INFO(!src->is_grad,
                       "Cannot automatically differentiate through a grad "
                       "tensor, if you really want to do that, pass the grad "
                       "tensor into the kernel directly");
        auto adj_ptr =
            insert<ExternalPtrStmt>(src->base_ptr, src->indices, src->ndim, src->element_shape, /*is_grad=*/true);
        adj_ptr->ret_type = src->ret_type;

        if (is_ptr_offset) {
          adj_ptr = insert<MatrixPtrStmt>(adj_ptr, stmt->src->as<MatrixPtrStmt>()->offset);
          adj_ptr->ret_type = stmt->src->ret_type;
          adj_ptr->ret_type.set_is_pointer(true);
        }
        insert<AtomicOpStmt>(AtomicOpType::add, adj_ptr, load(adjoint(stmt)));
      }
      return;
    }

    if (stmt->src->is<GlobalPtrStmt>() ||
        (stmt->src->is<MatrixPtrStmt>() && stmt->src->as<MatrixPtrStmt>()->origin->is<GlobalPtrStmt>())) {
      GlobalPtrStmt *src = nullptr;
      bool is_ptr_offset = false;
      if (stmt->src->is<MatrixPtrStmt>()) {
        is_ptr_offset = true;
        src = stmt->src->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
      } else {
        src = stmt->src->as<GlobalPtrStmt>();
      }

      auto snode = src->snode;
      if (!snode->has_adjoint()) {
        // No adjoint SNode. Do nothing
        return;
      }
      if (gradients_stopped(stmt, snode)) {
        // gradients stopped, do nothing.
        return;
      }
      QD_ASSERT(snode->get_adjoint() != nullptr);
      snode = snode->get_adjoint();
      auto adj_ptr = insert<GlobalPtrStmt>(snode, src->indices);
      adj_ptr->ret_type = src->ret_type;
      if (is_ptr_offset) {
        adj_ptr = insert<MatrixPtrStmt>(adj_ptr, stmt->src->as<MatrixPtrStmt>()->offset);
      }
      insert<AtomicOpStmt>(AtomicOpType::add, adj_ptr, load(adjoint(stmt)));
      return;
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    // erase and replace with global load adjoint

    Stmt *adjoint_ptr{nullptr};
    if (stmt->dest->is<ExternalPtrStmt>() ||
        (stmt->dest->is<MatrixPtrStmt>() && stmt->dest->as<MatrixPtrStmt>()->origin->is<ExternalPtrStmt>())) {
      ExternalPtrStmt *dest = nullptr;
      bool is_ptr_offset = false;
      if (stmt->dest->is<MatrixPtrStmt>()) {
        is_ptr_offset = true;
        dest = stmt->dest->as<MatrixPtrStmt>()->origin->as<ExternalPtrStmt>();
      } else {
        dest = stmt->dest->as<ExternalPtrStmt>();
      }

      auto arg = dest->base_ptr->as<ArgLoadStmt>();
      if (arg->ret_type.ptr_removed()->as<StructType>()->elements().size() <= TypeFactory::GRAD_PTR_POS_IN_NDARRAY) {
        return;
      }
      QD_ASSERT_INFO(!dest->is_grad,
                     "Cannot automatically differentiate through a grad "
                     "tensor, if you really want to do that, pass the grad "
                     "tensor into the kernel directly");
      adjoint_ptr = insert<ExternalPtrStmt>(dest->base_ptr, dest->indices, dest->ndim, dest->element_shape,
                                            /*is_grad=*/true);
      adjoint_ptr->ret_type = dest->ret_type;

      if (is_ptr_offset) {
        adjoint_ptr = insert<MatrixPtrStmt>(adjoint_ptr, stmt->dest->as<MatrixPtrStmt>()->offset);
        adjoint_ptr->ret_type = stmt->dest->ret_type;
        adjoint_ptr->ret_type.set_is_pointer(true);
      }

      accumulate(stmt->val, insert<GlobalLoadStmt>(adjoint_ptr));
    }

    if (stmt->dest->is<GlobalPtrStmt>() ||
        (stmt->dest->is<MatrixPtrStmt>() && stmt->dest->as<MatrixPtrStmt>()->origin->is<GlobalPtrStmt>())) {
      GlobalPtrStmt *dest = nullptr;
      bool is_ptr_offset = false;
      if (stmt->dest->is<MatrixPtrStmt>()) {
        is_ptr_offset = true;
        dest = stmt->dest->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
      } else {
        dest = stmt->dest->as<GlobalPtrStmt>();
      }

      auto snode = dest->snode;
      if (!snode->has_adjoint()) {
        // no gradient (likely integer types)
        return;
      }
      QD_ASSERT(snode->get_adjoint() != nullptr);
      snode = snode->get_adjoint();
      adjoint_ptr = insert<GlobalPtrStmt>(snode, dest->indices);
      adjoint_ptr->ret_type = dest->ret_type;
      if (is_ptr_offset) {
        adjoint_ptr = insert<MatrixPtrStmt>(adjoint_ptr, stmt->dest->as<MatrixPtrStmt>()->offset);
      }
      accumulate(stmt->val, insert<GlobalLoadStmt>(adjoint_ptr));
    }

    // Clear the gradient after accumulation finished.
    auto zero = insert_const_for_grad(adjoint_ptr->ret_type.ptr_removed(), stmt, 0);
    insert<GlobalStoreStmt>(adjoint_ptr, zero);

    stmt->parent->erase(stmt);
  }

  void visit(AtomicOpStmt *stmt) override {
    if (stmt->dest->is<ExternalPtrStmt>() ||
        (stmt->dest->is<MatrixPtrStmt>() && stmt->dest->as<MatrixPtrStmt>()->origin->is<ExternalPtrStmt>())) {
      ExternalPtrStmt *dest = nullptr;
      bool is_ptr_offset = false;
      if (stmt->dest->is<MatrixPtrStmt>()) {
        is_ptr_offset = true;
        dest = stmt->dest->as<MatrixPtrStmt>()->origin->as<ExternalPtrStmt>();
      } else {
        dest = stmt->dest->as<ExternalPtrStmt>();
      }

      auto arg = dest->base_ptr->as<ArgLoadStmt>();
      if (arg->ret_type.ptr_removed()->as<StructType>()->elements().size() > TypeFactory::GRAD_PTR_POS_IN_NDARRAY) {
        QD_ASSERT_INFO(!dest->is_grad,
                       "Cannot automatically differentiate through a grad "
                       "tensor, if you really want to do that, pass the grad "
                       "tensor into the kernel directly");
        auto adjoint_ptr =
            insert<ExternalPtrStmt>(dest->base_ptr, dest->indices, dest->ndim, dest->element_shape, /*is_grad=*/true);
        adjoint_ptr->ret_type = dest->ret_type;

        if (is_ptr_offset) {
          adjoint_ptr = insert<MatrixPtrStmt>(adjoint_ptr, stmt->dest->as<MatrixPtrStmt>()->offset);

          adjoint_ptr->ret_type = stmt->dest->ret_type;
          adjoint_ptr->ret_type.set_is_pointer(true);
        }
        adjoint_ptr->ret_type = dest->ret_type;
        accumulate(stmt->val, insert<GlobalLoadStmt>(adjoint_ptr));
        stmt->parent->erase(stmt);
      }
      return;
    }

    if (stmt->dest->is<GlobalPtrStmt>() ||
        (stmt->dest->is<MatrixPtrStmt>() && stmt->dest->as<MatrixPtrStmt>()->origin->is<GlobalPtrStmt>())) {
      GlobalPtrStmt *dest = nullptr;
      bool is_ptr_offset = false;
      if (stmt->dest->is<MatrixPtrStmt>()) {
        is_ptr_offset = true;
        dest = stmt->dest->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
      } else {
        dest = stmt->dest->as<GlobalPtrStmt>();
      }

      auto snode = dest->snode;
      if (!snode->has_adjoint()) {
        // no gradient (likely integer types)
        return;
      }

      QD_ASSERT(snode->get_adjoint() != nullptr);
      snode = snode->get_adjoint();
      auto adjoint_ptr = insert<GlobalPtrStmt>(snode, dest->indices);
      adjoint_ptr->ret_type = dest->ret_type;
      if (is_ptr_offset) {
        adjoint_ptr = insert<MatrixPtrStmt>(adjoint_ptr, stmt->dest->as<MatrixPtrStmt>()->offset);
      }
      accumulate(stmt->val, insert<GlobalLoadStmt>(adjoint_ptr));
      stmt->parent->erase(stmt);
      return;
    }

    // A65 note: atomic-dest head accumulators are rejected at the FRONTEND
    // (CarriedDestFinder). See visit(LocalStoreStmt) note above.
  }

  void visit(MatrixPtrStmt *stmt) override {
    if (stmt->origin->is<GlobalPtrStmt>() || stmt->origin->is<ExternalPtrStmt>()) {
      /*
        The case of MatrixPtrStmt(GlobalPtrStmt, ...) is already handled in GlobalPtrStmt, GlobalStoreStmt and
        AtomicStmt.

        TODO(zhanlue): Try to separate out the chain rule for MatrixPtrStmt from GlobalPtrStmt, GlobalStoreStmt and
        AtomicStmt and migrate the logics here.
      */
      return;
    }

    DataType prim_dtype = PrimitiveType::f32;
    if (is_real(stmt->ret_type.ptr_removed().get_element_type())) {
      prim_dtype = stmt->ret_type.ptr_removed().get_element_type();
    }

    Stmt *adjoint_value = nullptr;
    if (stmt->offset->is<ConstStmt>()) {
      /*
      [Static index]
      Fwd:
      $0 = alloca <4 x i32>
      $1 = matrix ptr $0, 2 // offset = 2

      Adjoint:
      $3 = matrix init [0, 0, $1_adj, 0] // adjoint_value

      accumulate($0_adj, $3)
      */
      int offset = stmt->offset->as<ConstStmt>()->val.val_int32();

      auto tensor_type = stmt->origin->ret_type.ptr_removed()->as<TensorType>();
      int num_elements = tensor_type->get_num_elements();

      auto zero = insert_const_for_grad(prim_dtype, stmt, 0);
      std::vector<Stmt *> values;
      for (int i = 0; i < num_elements; i++) {
        if (i == offset) {
          values.push_back(load(adjoint(stmt)));
        } else {
          values.push_back(zero);
        }
      }
      auto matrix_init_stmt = insert<MatrixInitStmt>(values);
      matrix_init_stmt->ret_type = tensor_type;

      adjoint_value = matrix_init_stmt;

    } else {
      /*
       [Dynamic index]
       Fwd:
       $0 = alloca <4 x i32>
       $1 = matrix ptr $0, $offset

       Adjoint:
       $3 = matrix init [0.0, 0.0, 0.0, 0.0]
       $4 = matrix init [$1_adj, $1_adj, $1_adj, $1_adj]

       $5 = matrix init [0, 1, 2, 3]
       $6 = matrix init [offset, offset, offset, offset]
       $7 = bin_eq $6, $5
       $8 = select $7, $4, $3 // adjoint_value

       accumulate($0_adj, $7)
      */
      auto tensor_type = stmt->origin->ret_type.ptr_removed()->as<TensorType>();
      auto tensor_shape = tensor_type->get_shape();
      int num_elements = tensor_type->get_num_elements();

      auto zero = insert_const_for_grad(prim_dtype, stmt, 0);
      auto stmt_adj = load(adjoint(stmt));

      std::vector<Stmt *> zero_values(num_elements, zero);
      std::vector<Stmt *> stmt_adj_values(num_elements, stmt_adj);
      std::vector<Stmt *> offset_values(num_elements, stmt->offset);
      std::vector<Stmt *> indices_values(num_elements);
      for (size_t i = 0; i < num_elements; i++) {
        indices_values[i] = insert<ConstStmt>(TypedConstant((int32)i));
      }

      auto zero_matrix_init_stmt = insert<MatrixInitStmt>(zero_values);
      zero_matrix_init_stmt->ret_type = tensor_type;
      auto stmt_adj_matrix_init_stmt = insert<MatrixInitStmt>(stmt_adj_values);
      stmt_adj_matrix_init_stmt->ret_type = tensor_type;

      auto index_tensor_type = TypeFactory::get_instance().get_tensor_type(tensor_shape, PrimitiveType::i32);
      auto indices_matrix_init_stmt = insert<MatrixInitStmt>(indices_values);
      indices_matrix_init_stmt->ret_type = index_tensor_type;

      auto offset_matrix_init_stmt = insert<MatrixInitStmt>(offset_values);
      offset_matrix_init_stmt->ret_type = index_tensor_type;
      auto cmp_tensor_type = TypeFactory::get_instance().get_tensor_type(tensor_shape, PrimitiveType::u1);
      auto bin_eq_stmt = insert<BinaryOpStmt>(BinaryOpType::cmp_eq, offset_matrix_init_stmt, indices_matrix_init_stmt);
      bin_eq_stmt->ret_type = cmp_tensor_type;

      auto select_stmt =
          insert<TernaryOpStmt>(TernaryOpType::select, bin_eq_stmt, stmt_adj_matrix_init_stmt, zero_matrix_init_stmt);
      adjoint_value = select_stmt;
    }

    accumulate(stmt->origin, adjoint_value);
  }

  void visit(MatrixInitStmt *stmt) override {
    auto adjoint_ptr = adjoint(stmt);

    auto tensor_type = stmt->ret_type->as<TensorType>();
    int num_elements = tensor_type->get_num_elements();

    for (size_t i = 0; i < num_elements; i++) {
      auto const_i = insert_const_for_grad(PrimitiveType::i32, stmt, i);

      auto matrix_ptr_stmt_i = insert<MatrixPtrStmt>(adjoint_ptr, const_i);
      matrix_ptr_stmt_i->ret_type = tensor_type->get_element_type();
      matrix_ptr_stmt_i->ret_type.set_is_pointer(true);

      accumulate(stmt->values[i], load(matrix_ptr_stmt_i));
    }
  }
};

}  // namespace

void make_adjoint(Block *ib, int ad_stack_size) {
  MakeAdjoint::run(ib, ad_stack_size);
}

}  // namespace quadrants::lang