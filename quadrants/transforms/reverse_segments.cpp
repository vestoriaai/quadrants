#include "quadrants/ir/ir.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/visitors.h"
#include "quadrants/ir/frontend_ir.h"
#include "quadrants/system/profiler.h"

#include <set>
#include <unordered_map>

namespace quadrants::lang {

namespace irpass {

// TODO: gather Expr as well?
class GatherStmts : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  std::vector<Stmt *> stmts;

  GatherStmts() {
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    stmts.push_back(stmt);
  }
};

// A65: frontend carried-head-accumulator detection. In the FRONTEND IR (pre
// hoisting), a top-level head-segment assignment whose variable is written
// inside the top-level loop (plain assign lhs, or as the destination of an
// in-loop atomic like `acc += x[i]` lowering to atomic_add(dest=acc)) is a
// loop-carried accumulator: reverse-mode does not support it (the adjoint
// needs per-iteration state that offload-level lowering does not provide).
// Read-only head variables and per-iteration inits (assigned INSIDE the loop)
// are safe. Post-hoisting IR cannot distinguish these - hence the frontend
// check.
// A65: frontend carried-head-accumulator detection (first-touch, two passes).
//
// At the FRONTEND (pre hoisting/lowering) source structure survives, so the
// distinction the post-simplify IR cannot make is available here:
//   - head-assigned variable whose FIRST in-loop touch is a READ and which is
//     also written somewhere in the loop = loop-carried accumulator (in this
//     codebase's parallel-__t convention that is the cross-iteration
//     carry-over bug family; the adjoint needs per-iteration state that
//     offload-level lowering does not provide) -> reject;
//   - head-assigned variable written before any read (per-iteration temp with
//     a hoisted-out init) -> safe;
//   - head-assigned variable only read (loop-invariant) -> safe.
// A65: find the dtype of a head-variable's initializing expression by walking
// its RHS for a constant literal (constants carry their dtype at the frontend;
// FrontendAllocaStmt::ret_type is only populated at type_check, which runs
// AFTER reverse_segments). Unknown RHS conservatively = real (reject).
class RhsRealTypeFinder : public ExpressionVisitor {
 public:
  explicit RhsRealTypeFinder()
      : ExpressionVisitor(/*allow_undefined_visitor=*/true, /*invoke_default_visitor=*/true) {
  }

  bool found = false;
  bool is_real_type = false;

  using ExpressionVisitor::visit;

  void visit(ConstExpression *expr) override {
    if (!found) {
      found = true;
      is_real_type = is_real(expr->ret_type);
    }
  }

  void visit(IdExpression *expr) override {
    if (!found) {
      // identifier RHS (copy from another var): dtype unknown here
      found = true;
      is_real_type = true;
    }
  }

  void visit(BinaryOpExpression *expr) override {
    visit(expr->lhs);
    visit(expr->rhs);
  }

  void visit(UnaryOpExpression *expr) override {
    visit(expr->operand);
  }

  void visit(TernaryOpExpression *expr) override {
    visit(expr->op1);
    visit(expr->op2);
    visit(expr->op3);
  }

  void visit(ExprGroup &expr_group) override {
    for (int i = 0; i < (int)expr_group.size(); i++) {
      Expr e = expr_group[i];
      if (e.expr)
        visit(e);
    }
  }
};

class FrontDetector : public ExpressionVisitor {
 public:
  explicit FrontDetector(const std::unordered_set<std::string> &head, const std::unordered_set<std::string> &written)
      : ExpressionVisitor(/*allow_undefined_visitor=*/true, /*invoke_default_visitor=*/true), head_(head), written_(written) {
  }

  bool carried = false;

  using ExpressionVisitor::visit;

  void visit(ExprGroup &expr_group) override {
    for (int i = 0; i < (int)expr_group.size(); i++) {
      Expr e = expr_group[i];
      if (e.expr)
        visit(e);
    }
  }

  void visit(IdExpression *expr) override {
    // read of a head variable
    if (head_.count(expr->id.raw_name()) && written_.count(expr->id.raw_name()) && cleared_.count(expr->id.raw_name()) == 0)
      carried = true;
  }

  void visit(BinaryOpExpression *expr) override {
    visit(expr->lhs);
    visit(expr->rhs);
  }

  void visit(UnaryOpExpression *expr) override {
    visit(expr->operand);
  }

  void visit(TernaryOpExpression *expr) override {
    visit(expr->op1);
    visit(expr->op2);
    visit(expr->op3);
  }

  void visit(AtomicOpExpression *expr) override {
    // atomic dest reads the OLD value then writes: read + write in one
    if (auto id = expr->dest.cast<IdExpression>()) {
      std::string name = id->id.raw_name();
      if (head_.count(name) && written_.count(name) && cleared_.count(name) == 0)
        carried = true;
      cleared_.insert(name);
    }
    visit(expr->val);
    if (expr->expected.expr)
      visit(expr->expected);
  }

  std::unordered_set<std::string> cleared_;  // public: callers seed/collect state
  const std::unordered_set<std::string> &head_;
  const std::unordered_set<std::string> &written_;
};

// Pass 1: which head variables are written anywhere in the loop segment.
void collect_loop_writes(Block *b, const std::unordered_set<std::string> &head,
                         std::unordered_set<std::string> &out) {
  for (auto &s : b->statements) {
    if (auto *fa = s->cast<FrontendAssignStmt>()) {
      if (auto id = fa->lhs.cast<IdExpression>()) {
        if (head.count(id->id.raw_name()))
          out.insert(id->id.raw_name());
      }
      // atomic dest writes may hide in the rhs
      if (fa->rhs.expr) {
        FrontDetector d(head, out);
        d.carried = false;
        d.visit(fa->rhs);  // only used to reach atomics; cleared_ state unused here
        for (const auto &n : d.cleared_)
          out.insert(n);
      }
    } else if (auto *fi = s->cast<FrontendIfStmt>()) {
      if (fi->condition.expr) {
        FrontDetector d(head, out);
        d.visit(fi->condition);
        for (const auto &n : d.cleared_)
          out.insert(n);
      }
      if (fi->true_statements)
        collect_loop_writes(fi->true_statements.get(), head, out);
      if (fi->false_statements)
        collect_loop_writes(fi->false_statements.get(), head, out);
    } else if (auto *fw = s->cast<FrontendWhileStmt>()) {
      if (fw->body)
        collect_loop_writes(fw->body.get(), head, out);
    }
  }
}

// Pass 2: program-order first-touch walk; sets carried=true when a head
// variable that is written in the loop is READ before its first in-loop write.
void walk_first_touch(Block *b, const std::unordered_set<std::string> &head,
                      const std::unordered_set<std::string> &written, bool &carried,
                      std::unordered_set<std::string> &cleared) {
  for (auto &s : b->statements) {
    if (auto *fa = s->cast<FrontendAssignStmt>()) {
      // rhs reads happen before the lhs write
      if (fa->rhs.expr) {
        FrontDetector d(head, written);
        // seed cleared so the detector only flags not-yet-written names
        d.cleared_ = cleared;
        d.visit(fa->rhs);
        if (d.carried)
          carried = true;
        for (const auto &n : d.cleared_)
          cleared.insert(n);
      }
      if (auto id = fa->lhs.cast<IdExpression>()) {
        if (head.count(id->id.raw_name()))
          cleared.insert(id->id.raw_name());
      }
    } else if (auto *fi = s->cast<FrontendIfStmt>()) {
      if (fi->condition.expr) {
        FrontDetector d(head, written);
        d.cleared_ = cleared;
        d.visit(fi->condition);
        if (d.carried)
          carried = true;
        for (const auto &n : d.cleared_)
          cleared.insert(n);
      }
      if (fi->true_statements)
        walk_first_touch(fi->true_statements.get(), head, written, carried, cleared);
      if (fi->false_statements)
        walk_first_touch(fi->false_statements.get(), head, written, carried, cleared);
    } else if (auto *fw = s->cast<FrontendWhileStmt>()) {
      if (fw->body)
        walk_first_touch(fw->body.get(), head, written, carried, cleared);
    }
  }
}

void reverse_segments(IRNode *root) {
  QD_AUTO_PROF;
  auto block = dynamic_cast<Block *>(root);
  std::vector<std::vector<pStmt>> statement_blocks(1);
  bool has_for = false;
  bool has_non_for = false;
  for (auto &&s : block->statements) {
    if (s->is<FrontendForStmt>()) {
      has_for = true;
      statement_blocks.emplace_back();
      statement_blocks.back().push_back(std::move(s));
      statement_blocks.emplace_back();
    } else {
      has_non_for = true;
      statement_blocks.back().push_back(std::move(s));
    }
  }
  block->statements.clear();
  std::reverse(statement_blocks.begin(), statement_blocks.end());
  /*
  for (auto &b : statement_blocks) {
    std::vector<Stmt *> stmts;
    for (auto &s : b) {
      GatherStmts gather;
      s->accept(&gather);
      stmts.insert(stmts.end(), gather.stmts.begin(), gather.stmts.end());
    }
    std::set<Stmt *> stmt_set(stmts.begin(), stmts.end());
    bool valid = true;
    for (auto s : stmts) {
      for (auto op : s->get_operands()) {
        if (stmt_set.find(op) == stmt_set.end()) {
          valid = false;
        }
      }
    }
  }
    */
  if (has_for && has_non_for) {
    // A65: only REAL-typed head variables carry gradient state. Bool/int head
    // latches (control flags, loop cursors) have no adjoint semantics -
    // make_adjoint treats their allocas as no-gradient - so they are exempt
    // from the carried check (e.g. `terminated` latches replacing upstream
    // early returns, index cursors).
    std::unordered_map<std::string, bool> head_names;
    for (auto &sblock : statement_blocks) {
      bool is_for_seg = sblock.size() == 1 && sblock[0]->is<FrontendForStmt>();
      if (!is_for_seg) {
        for (auto &s : sblock) {
          if (auto *fa = s->cast<FrontendAssignStmt>()) {
            auto id = fa->lhs.cast<IdExpression>();
            if (!id)
              continue;
            auto name = id->id.raw_name();
            if (head_names.count(name) != 0 && head_names[name])
              continue;  // already known real
            // dtype from the initializing RHS (constant literals carry their
            // dtype at the frontend); unknown RHS conservatively = real
            RhsRealTypeFinder finder;
            if (fa->rhs.expr)
              finder.visit(fa->rhs);
            head_names[name] = finder.found ? finder.is_real_type : true;
          }
        }
      }
    }
    std::unordered_set<std::string> real_head_names;
    for (const auto &kv : head_names) {
      QD_INFO("[a65-dbg] head name={} real={}", kv.first, kv.second ? 1 : 0);
      if (kv.second)
        real_head_names.insert(kv.first);
    }
    if (!real_head_names.empty()) {
      bool carried = false;
      for (auto &sblock : statement_blocks) {
        if (sblock.size() == 1 && sblock[0]->is<FrontendForStmt>()) {
          if (auto *ff = sblock[0]->cast<FrontendForStmt>()) {
            if (ff->body) {
              std::unordered_set<std::string> written;
              collect_loop_writes(ff->body.get(), real_head_names, written);
              std::unordered_set<std::string> cleared;
              walk_first_touch(ff->body.get(), real_head_names, written, carried, cleared);
            }
          }
        }
      }
      if (carried) {
        QD_ERROR(
            "Invalid program input for autodiff: a top-level head variable is "
            "accumulated/overwritten inside the top-level for-loop "
            "(loop-carried head accumulator). Reverse-mode autodiff does not "
            "support this form: initialize the accumulator inside the loop "
            "(per-iteration semantics), accumulate into a global (needs_grad) "
            "array, or split the kernel.");
        return;
      }
    }
    QD_INFO(
        "[a65] reverse_segments: mixed top-level program accepted (no carried head accumulators)");
  }
  // A65: hoist frontend allocas (variable declarations) to the block head.
  // The segment reordering can otherwise place declarations AFTER statements
  // that reference them (the reversed adjoint of a mixed kernel puts the for
  // first and the head allocas/stores after), breaking lower_ast symbol
  // resolution and downstream alloca-order invariants (uses-before-decl).
  // Alloca hoisting moves only the declaration: stores/reads keep their
  // relative order, so adjoint semantics are unchanged.
  for (auto &sblock : statement_blocks) {
    for (auto &&s : sblock) {
      if (s && s->is<FrontendAllocaStmt>())
        block->statements.push_back(std::move(s));
    }
  }
  for (auto &sblock : statement_blocks) {
    for (auto &&s : sblock) {
      if (s && !s->is<FrontendAllocaStmt>())
        block->statements.push_back(std::move(s));
    }
  }
}

}  // namespace irpass

}  // namespace quadrants::lang
