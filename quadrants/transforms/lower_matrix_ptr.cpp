#include "quadrants/ir/ir.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/visitors.h"
#include "quadrants/system/profiler.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/rhi/arch.h"
#include <map>
#include <numeric>
#include <functional>
#include <tuple>

namespace quadrants::lang {

bool is_aos_matrix_of_global_ptr(MatrixOfGlobalPtrStmt *stmt) {
  auto ret_type = stmt->ret_type;
  auto indices = stmt->indices;
  auto snodes = stmt->snodes;

  bool is_continuous_addr = true;
  SNode *parent_snode = snodes[0]->parent;
  for (auto snode : snodes) {
    QD_ASSERT(snode->type == SNodeType::place);
    if (snode->parent != parent_snode || parent_snode->type != SNodeType::dense) {
      is_continuous_addr = false;
      break;
    }
  }

  return is_continuous_addr;
}

class GatherValidAOSGlobalPtrStmt : public BasicStmtVisitor {
 public:
  DelayedIRModifier modifier_;
  std::unordered_set<Stmt *> invalid_aos_global_ptr_stmts_;

  explicit GatherValidAOSGlobalPtrStmt(IRNode *node) {
    node->accept(this);

    modifier_.modify_ir();
  }

  void visit(MatrixPtrStmt *stmt) override {
    if (stmt->origin->is<MatrixOfGlobalPtrStmt>()) {
      auto origin = stmt->origin->as<MatrixOfGlobalPtrStmt>();

      if (is_quant(origin->snodes[0]->dt)) {
        invalid_aos_global_ptr_stmts_.insert(stmt->origin);
      }

      if (origin->snodes[0]->dt->is<PointerType>() && origin->snodes[0]->dt->as<PointerType>()->is_bit_pointer()) {
        invalid_aos_global_ptr_stmts_.insert(stmt->origin);
      }

      if (!stmt->offset->is<ConstStmt>()) {
        invalid_aos_global_ptr_stmts_.insert(stmt->origin);
      }
    }
  }

 private:
  using BasicStmtVisitor::visit;
};

class LowerAOSGlobalPtrStmt : public BasicStmtVisitor {
 public:
  ImmediateIRModifier immediate_modifier_;
  DelayedIRModifier delayed_modifier_;
  std::unordered_set<Stmt *> invalid_aos_global_ptr_stmts_;

  explicit LowerAOSGlobalPtrStmt(IRNode *node, std::unordered_set<Stmt *> &invalid_aos_global_ptr_stmts)
      : immediate_modifier_(node), invalid_aos_global_ptr_stmts_(invalid_aos_global_ptr_stmts) {
    node->accept(this);

    delayed_modifier_.modify_ir();
  }

  void visit(MatrixOfGlobalPtrStmt *stmt) override {
    bool is_aos = is_aos_matrix_of_global_ptr(stmt);
    auto ret_type = stmt->ret_type;
    auto indices = stmt->indices;
    auto snodes = stmt->snodes;

    if (is_aos && invalid_aos_global_ptr_stmts_.find(stmt) == invalid_aos_global_ptr_stmts_.end()) {
      auto new_stmt = std::make_unique<GlobalPtrStmt>(snodes[0], indices);
      new_stmt->ret_type = ret_type;
      new_stmt->ret_type.set_is_pointer(true);

      stmt->replace_usages_with(new_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(new_stmt));
      delayed_modifier_.erase(stmt);
    }
  }

 private:
  using BasicStmtVisitor::visit;
};

class ScalarizeMatrixPtr : public BasicStmtVisitor {
 public:
  ImmediateIRModifier immediate_modifier_;
  DelayedIRModifier delayed_modifier_;

  explicit ScalarizeMatrixPtr(IRNode *node) : immediate_modifier_(node) {
    node->accept(this);

    delayed_modifier_.modify_ir();
  }

  /*
    [MatrixOfGlobalPtrStmt]
    Before:
      StoreStmt(TensorType<4 x i32>* MatrixOfGlobalPtrStmt, TensorType<4 x i32>
    val)

    After:
      AllocaStmt alloca = AllocaStmt(TensorType<4 x i32>)
      StoreStmt(alloca, val)

      matrix_ptr0 = MatrixPtrStmt(i32* alloca, 0)
      matrix_ptr1 = MatrixPtrStmt(i32* alloca, 1)
      matrix_ptr2 = MatrixPtrStmt(i32* alloca, 2)
      matrix_ptr3 = MatrixPtrStmt(i32* alloca, 3)

      val0 = LoadStmt(matrix_ptr0)
      val1 = LoadStmt(matrix_ptr1)
      val2 = LoadStmt(matrix_ptr2)
      val3 = LoadStmt(matrix_ptr3)

      GlobalPtrStmt* ptr0 = MatrixOfGlobalPtrStmt->snodes[0]
      GlobalPtrStmt* ptr1 = MatrixOfGlobalPtrStmt->snodes[1]
      GlobalPtrStmt* ptr2 = MatrixOfGlobalPtrStmt->snodes[2]
      GlobalPtrStmt* ptr3 = MatrixOfGlobalPtrStmt->snodes[3]
      StoreStmt(i32* ptr0, i32 val0)
      StoreStmt(i32* ptr1, i32 val1)
      StoreStmt(i32* ptr2, i32 val2)
      StoreStmt(i32* ptr3, i32 val3)

    [MatrixOfMatrixPtrStmt]
    Before:
      StoreStmt(TensorType<4 x i32>* MatrixOfMatrixPtrStmt, TensorType<4 x i32>
    val)

    After:
      stmt0 = MatrixOfMatrixPtrStmt->stmts[0]
      stmt1 = MatrixOfMatrixPtrStmt->stmts[1]
      stmt2 = MatrixOfMatrixPtrStmt->stmts[2]
      stmt3 = MatrixOfMatrixPtrStmt->stmts[3]

      AllocaStmt alloca = AllocaStmt(TensorType<4 x i32>)
      LocalStoreStmt(alloca, val)

      matrix_ptr0 = MatrixPtrStmt(i32* alloca, 0)
      matrix_ptr1 = MatrixPtrStmt(i32* alloca, 1)
      matrix_ptr2 = MatrixPtrStmt(i32* alloca, 2)
      matrix_ptr3 = MatrixPtrStmt(i32* alloca, 3)

      val0 = LoadStmt(matrix_ptr0)
      val1 = LoadStmt(matrix_ptr1)
      val2 = LoadStmt(matrix_ptr2)
      val3 = LoadStmt(matrix_ptr3)

      StoreStmt(i32* stmt0, i32 val0)
      StoreStmt(i32* stmt1, i32 val1)
      StoreStmt(i32* stmt2, i32 val2)
      StoreStmt(i32* stmt3, i32 val3)
  */
  template <typename T>
  void scalarize_store_stmt(T *stmt) {
    if (stmt->dest->template is<MatrixOfGlobalPtrStmt>()) {
      auto matrix_of_global_ptr_stmt = stmt->dest->template as<MatrixOfGlobalPtrStmt>();
      auto val = stmt->val;
      auto val_tensor_type = val->ret_type->template as<TensorType>();
      int num_elements = val_tensor_type->get_num_elements();
      auto primitive_type = val_tensor_type->get_element_type();

      auto alloca_stmt = std::make_unique<AllocaStmt>(val->ret_type);
      Stmt *alloca_stmt_ptr = alloca_stmt.get();

      auto store_stmt = std::make_unique<LocalStoreStmt>(alloca_stmt_ptr, val);

      delayed_modifier_.insert_before(stmt, std::move(alloca_stmt));
      delayed_modifier_.insert_before(stmt, std::move(store_stmt));
      for (int i = 0; i < num_elements; i++) {
        auto const_stmt = std::make_unique<ConstStmt>(TypedConstant(i));

        auto matrix_ptr_stmt = std::make_unique<MatrixPtrStmt>(alloca_stmt_ptr, const_stmt.get());
        matrix_ptr_stmt->ret_type = primitive_type;
        matrix_ptr_stmt->ret_type.set_is_pointer(true);

        auto load_stmt = std::make_unique<LocalLoadStmt>(matrix_ptr_stmt.get());
        load_stmt->ret_type = primitive_type;

        auto global_ptr_stmt =
            std::make_unique<GlobalPtrStmt>(matrix_of_global_ptr_stmt->snodes[i], matrix_of_global_ptr_stmt->indices);
        global_ptr_stmt->ret_type.set_is_pointer(true);

        auto store_stmt = std::make_unique<T>(global_ptr_stmt.get(), load_stmt.get());

        delayed_modifier_.insert_before(stmt, std::move(const_stmt));
        delayed_modifier_.insert_before(stmt, std::move(matrix_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(load_stmt));
        delayed_modifier_.insert_before(stmt, std::move(global_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(store_stmt));
      }

      delayed_modifier_.erase(stmt);

    } else if (stmt->dest->template is<MatrixOfMatrixPtrStmt>()) {
      auto matrix_of_matrix_ptr_stmt = stmt->dest->template as<MatrixOfMatrixPtrStmt>();
      auto val = stmt->val;
      auto val_tensor_type = val->ret_type->template as<TensorType>();
      int num_elements = val_tensor_type->get_num_elements();
      auto primitive_type = val_tensor_type->get_element_type();

      auto alloca_stmt = std::make_unique<AllocaStmt>(val->ret_type);
      Stmt *alloca_stmt_ptr = alloca_stmt.get();

      auto store_stmt = std::make_unique<LocalStoreStmt>(alloca_stmt_ptr, val);

      delayed_modifier_.insert_before(stmt, std::move(alloca_stmt));
      delayed_modifier_.insert_before(stmt, std::move(store_stmt));
      for (int i = 0; i < num_elements; i++) {
        auto const_stmt = std::make_unique<ConstStmt>(TypedConstant(i));

        auto matrix_ptr_stmt = std::make_unique<MatrixPtrStmt>(alloca_stmt_ptr, const_stmt.get());
        matrix_ptr_stmt->ret_type = primitive_type;
        matrix_ptr_stmt->ret_type.set_is_pointer(true);

        auto load_stmt = std::make_unique<LocalLoadStmt>(matrix_ptr_stmt.get());
        load_stmt->ret_type = primitive_type;

        auto store_stmt = std::make_unique<T>(matrix_of_matrix_ptr_stmt->stmts[i], load_stmt.get());

        delayed_modifier_.insert_before(stmt, std::move(const_stmt));
        delayed_modifier_.insert_before(stmt, std::move(matrix_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(load_stmt));
        delayed_modifier_.insert_before(stmt, std::move(store_stmt));
      }

      delayed_modifier_.erase(stmt);
    }
  }

  /*
    [MatrixOfGlobalPtrStmt]
    Before:
      TensorType<4 x i32> val = LoadStmt(TensorType<4 x i32>*
    MatrixOfGlobalPtrStmt)

    After:
      GlobalPtrStmt* ptr0 = MatrixOfGlobalPtrStmt->snodes[0]
      GlobalPtrStmt* ptr1 = MatrixOfGlobalPtrStmt->snodes[1]
      GlobalPtrStmt* ptr2 = MatrixOfGlobalPtrStmt->snodes[2]
      GlobalPtrStmt* ptr3 = MatrixOfGlobalPtrStmt->snodes[3]

      i32 val0 = LoadStmt(ptr0)
      i32 val1 = LoadStmt(ptr1)
      i32 val2 = LoadStmt(ptr2)
      i32 val3 = LoadStmt(ptr3)

      tmp = MatrixInitStmt(val0, val1, val2, val3)
      stmt->replace_all_usages_with(tmp)

    [MatrixOfMatrixPtrStmt]
    Before:
      TensorType<4 x i32> val = LoadStmt(TensorType<4 x i32>*
    MatrixOfMatrixPtrStmt)

    After:
      i32* ptr0 = MatrixOfMatrixPtr->stmts[0] // usually it's a MatrixPtrStmt
      i32* ptr1 = MatrixOfMatrixPtr->stmts[1] // usually it's a MatrixPtrStmt
      i32* ptr2 = MatrixOfMatrixPtr->stmts[2] // usually it's a MatrixPtrStmt
      i32* ptr3 = MatrixOfMatrixPtr->stmts[3] // usually it's a MatrixPtrStmt

      i32 val0 = LoadStmt(ptr0)
      i32 val1 = LoadStmt(ptr1)
      i32 val2 = LoadStmt(ptr2)
      i32 val3 = LoadStmt(ptr3)

      tmp = MatrixInitStmt(val0, val1, val2, val3)
      stmt->replace_all_usages_with(tmp)
  */
  template <typename T>
  void scalarize_load_stmt(T *stmt) {
    if (stmt->src->template is<MatrixOfGlobalPtrStmt>()) {
      auto matrix_of_global_ptr_stmt = stmt->src->template as<MatrixOfGlobalPtrStmt>();
      auto dest_tensor_type = stmt->ret_type->template as<TensorType>();

      std::vector<Stmt *> matrix_init_values;
      int num_elements = dest_tensor_type->get_num_elements();

      auto primitive_type = dest_tensor_type->get_element_type();
      for (size_t i = 0; i < num_elements; i++) {
        auto global_ptr_stmt =
            std::make_unique<GlobalPtrStmt>(matrix_of_global_ptr_stmt->snodes[i], matrix_of_global_ptr_stmt->indices);
        global_ptr_stmt->ret_type.set_is_pointer(true);

        auto load_stmt = std::make_unique<T>(global_ptr_stmt.get());
        load_stmt->ret_type = primitive_type;

        matrix_init_values.push_back(load_stmt.get());

        delayed_modifier_.insert_before(stmt, std::move(global_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(load_stmt));
      }

      auto matrix_init_stmt = std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = stmt->ret_type;

      stmt->replace_usages_with(matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));
      delayed_modifier_.erase(stmt);

    } else if (stmt->src->template is<MatrixOfMatrixPtrStmt>()) {
      auto matrix_of_matrix_ptr_stmt = stmt->src->template as<MatrixOfMatrixPtrStmt>();
      auto dest_tensor_type = stmt->ret_type->template as<TensorType>();

      std::vector<Stmt *> matrix_init_values;
      int num_elements = dest_tensor_type->get_num_elements();

      auto primitive_type = dest_tensor_type->get_element_type();
      for (size_t i = 0; i < num_elements; i++) {
        Stmt *matrix_ptr_stmt = matrix_of_matrix_ptr_stmt->stmts[i];

        auto load_stmt = std::make_unique<T>(matrix_ptr_stmt);
        load_stmt->ret_type = primitive_type;

        matrix_init_values.push_back(load_stmt.get());

        delayed_modifier_.insert_before(stmt, std::move(load_stmt));
      }

      auto matrix_init_stmt = std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = stmt->ret_type;

      stmt->replace_usages_with(matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));
      delayed_modifier_.erase(stmt);
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    scalarize_store_stmt<GlobalStoreStmt>(stmt);
  }

  void visit(LocalStoreStmt *stmt) override {
    scalarize_store_stmt<LocalStoreStmt>(stmt);
  }

  void visit(GlobalLoadStmt *stmt) override {
    scalarize_load_stmt<GlobalLoadStmt>(stmt);
  }

  void visit(LocalLoadStmt *stmt) override {
    scalarize_load_stmt<LocalLoadStmt>(stmt);
  }

  /*
    [MatrixOfGlobalPtrStmt]
    Before:
      StoreStmt(TensorType<4 x i32>* MatrixOfGlobalPtrStmt, TensorType<4 x i32>
    val)

    After:
      AllocaStmt alloca = AllocaStmt(TensorType<4 x i32>)
      StoreStmt(alloca, val)

      matrix_ptr0 = MatrixPtrStmt(i32* alloca, 0)
      matrix_ptr1 = MatrixPtrStmt(i32* alloca, 1)
      matrix_ptr2 = MatrixPtrStmt(i32* alloca, 2)
      matrix_ptr3 = MatrixPtrStmt(i32* alloca, 3)

      val0 = LoadStmt(matrix_ptr0)
      val1 = LoadStmt(matrix_ptr1)
      val2 = LoadStmt(matrix_ptr2)
      val3 = LoadStmt(matrix_ptr3)

      GlobalPtrStmt* ptr0 = MatrixOfGlobalPtrStmt->snodes[0]
      GlobalPtrStmt* ptr1 = MatrixOfGlobalPtrStmt->snodes[1]
      GlobalPtrStmt* ptr2 = MatrixOfGlobalPtrStmt->snodes[2]
      GlobalPtrStmt* ptr3 = MatrixOfGlobalPtrStmt->snodes[3]
      i32 val_o0 = AtomicStmt(i32* ptr0, i32 val0)
      i32 val_o1 = AtomicStmt(i32* ptr1, i32 val1)
      i32 val_o2 = AtomicStmt(i32* ptr2, i32 val2)
      i32 val_o3 = AtomicStmt(i32* ptr3, i32 val3)

      tmp = MatrixInitStmt(val_o0, val_o1, val_o2, val_o3)
      stmt->replace_all_usages_with(tmp)
  */
  void visit(AtomicOpStmt *stmt) override {
    if (stmt->dest->is<MatrixOfGlobalPtrStmt>()) {
      auto matrix_of_global_ptr_stmt = stmt->dest->as<MatrixOfGlobalPtrStmt>();
      auto val = stmt->val;
      auto val_tensor_type = val->ret_type->as<TensorType>();
      int num_elements = val_tensor_type->get_num_elements();
      auto primitive_type = val_tensor_type->get_element_type();

      std::vector<Stmt *> matrix_init_values;

      auto alloca_stmt = std::make_unique<AllocaStmt>(val->ret_type);
      Stmt *alloca_stmt_ptr = alloca_stmt.get();

      auto store_stmt = std::make_unique<LocalStoreStmt>(alloca_stmt_ptr, val);

      delayed_modifier_.insert_before(stmt, std::move(alloca_stmt));
      delayed_modifier_.insert_before(stmt, std::move(store_stmt));
      for (int i = 0; i < num_elements; i++) {
        auto const_stmt = std::make_unique<ConstStmt>(TypedConstant(i));

        auto matrix_ptr_stmt = std::make_unique<MatrixPtrStmt>(alloca_stmt_ptr, const_stmt.get());
        matrix_ptr_stmt->ret_type = primitive_type;
        matrix_ptr_stmt->ret_type.set_is_pointer(true);

        auto load_stmt = std::make_unique<LocalLoadStmt>(matrix_ptr_stmt.get());
        load_stmt->ret_type = primitive_type;

        auto global_ptr_stmt =
            std::make_unique<GlobalPtrStmt>(matrix_of_global_ptr_stmt->snodes[i], matrix_of_global_ptr_stmt->indices);
        global_ptr_stmt->ret_type.set_is_pointer(true);

        auto atomic_stmt = std::make_unique<AtomicOpStmt>(stmt->op_type, global_ptr_stmt.get(), load_stmt.get());
        atomic_stmt->ret_type = primitive_type;

        matrix_init_values.push_back(atomic_stmt.get());

        delayed_modifier_.insert_before(stmt, std::move(const_stmt));
        delayed_modifier_.insert_before(stmt, std::move(matrix_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(load_stmt));
        delayed_modifier_.insert_before(stmt, std::move(global_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(atomic_stmt));
      }

      auto matrix_init_stmt = std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = stmt->ret_type;

      stmt->replace_usages_with(matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));
      delayed_modifier_.erase(stmt);

    } else if (stmt->dest->is<MatrixOfMatrixPtrStmt>()) {
      auto matrix_of_matrix_ptr_stmt = stmt->dest->as<MatrixOfMatrixPtrStmt>();
      auto val = stmt->val;
      auto val_tensor_type = val->ret_type->as<TensorType>();
      int num_elements = val_tensor_type->get_num_elements();
      auto primitive_type = val_tensor_type->get_element_type();

      std::vector<Stmt *> matrix_init_values;

      auto alloca_stmt = std::make_unique<AllocaStmt>(val->ret_type);
      Stmt *alloca_stmt_ptr = alloca_stmt.get();

      auto store_stmt = std::make_unique<LocalStoreStmt>(alloca_stmt_ptr, val);

      delayed_modifier_.insert_before(stmt, std::move(alloca_stmt));
      delayed_modifier_.insert_before(stmt, std::move(store_stmt));
      for (int i = 0; i < num_elements; i++) {
        auto const_stmt = std::make_unique<ConstStmt>(TypedConstant(i));

        auto matrix_ptr_stmt = std::make_unique<MatrixPtrStmt>(alloca_stmt_ptr, const_stmt.get());
        matrix_ptr_stmt->ret_type = primitive_type;
        matrix_ptr_stmt->ret_type.set_is_pointer(true);

        auto load_stmt = std::make_unique<LocalLoadStmt>(matrix_ptr_stmt.get());
        load_stmt->ret_type = primitive_type;

        auto atomic_stmt =
            std::make_unique<AtomicOpStmt>(stmt->op_type, matrix_of_matrix_ptr_stmt->stmts[i], load_stmt.get());
        atomic_stmt->ret_type = primitive_type;

        matrix_init_values.push_back(atomic_stmt.get());

        delayed_modifier_.insert_before(stmt, std::move(const_stmt));
        delayed_modifier_.insert_before(stmt, std::move(matrix_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(load_stmt));
        delayed_modifier_.insert_before(stmt, std::move(atomic_stmt));
      }

      auto matrix_init_stmt = std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = stmt->ret_type;

      stmt->replace_usages_with(matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));
      delayed_modifier_.erase(stmt);
    }
  }

 private:
  using BasicStmtVisitor::visit;
};

class LowerMatrixPtr : public BasicStmtVisitor {
 private:
  using BasicStmtVisitor::visit;
  DelayedIRModifier modifier_;

 public:
  void visit(MatrixPtrStmt *stmt) override {
    if (stmt->origin->is<MatrixOfGlobalPtrStmt>()) {
      auto origin = stmt->origin->as<MatrixOfGlobalPtrStmt>();
      if (stmt->offset->is<ConstStmt>()) {
        auto offset = stmt->offset->as<ConstStmt>();
        auto lowered = std::make_unique<GlobalPtrStmt>(origin->snodes[offset->val.val_int()], origin->indices);
        lowered->ret_type.set_is_pointer(true);
        stmt->replace_usages_with(lowered.get());
        modifier_.insert_before(stmt, std::move(lowered));
        modifier_.erase(stmt);
      } else {
        QD_ASSERT_INFO(origin->dynamic_indexable, "Element of the MatrixField is not dynamic indexable.\n{}",
                       stmt->get_tb());
        auto stride = std::make_unique<ConstStmt>(TypedConstant(origin->dynamic_index_stride));
        auto offset = std::make_unique<BinaryOpStmt>(BinaryOpType::mul, stmt->offset, stride.get());
        offset->ret_type = stmt->offset->ret_type;

        auto ptr_base = std::make_unique<GlobalPtrStmt>(origin->snodes[0], origin->indices);
        ptr_base->ret_type.set_is_pointer(true);
        auto lowered = std::make_unique<MatrixPtrStmt>(ptr_base.get(), offset.get());
        stmt->replace_usages_with(lowered.get());
        modifier_.insert_before(stmt, std::move(stride));
        modifier_.insert_before(stmt, std::move(offset));
        modifier_.insert_before(stmt, std::move(ptr_base));
        modifier_.insert_before(stmt, std::move(lowered));
        modifier_.erase(stmt);
      }
      return;
    }
    if (stmt->origin->is<MatrixOfMatrixPtrStmt>()) {
      auto origin = stmt->origin->as<MatrixOfMatrixPtrStmt>();
      QD_ASSERT(stmt->offset->is<ConstStmt>());
      auto offset = stmt->offset->as<ConstStmt>();
      stmt->replace_usages_with(origin->stmts[offset->val.val_int()]);
      modifier_.erase(stmt);
      return;
    }
  }

  static void run(IRNode *node) {
    LowerMatrixPtr pass;
    node->accept(&pass);
    pass.modifier_.modify_ir();
  }
};

class RemoveMatrixOfPtr : public BasicStmtVisitor {
 private:
  using BasicStmtVisitor::visit;
  DelayedIRModifier modifier_;

 public:
  void visit(MatrixOfGlobalPtrStmt *stmt) override {
    modifier_.erase(stmt);
  }

  void visit(MatrixOfMatrixPtrStmt *stmt) override {
    modifier_.erase(stmt);
  }

  static void run(IRNode *node) {
    RemoveMatrixOfPtr pass;
    node->accept(&pass);
    pass.modifier_.modify_ir();
  }
};

namespace irpass {

// §12.41 (mujoqo, QD_MATRIX_VEC_MEM=1): contiguous MatrixInit gathers of
// ExternalPtrStmt lanes (a vec4/vec3 read of a flattened row) lower per-lane
// today. Replace such a gather with ONE vector-typed ExternalPtrStmt (clone of
// lane 0, pointee = TensorType(lanes)) + a whole-value GlobalLoadStmt — the
// downstream whole-vector BinaryOp then consumes it with native <N x float>
// ops. Contiguity: lanes share base/dt/outer indices and their last indices
// are base + 0,+1,...,n-1 (structurally compared).
class VectorizeContiguousExtInit : public BasicStmtVisitor {
 public:
  DelayedIRModifier modifier_;
  bool enabled_;
  int vectorized_ = 0;

  explicit VectorizeContiguousExtInit(IRNode *node, bool enabled) : enabled_(enabled) {
    if (enabled)
      node->accept(this);
    flush_row_groups();
    modifier_.modify_ir();
  }
  // §12.46: contiguous-row scalar-store merging (mujoqo o-row stores)
  struct RowStoreGroup {
    std::map<int, GlobalStoreStmt *> by_col;
    std::map<int, Stmt *> val_by_col;
    std::map<int, int> val_lane_by_col;
    Stmt *val_alloca = nullptr;
    int col0 = INT_MAX;
  };
  // §12.53: value fingerprint — the 4 stores of a row are lowered from
  // independently-instantiated index expressions (per-col static unroll), so
  // pointer identity does NOT group them; full_simplify's CSE runs later.
  // Const/BinaryOp folds recursively; other leaves use instance identity
  // (loop index, alloca, argload are shared across the 4 stores).
  static std::string stmt_fingerprint(Stmt *s) {
    if (s == nullptr)
      return "x";
    if (auto *cs = s->template cast<ConstStmt>())
      return "C" + std::to_string(cs->val.val_i32);
    if (auto *b = s->template cast<BinaryOpStmt>())
      return "B" + std::to_string((int)b->op_type) + "(" +
             stmt_fingerprint(b->lhs) + "," + stmt_fingerprint(b->rhs) + ")";
    if (auto *l = s->template cast<LocalLoadStmt>())
      return "LL(" + stmt_fingerprint(l->src) + ")";
    if (auto *m = s->template cast<MatrixPtrStmt>())
      return "MP(" + stmt_fingerprint(m->origin) + "+" +
             stmt_fingerprint(m->offset) + ")";
    return "S" + s->raw_name();
  }

  // (arg_slot, mid_fp, last_base_fp)
  std::map<std::tuple<int, std::string, std::string>, RowStoreGroup>
      row_store_groups_;

  void flush_row_groups() {
    for (auto &kv : row_store_groups_) {
      auto &g = kv.second;
      if (FILE *dg = fopen("/tmp/qd_veclang.log", "a")) {
        fprintf(dg, "flush: key=(%d,%.24s|%.24s) by_col=%d val_alloca=%s\n",
                std::get<0>(kv.first), std::get<1>(kv.first).c_str(),
                std::get<2>(kv.first).c_str(), (int)g.by_col.size(),
                g.val_alloca ? "set" : "null");
        fclose(dg);
      }
      if ((int)g.by_col.size() != 4)
        continue;
      bool complete = true;
      for (int c = 0; c < 4; c++) {
        if (g.by_col.count(c) == 0 || g.val_lane_by_col.count(c) == 0 ||
            g.val_lane_by_col.at(c) != c)
          complete = false;
      }
      if (!complete)
        continue;
      auto dt = g.by_col.at(0)->val->ret_type;
      auto vec_tensor = TypeFactory::get_instance().get_tensor_type(std::vector<int>{4}, dt);
      auto cloned = std::unique_ptr<Stmt>(g.by_col.at(0)->dest->clone());
      cloned->ret_type = TypeFactory::get_instance().get_pointer_type(vec_tensor);
      cloned->ret_type.set_is_pointer(true);

      auto zero = std::make_unique<ConstStmt>(TypedConstant(0));
      auto whole_ptr = std::make_unique<MatrixPtrStmt>(g.val_alloca, zero.get());
      whole_ptr->ret_type = TypeFactory::get_instance().get_pointer_type(vec_tensor);
      whole_ptr->ret_type.set_is_pointer(true);
      auto whole_load = std::make_unique<LocalLoadStmt>(whole_ptr.get());
      whole_load->ret_type = vec_tensor;

      auto merged = std::make_unique<GlobalStoreStmt>(cloned.get(), whole_load.get());

      modifier_.insert_before(g.by_col.at(0), std::move(cloned));
      modifier_.insert_before(g.by_col.at(0), std::move(zero));
      modifier_.insert_before(g.by_col.at(0), std::move(whole_ptr));
      modifier_.insert_before(g.by_col.at(0), std::move(whole_load));
      modifier_.insert_before(g.by_col.at(0), std::move(merged));
      for (int c = 0; c < 4; c++)
        modifier_.erase(g.by_col.at(c));
    }
    row_store_groups_.clear();
  }

  void visit(GlobalStoreStmt *stmt) override {
    if (!enabled_)
      return;
    auto *dest = stmt->dest->template cast<ExternalPtrStmt>();
    if (dest == nullptr ||
        (dest->indices.size() != 2 && dest->indices.size() != 3))
      return;
    auto *vl = stmt->val->template cast<LocalLoadStmt>();
    if (vl == nullptr || !vl->src->is<MatrixPtrStmt>())
      return;
    auto *mp = vl->src->as<MatrixPtrStmt>();
    if (!mp->origin->is<AllocaStmt>() || !mp->offset->is<ConstStmt>())
      return;
    int lane = mp->offset->as<ConstStmt>()->val.val_i32;
    // §12.44 fix: group by ARG SLOT (each store has its own argaddr stmt
    // instance; the slot id identifies the array) + the row accumulator
    int arg_slot = -1;
    if (auto *al = dest->base_ptr->template cast<ArgLoadStmt>())
      if (!al->arg_id.empty())
        arg_slot = al->arg_id[0];
    auto key = std::make_tuple(arg_slot, std::string(""), std::string(""));
    int c = -1;
    if (dest->indices.size() == 2) {
      // probe form: [Const row, Const col]
      auto *ri = dest->indices[0]->template cast<ConstStmt>();
      auto *ci = dest->indices[1]->template cast<ConstStmt>();
      if (ri == nullptr || ci == nullptr)
        return;
      std::get<1>(key) = "r" + std::to_string(ri->val.val_i32);
      c = ci->val.val_i32;
    } else {
      // §12.52-supplement MAT4TILE main-line form: [runtime, mid (shared
      // stmt per row), last = shared base + {0,1,2,3}]
      Stmt *mid = dest->indices[1];
      Stmt *last = dest->indices[2];
      Stmt *last_base = nullptr;
      int delta = 0;
      if (auto *b = last->template cast<BinaryOpStmt>()) {
        if (b->op_type != BinaryOpType::add)
          return;
        if (auto *cs = b->lhs->template cast<ConstStmt>()) {
          last_base = b->rhs;
          delta = cs->val.val_i32;
        } else if (auto *cs = b->rhs->template cast<ConstStmt>()) {
          last_base = b->lhs;
          delta = cs->val.val_i32;
        } else {
          return;
        }
      } else {
        last_base = last;
      }
      if (delta < 0 || delta > 3)
        return;
      std::get<1>(key) = "m" + stmt_fingerprint(mid);
      std::get<2>(key) = "l" + stmt_fingerprint(last_base);
      c = delta;
    }
    auto &g = row_store_groups_[key];
    if (g.by_col.empty()) {
      g.val_alloca = mp->origin;
      g.col0 = c;
    }
    g.by_col[c] = stmt;
    g.val_by_col[c] = stmt->val;
    g.val_lane_by_col[c] = lane;
  }

  using BasicStmtVisitor::visit;
  void visit(MatrixInitStmt *stmt) override {
    if (!enabled_)
      return;
    const int n = (int)stmt->values.size();
    if (FILE *dg = fopen("/tmp/qd_veclang.log", "a")) {
      fprintf(dg, "MI n=%d lanes=[", n);
      for (int li = 0; li < n; li++) {
        auto *gl = stmt->values[li]->template cast<GlobalLoadStmt>();
        auto *ll = stmt->values[li]->template cast<LocalLoadStmt>();
        auto *ep = stmt->values[li]->template cast<ExternalPtrStmt>();
        fprintf(dg, "%s%s%s", gl ? "GL" : "", ll ? "LL" : "", ep ? "EP" : "");
        if (!gl && !ll && !ep)
          fprintf(dg, "?%s", stmt->values[li]->ret_type.to_string().c_str());
        fprintf(dg, "%s", li + 1 < n ? "," : "");
      }
      fprintf(dg, "] dt=%s\n", stmt->ret_type.to_string().c_str());
      fclose(dg);
    }
    if (n < 2 || n > 16)
      return;
    // lanes must be GlobalLoadStmts from the same contiguous ExternalPtrStmt
    std::vector<GlobalLoadStmt *> loads;
    std::vector<ExternalPtrStmt *> ptrs;
    // §12.53: the per-col gathers carry their own ArgLoadStmt instance per
    // lane (same arg slot) — compare by slot id, falling back to pointer
    // identity for non-arg bases. Previously this was a silent return with
    // no bail log (GL4 zero-hit on the jv gathers).
    auto base_slot = [](Stmt *b) -> int {
      if (auto *al = b->template cast<ArgLoadStmt>())
        if (!al->arg_id.empty())
          return al->arg_id[0];
      return -1;
    };
    auto bail = [&](const char *w) {
      if (FILE *dg = fopen("/tmp/qd_veclang.log", "a")) {
        fprintf(dg, "GL4 miss why=%s\n", w);
        fclose(dg);
      }
    };

    const int slot0 = base_slot(ptrs[0]->base_ptr);
    for (auto *st : stmt->values) {
      auto *ld = st->template cast<GlobalLoadStmt>();
      if (ld == nullptr)
        return;
      auto *src = ld->src->template cast<ExternalPtrStmt>();
      if (src == nullptr || !src->ret_type->is<PointerType>() || src->indices.empty())
        return;
      if (src->indices.size() != ptrs[0]->indices.size())
        return;
      if (slot0 < 0) {
        if (src->base_ptr != ptrs[0]->base_ptr) {
          bail("baseptr");
          return;
        }
      } else if (base_slot(src->base_ptr) != slot0) {
        bail("baseslot");
        return;
      }
      loads.push_back(ld);
      ptrs.push_back(src);
    }
    const size_t last = ptrs[0]->indices.size() - 1;
    for (int i = 1; i < n; i++) {
      for (size_t d = 0; d < last; d++)
        if (!irpass::analysis::same_value(ptrs[0]->indices[d], ptrs[i]->indices[d])) {
          bail("outer");
          return;
        }
    }
    auto split_const_offset = [&](Stmt *last_index, int *c) -> Stmt * {
      if (auto *b = last_index->cast<BinaryOpStmt>()) {
        if (b->op_type != BinaryOpType::add || !b->rhs->is<ConstStmt>())
          return nullptr;
        *c = b->rhs->as<ConstStmt>()->val.val_i32;
        return b->lhs;
      }
      if (auto *cst = last_index->cast<ConstStmt>()) {
        *c = cst->val.val_i32;
        return nullptr;
      }
      return nullptr;
    };
    int c0 = 0;
    Stmt *base0 = split_const_offset(ptrs[0]->indices[last], &c0);
    for (int i = 1; i < n; i++) {
      int ci = 0;
      Stmt *basei = split_const_offset(ptrs[i]->indices[last], &ci);
      if ((base0 == nullptr) != (basei == nullptr))
        return;
      if (base0 != nullptr && !irpass::analysis::same_value(base0, basei)) {
        bail("lastbase");
        return;
      }
      if (ci != c0 + i) {
        if (FILE *dg = fopen("/tmp/qd_veclang.log", "a")) {
          fprintf(dg, "stride detail: gather_n=%d c0=%d lane%d ci=%d\n", n, c0, i, ci);
          fclose(dg);
        }
        bail("stride");
        return;
      }
    }
    auto dt = ptrs[0]->ret_type.ptr_removed()->as<PointerType>()->get_pointee_type();
    auto vec_tensor = TypeFactory::get_instance().get_tensor_type(std::vector<int>{n}, dt);
    auto cloned = std::unique_ptr<Stmt>(ptrs[0]->clone());
    cloned->ret_type = TypeFactory::get_instance().get_pointer_type(vec_tensor);
    cloned->ret_type.set_is_pointer(true);

    auto load_stmt = std::make_unique<GlobalLoadStmt>(cloned.get());
    load_stmt->ret_type = vec_tensor;

    stmt->replace_usages_with(load_stmt.get());
    modifier_.insert_before(stmt, std::move(cloned));
    modifier_.insert_before(stmt, std::move(load_stmt));
    modifier_.erase(stmt);
    vectorized_++;
  }
};

void lower_matrix_ptr(IRNode *root, bool force_scalarize, bool arch_cpu) {
  QD_AUTO_PROF;

  // CPU-only: the SPIRV/Metal lowering has no representation for the
  // vector-typed ExternalPtrStmt this pass produces (§12.41).
  bool vec_mem = arch_cpu && std::getenv("QD_MATRIX_VEC_MEM") != nullptr &&
                 std::string(std::getenv("QD_MATRIX_VEC_MEM")) != "0";
  if (!force_scalarize) {
    if (vec_mem) {
      VectorizeContiguousExtInit vec_pass{root, true};
      if (vec_pass.vectorized_ > 0) {
        if (FILE *dg = fopen("/tmp/qd_veclang.log", "a")) {
          fprintf(dg, "vectorized %d gathers\n", vec_pass.vectorized_);
          fclose(dg);
        }
      }
    }
    GatherValidAOSGlobalPtrStmt gather_valid_aos_global_ptr_pass(root);

    LowerAOSGlobalPtrStmt lower_aos_global_ptr_stmt_pass(
        root, gather_valid_aos_global_ptr_pass.invalid_aos_global_ptr_stmts_);
  }

  ScalarizeMatrixPtr scalarize_matrix_ptr_pass(root);
  LowerMatrixPtr::run(root);
  RemoveMatrixOfPtr::run(root);
}

}  // namespace irpass

}  // namespace quadrants::lang
