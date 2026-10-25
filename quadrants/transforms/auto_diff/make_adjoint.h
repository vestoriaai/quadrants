#pragma once

namespace quadrants::lang {

class Block;

// Stage: Reverse-mode codegen. Emit the reverse pass IR in `ib` (an Independent Block produced by
// identify_independent_blocks). Forward IR must already have been shaped by ir_shaping + forward_state_spill.
// `ad_stack_size` is the configured stack capacity (config.ad_stack_size); the WhileStmt reverse visitor
// uses it to size its synthetic iteration-trail stack and refuses to run without it (stack-mode only).
void make_adjoint(Block *ib, int ad_stack_size = 0);

}  // namespace quadrants::lang
