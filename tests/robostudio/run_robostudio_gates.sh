#!/bin/zsh
# robostudio-main union gate runner — one command, all robostudio gate families.
#
# Reproduces the consolidation checks plus gate 9 (reassociate flag),
# gate 10 (simt.subgroup/block on Metal + CPU - mujocoqd/subgroup-cpu-support),
# gate 11 (C2 emitter fixes - mujocoqd/c2-emitter-fixes), gate 12
# (to_numpy_type compound dtypes - mujocoqd/n2-to-numpy-type; CPU only, the
# API is arch-independent), gate 13 (A10 nested-loop order semantics -
# mujocoqd/a10-nested-loop-raw; cpu+metal dual arm), gate 14 (A19
# store-to-load-forwarding fallback - mujocoqd/a19-stlf-fallback; CPU only,
# vendored mujoco_qd _sensor_touch repro), gate 15 (A20 actionable
# branch-scope NameError + predecl recipe - mujocoqd/a20-loopvar-scope; CPU
# only, text-level pin), gate 16 (metal i64 matrix incl segmented sort -
# mujocoqd/metal-i64; cpu+metal dual arm, perf informational) and gate 17
# (early-return compound forms - mujocoqd/a7-m2-compound; cpu+metal dual arm:
# compound slot init + merged-guard rewrite + new loud rejections, F1-F14
# matrix with numpy twins) and gate 18 (qd-native BVH - mujocoqd/bvh;
# quadrants/bvh.py build/traversal/refit vs numpy-twin predicates + warp
# wp.Bvh A/A; warp borrowed via QD_GATE_WARP_SP, its quadrants shadowed by
# $QSP - path order matters; cpu+metal dual arm, 27 assertions each).
# matrix with numpy twins), and gate 19 (CPU parallel-kernel head-declaration
# state per-iteration - mujocoqd/cpu-head-state-per-iteration; CPU only:
# head/serialize/body/read-only/computed-offset forms; unserialized loops
# rebuild per-iteration, serialized loops keep pre-fix deterministic
# semantics - ADJUDICATED R3 (2026-10-18), see the gate header).
#   7  C2        probe_spirv_vectorize.py metal    -> ALL PASS (bitwise scalar==vec)
#   8  R5        gate_r5_host_visible.py metal     -> RESULT: PASS (byte parity +
#                  relative perf pins; small readback <=x0.5 default, 64MB <=x1.2)
#   1  S2        probe_s2_loopvar_assign.py        -> RESULT: PASS
#   2  S3        probe_s3_static_if_forms.py       -> RESULT: PASS (10/10)
#   3  u8        gate_u8_dlpack.py cpu+metal       -> cpu ALL GATES PASS;
#                  metal fails EXACTLY the pinned f64 cell (qd SPIR-V has no f64, T19)
#   4  readback  gate_readback_r2r3.py metal       -> 64MB to<=7ms / from<=5ms + byte sweep
#   5  S-guard   pytest tests/python/test_stale_field.py -p xdist -n0  -> 4 passed
#   6  C0S       battery_c0s_multi_init.py --single-init -> all arms PASS
#                  (multi-init arms raising is the S-guard contract; see
#                  docs/consolidation_robostudio_main.md "Union integration notes")
#
# Usage: zsh tests/robostudio/run_robostudio_gates.sh [venv-python-dir]
#   default venv: ~/Code/quadrants/.venv-consolidated (the documented private
#   verification venv; BUILD.md venv-roles table). Override: pass a venv dir.
#   Torch for the u8 gate is borrowed (PYTHONPATH prefix, venv quadrants must
#   shadow it) via QD_GATE_TORCH_SP, default the qd-render-poc venv path.
# Metal arms (3b/4) wait for GPU idle first (tests/robostudio/_gpu_busy.sh,
# retry 15s up to 5 min); still busy = gate FAIL "gpu-busy" (honest: no timing
# checks on a contended GPU). Exit 0 iff every gate passes.
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
V="${1:-$HOME/Code/quadrants/.venv-consolidated}"
PY="$V/bin/python"
QSP="$V/lib/python3.12/site-packages"
TORCH_SP="${QD_GATE_TORCH_SP:-/Users/mini0/Code/simtest/qd-render-poc/.venv/lib/python3.12/site-packages}"
TMP="$(mktemp -d /tmp/qds_union_gate.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

[[ -x "$PY" ]] || { echo "FATAL: no venv python at $PY (create .venv-consolidated, see docs/build_robostudio.md)" >&2; exit 1; }

PASS=0; NFAIL=0; TOTAL=20
gate_pass() { echo "gate $1/$TOTAL [$2]: PASS"; (( PASS = PASS + 1 )); }
gate_fail() { echo "gate $1/$TOTAL [$2]: FAIL -- $3"; (( NFAIL = NFAIL + 1 )); }

# wait_gpu_idle: 0 if idle (possibly after waiting), 1 if still busy after 5 min
wait_gpu_idle() {
  local deadline=$(( SECONDS + 300 )); local announced=0
  while true; do
    if zsh "$ROOT/tests/robostudio/_gpu_busy.sh" >/dev/null 2>&1; then
      (( announced )) && echo "  gpu idle again; continuing"
      return 0
    fi
    if (( SECONDS >= deadline )); then echo "  gpu still busy after 5 min" >&2; return 1; fi
    if (( ! announced )); then echo "  gpu busy; waiting (retry every 15s, up to 5 min)"; announced=1; fi
    sleep 15
  done
}

echo "=== robostudio union gate ($(date '+%F %T')) ==="
echo "venv: $V"

# 1  S2 ----------------------------------------------------------------------
"$PY" -u "$ROOT/tests/robostudio/probe_s2_loopvar_assign.py" >"$TMP/g1.log" 2>&1 \
  && grep -q "^RESULT: PASS" "$TMP/g1.log" \
  && gate_pass 1 "S2 probe" \
  || { gate_fail 1 "S2 probe" "tail below"; tail -8 "$TMP/g1.log"; }

# 2  S3 ----------------------------------------------------------------------
"$PY" -u "$ROOT/tests/robostudio/probe_s3_static_if_forms.py" >"$TMP/g2.log" 2>&1 \
  && grep -q "^RESULT: PASS" "$TMP/g2.log" \
  && gate_pass 2 "S3 form matrix" \
  || { gate_fail 2 "S3 form matrix" "tail below"; tail -8 "$TMP/g2.log"; }

# 3  u8 DLPack (torch borrowed; our venv's quadrants must shadow the RP copy) -
BORROW=(env PYTHONPATH="$QSP:$TORCH_SP")
if ! "${BORROW[@]}" "$PY" -c "import quadrants, torch; assert '$V' in quadrants.__file__, quadrants.__file__" 2>"$TMP/g3_borrow.log"; then
  gate_fail 3 "u8 gate" "torch-borrow preflight failed (QD_GATE_TORCH_SP=$TORCH_SP)"
else
  G3_OK=1
  "${BORROW[@]}" "$PY" -u "$ROOT/tests/robostudio/gate_u8_dlpack.py" --arch cpu >"$TMP/g3_cpu.log" 2>&1
  if (( $? == 0 )) && grep -q "^RESULT: ALL GATES PASS" "$TMP/g3_cpu.log"; then
    echo "  u8 cpu: ALL GATES PASS"
  else
    G3_OK=0; echo "  u8 cpu: expected ALL PASS:"; grep "^\[FAIL\]\|^RESULT" "$TMP/g3_cpu.log" || tail -8 "$TMP/g3_cpu.log"
  fi
  if wait_gpu_idle; then
    "${BORROW[@]}" "$PY" -u "$ROOT/tests/robostudio/gate_u8_dlpack.py" --arch metal >"$TMP/g3_metal.log" 2>&1
    if grep -q "^RESULT: 1 FAILURES: \['G1 f64 matrix cell'\]" "$TMP/g3_metal.log" \
       && (( $(grep -c "^\[FAIL\]" "$TMP/g3_metal.log") == 1 )); then
      echo "  u8 metal: exactly the pinned f64-cell failure (T19)"
    else
      G3_OK=0; echo "  u8 metal: unexpected result (want exactly the pinned f64 cell):"
      grep "^\[FAIL\]\|^RESULT" "$TMP/g3_metal.log" || tail -8 "$TMP/g3_metal.log"
    fi
  else
    G3_OK=0; echo "  u8 metal: gpu-busy"
  fi
  (( G3_OK )) && gate_pass 3 "u8 gate (cpu+metal)" || gate_fail 3 "u8 gate (cpu+metal)" "see arms above"
fi

# 4  readback ----------------------------------------------------------------
if wait_gpu_idle; then
  "$PY" -u "$ROOT/tests/robostudio/gate_readback_r2r3.py" >"$TMP/g4.log" 2>&1 \
    && grep -q "^RESULT: PASS" "$TMP/g4.log" \
    && gate_pass 4 "readback R2/R3 pins" \
    || { gate_fail 4 "readback R2/R3 pins" "output below"; cat "$TMP/g4.log"; }
else
  gate_fail 4 "readback R2/R3 pins" "gpu-busy"
fi

# 5  S-guard stale-field battery --------------------------------------------
if [[ -x "$PY" ]] && "$PY" -c "import xdist, pytest" 2>/dev/null; then
  (cd "$ROOT" && "$PY" -m pytest tests/python/test_stale_field.py -p xdist -n0 -q) >"$TMP/g5.log" 2>&1 \
    && grep -q "4 passed" "$TMP/g5.log" \
    && gate_pass 5 "stale-field battery (4/4)" \
    || { gate_fail 5 "stale-field battery (4/4)" "tail below"; tail -12 "$TMP/g5.log"; }
else
  gate_fail 5 "stale-field battery (4/4)" "pytest/xdist missing in venv (uv pip install pytest pytest-xdist cffi setuptools)"
fi

# 6  C0S single-init battery -------------------------------------------------
"$PY" -u "$ROOT/tests/robostudio/battery_c0s_multi_init.py" --single-init >"$TMP/g6.log" 2>&1
BAD=$(grep -cE "^  [a-z0-9_]+: (FAIL|RAISED)$" "$TMP/g6.log")
GOOD=$(grep -cE "^  [a-z0-9_]+: PASS$" "$TMP/g6.log")
if (( BAD == 0 && GOOD == 4 )); then
  gate_pass 6 "C0S single-init battery (4 arms PASS)"
else
  gate_fail 6 "C0S single-init battery" "want 4 PASS / 0 bad, got $GOOD PASS / $BAD bad (tail below)"; tail -12 "$TMP/g6.log"
fi

# 7  C2 spirv-vectorize bitwise parity ---------------------------------------
# Two subprocess arms (scalar / vec) re-executed by the probe itself; PASS iff
# every kernel's output hash is bitwise identical across arms.
if [[ -x "$PY" ]]; then
  "$PY" -u "$ROOT/tests/robostudio/probe_spirv_vectorize.py" metal >"$TMP/g7.log" 2>&1 \
    && grep -q "ALL PASS" "$TMP/g7.log" \
    && gate_pass 7 "C2 spirv-vectorize bitwise" \
    || { gate_fail 7 "C2 spirv-vectorize bitwise" "output below"; tail -16 "$TMP/g7.log"; }
else
  gate_fail 7 "C2 spirv-vectorize bitwise" "venv python missing"
fi

# 8  R5 host-visible ndarray readback ----------------------------------------
# Byte parity default-vs-host_visible + obs-pattern ordering + relative perf
# pins (metal perf arm waits for GPU idle; perf pins are relative in-process,
# per the gate-4 absolute-pin lesson).
if wait_gpu_idle; then
  "$PY" -u "$ROOT/tests/robostudio/gate_r5_host_visible.py" >"$TMP/g8.log" 2>&1 \
    && grep -q "^RESULT: PASS" "$TMP/g8.log" \
    && gate_pass 8 "R5 host-visible readback" \
    || { gate_fail 8 "R5 host-visible readback" "output below"; tail -20 "$TMP/g8.log"; }
else
  gate_fail 8 "R5 host-visible readback" "gpu-busy"
fi

# 9  reassociate flag (mujocoqd/reassociate-flag) ------------------------------
# Three subprocess arms (scalar / vec / rea): OFF-path bitwise parity (hard),
# non-bitwise reassociated reductions (hard), implication semantics (hard),
# 1e-5 accuracy vs f64 (hard); in-process interleaved perf pin is WARN-only.
if wait_gpu_idle; then
  "$PY" -u "$ROOT/tests/robostudio/probe_reassociate_flag.py" metal >"$TMP/g9.log" 2>&1 \
    && grep -q "ALL HARD GATES PASS" "$TMP/g9.log" \
    && gate_pass 9 "reassociate flag" \
    || { gate_fail 9 "reassociate flag" "output below"; tail -20 "$TMP/g9.log"; }
else
  gate_fail 9 "reassociate flag" "gpu-busy"
fi

# 10 subgroup/block stack, Metal + CPU arms (mujocoqd/subgroup-cpu-support) ---------------------
# Metal arm: the 24/24 availability probe (SPIR-V -> MSL). CPU arm: the virtual-subgroup
# (group_size=1) + virtual-block (true concurrent threads) probe plus the algorithms CPU gate
# whose three former expected-error staircase pins are now positive value gates.
G10_OK=1
if wait_gpu_idle; then
  "$PY" -u "$ROOT/tests/robostudio/probe_subgroup_metal.py" metal >"$TMP/g10_metal.log" 2>&1
  if (( $? == 0 )) && grep -q "ALL PASS" "$TMP/g10_metal.log"; then
    echo "  subgroup metal: ALL PASS"
  else
    G10_OK=0; echo "  subgroup metal: expected ALL PASS:"; grep "^\[FAIL\]\|^\[RAISED\]\|SUMMARY" "$TMP/g10_metal.log" || tail -8 "$TMP/g10_metal.log"
  fi
else
  G10_OK=0; echo "  subgroup metal: gpu-busy"
fi
"$PY" -u "$ROOT/tests/robostudio/probe_subgroup_cpu.py" cpu >"$TMP/g10_cpu.log" 2>&1 \
  && grep -q "ALL PASS" "$TMP/g10_cpu.log" \
  && echo "  subgroup cpu: ALL PASS" \
  || { G10_OK=0; echo "  subgroup cpu: expected ALL PASS:"; grep "^\[FAIL\]" "$TMP/g10_cpu.log" || tail -8 "$TMP/g10_cpu.log"; }
"$PY" -u "$ROOT/tests/robostudio/probe_algorithms_cpu.py" cpu >"$TMP/g10_algos.log" 2>&1 \
  && grep -q "ALL PASS" "$TMP/g10_algos.log" \
  && echo "  algorithms cpu: ALL PASS" \
  || { G10_OK=0; echo "  algorithms cpu: expected ALL PASS:"; grep "^\[FAIL\]" "$TMP/g10_algos.log" || tail -8 "$TMP/g10_algos.log"; }
(( G10_OK )) && gate_pass 10 "subgroup stack (metal+cpu)" || gate_fail 10 "subgroup stack (metal+cpu)" "see arms above"

# 11 C2 emitter fixes (mujocoqd/c2-emitter-fixes) ----------------------------------------------
# Three HEAD C2 defects, one hard-gate arm each: arg-in-DAG env miss (G1),
# k_elementwise store-loop admission + real float4 vectorization (G2), and the
# coeff-0 remainder duplicate-registration (G3). Scalar arm = flags OFF.
if wait_gpu_idle; then
  "$PY" -u "$ROOT/tests/robostudio/probe_c2_emitter_fixes.py" metal >"$TMP/g11.log" 2>&1 \
    && grep -q "ALL HARD GATES PASS" "$TMP/g11.log" \
    && gate_pass 11 "c2-emitter-fixes" \
    || { gate_fail 11 "c2-emitter-fixes" "output below"; tail -16 "$TMP/g11.log"; }
else
  gate_fail 11 "c2-emitter-fixes" "gpu-busy"
fi

# 12 to_numpy_type compound dtypes (mujocoqd/n2-to-numpy-type) ------------------
# to_numpy_type accepts VectorType/MatrixType and yields the element np dtype
# (scalar table pinned object-exact; compound direct calls; loud unknown dtype;
# compound-dtype ndarray round trip). CPU-only: the API is arch-independent.
"$PY" -u "$ROOT/tests/robostudio/probe_n2_to_numpy_type.py" cpu >"$TMP/g12.log" 2>&1 \
  && grep -q "RESULT: PASS" "$TMP/g12.log" \
  && gate_pass 12 "n2 to_numpy_type compound" \
  || { gate_fail 12 "n2 to_numpy_type compound" "output below"; tail -16 "$TMP/g12.log"; }

# 13 A10 nested-loop order semantics (mujocoqd/a10-nested-loop-raw) ------------
# Corrected-order column/row Cholesky forms must match their numpy twins on
# BOTH backends, and the degenerate invalid-order form must equal its twin
# exactly (faithful-compilation pin; the original A10 "miscompile" was an
# algorithm-order bug in the evidence, see the probe header + ledger).
"$PY" -u "$ROOT/tests/robostudio/probe_a10_nested_loop_order.py" cpu >"$TMP/g13_cpu.log" 2>&1 \
  && grep -q "RESULT: PASS" "$TMP/g13_cpu.log" && G13=1 || { G13=0; echo "  a10 cpu arm: expected RESULT: PASS:"; tail -8 "$TMP/g13_cpu.log"; }
if wait_gpu_idle; then
  "$PY" -u "$ROOT/tests/robostudio/probe_a10_nested_loop_order.py" metal >"$TMP/g13_metal.log" 2>&1 \
    && grep -q "RESULT: PASS" "$TMP/g13_metal.log" && G13=$((G13 + 1)) || { echo "  a10 metal arm: expected RESULT: PASS:"; tail -8 "$TMP/g13_metal.log"; }
else
  echo "  a10 metal arm: skipped (gpu-busy)"
fi
(( G13 == 2 )) && gate_pass 13 "a10 nested-loop order (cpu+metal)" || gate_fail 13 "a10 nested-loop order (cpu+metal)" "see arms above"

# 14 A19 store-to-load-forwarding fallback (mujocoqd/a19-stlf-fallback) --------
# The vendored mujoco_qd _sensor_touch kernel + ray closure, cpu only (the
# defect is an IR-form defect, arch-independent). On a wheel without the fix
# this dies at control_flow_graph.cpp store_to_load_forwarding@425; with the
# fix it must compile through and match the pinned outputs (see the probe
# header for the pre-existing ray_geom dist-field note).
# A37 (2026-10-18): this probe is also the host of a nondeterministic
# compiler-side UAF (SEGV inside compile_kernel -> __dynamic_cast on a
# dangling pointer, exposure depends on process memory layout). A single
# run passing is luck - the gate therefore requires N=3 consecutive clean
# runs (any nonzero exit = FAIL) until the UAF is fixed and the census
# (mujoqo _probe/a37_uaf_census.py) goes STABLE.
G14_OK=1
for g14i in 1 2 3; do
  "$PY" -u "$ROOT/tests/robostudio/probe_a19_stlf_fallback.py" >"$TMP/g14_$g14i.log" 2>&1 \
    && grep -q "RESULT: PASS" "$TMP/g14_$g14i.log" || { G14_OK=0; tail -16 "$TMP/g14_$g14i.log"; break; }
done
(( G14_OK == 1 )) \
  && gate_pass 14 "a19 stlf fallback (x3 consensus)" \
  || gate_fail 14 "a19 stlf fallback (x3 consensus)" "output above"

# 15 A20 branch-scope NameError (mujocoqd/a20-loopvar-scope) --------------------
# A name first bound inside a runtime if branch / loop body is invisible after
# the block closes; the failed lookup must raise the actionable predeclaration
# diagnostic WITHOUT the chained builtins AttributeError, true typos must keep
# the generic message, and the predecl recipe kernel must compile + compute
# (the fix handed to the mujoco_qd collision_primitive owner). CPU-only:
# the pin is frontend/text-level.
"$PY" -u "$ROOT/tests/robostudio/probe_a20_branch_scope_diag.py" cpu >"$TMP/g15.log" 2>&1 \
  && grep -q "RESULT: PASS" "$TMP/g15.log" \
  && gate_pass 15 "a20 branch-scope NameError" \
  || { gate_fail 15 "a20 branch-scope NameError" "output below"; tail -16 "$TMP/g15.log"; }

# 16 metal i64 matrix (mujocoqd/metal-i64) ------------------------------------
# Full i64 form matrix (load/store, arith, shifts/bitwise, compares, casts,
# clamp, bit_cast, INT64_MAX sentinel) + plain sort i64 keys + segmented sort
# i64 keys (fixed here: _seg_scatter key_width branch) + segmented i32
# control. Correctness gates the exit code; the 8M i64/i32 perf section is
# informational (interleaved paired medians, >4x slowdown prints an alert).
G16=0
"$PY" -u "$ROOT/tests/robostudio/probe_metal_i64.py" cpu >"$TMP/g16_cpu.log" 2>&1 \
  && grep -q "I64 PROBE \[cpu\]: ALL PASS" "$TMP/g16_cpu.log" && G16=1 \
  || { echo "  i64 cpu arm: expected I64 PROBE ALL PASS:"; tail -12 "$TMP/g16_cpu.log"; }
if wait_gpu_idle; then
  "$PY" -u "$ROOT/tests/robostudio/probe_metal_i64.py" metal >"$TMP/g16_metal.log" 2>&1 \
    && grep -q "I64 PROBE \[metal\]: ALL PASS" "$TMP/g16_metal.log" && G16=$((G16 + 1)) \
    || { echo "  i64 metal arm: expected I64 PROBE ALL PASS:"; tail -12 "$TMP/g16_metal.log"; }
else
  echo "  i64 metal arm: skipped (gpu-busy)"
fi
(( G16 == 2 )) && gate_pass 16 "i64 segmented sort (cpu+metal)" || gate_fail 16 "i64 segmented sort (cpu+metal)" "see arms above"

# 17 early-return compound forms (mujocoqd/a7-m2-compound) ---------------------
# @qd.kernel/@qd.func(early_return=True) root fix: compound slots (vec/mat
# annotations init via qd.cast(T(0), T.dtype), the broadcast constructor keeps
# the slot a correctly-shaped tensor) + post-return statements merged into ONE
# __er_done guard (per-statement guards stranded first-bindings in sibling
# runtime-if blocks -> actionable branch-scope NameError) + loud rejections
# (arity mismatch / bare+value mix / nested tuple / struct return). The probe
# pins the pre-existing 7 forms AND the F1-F14 matrix (numpy twins), incl. the
# box_box / ray_mesh_with_bvh return shapes. cpu+metal dual arm.
G17=0
"$PY" -u "$ROOT/tests/robostudio/probe_early_return_forms.py" cpu >"$TMP/g17_cpu.log" 2>&1 \
  && grep -q "EARLY_RETURN GATE \[cpu\]: ALL PASS" "$TMP/g17_cpu.log" && G17=1 \
  || { echo "  early-return cpu arm: expected EARLY_RETURN GATE ALL PASS:"; tail -12 "$TMP/g17_cpu.log"; }
if wait_gpu_idle; then
  "$PY" -u "$ROOT/tests/robostudio/probe_early_return_forms.py" metal >"$TMP/g17_metal.log" 2>&1 \
    && grep -q "EARLY_RETURN GATE \[metal\]: ALL PASS" "$TMP/g17_metal.log" && G17=$((G17 + 1)) \
    || { echo "  early-return metal arm: expected EARLY_RETURN GATE ALL PASS:"; tail -12 "$TMP/g17_metal.log"; }
else
  echo "  early-return metal arm: skipped (gpu-busy)"
fi
(( G17 == 2 )) && gate_pass 17 "early-return compound forms (cpu+metal)" || gate_fail 17 "early-return compound forms (cpu+metal)" "see arms above"

# 18 qd-native BVH (mujocoqd/bvh) ------------------------------------------------
# quadrants/bvh.py (file-loaded from the worktree; the venv runtime provides
# the compiled quadrants). Build invariants + numpy-twin traversal semantics +
# warp wp.Bvh A/A + ray hand cases (any-hit cutoff, nearest via iterator
# narrowing) + refit + empty/n=1/leaf_size/determinism. cpu arm must pass;
# metal arm runs the same 27 assertions.
G18=0
BVH_BORROW=(env PYTHONPATH="$QSP:${QD_GATE_WARP_SP:-/Users/mini0/Code/mujoqo/_port.venv/lib/python3.12/site-packages}")
if ! "${BVH_BORROW[@]}" "$PY" -c "import quadrants, warp; assert '$V' in quadrants.__file__, quadrants.__file__" 2>"$TMP/g18_borrow.log"; then
  gate_fail 18 "qd-native BVH" "warp-borrow preflight failed (QD_GATE_WARP_SP=$QD_GATE_WARP_SP)"
else
  "${BVH_BORROW[@]}" "$PY" -u "$ROOT/tests/robostudio/gate_bvh.py" cpu >"$TMP/g18_cpu.log" 2>&1
  if (( $? == 0 )) && grep -q "RESULT: BVH GATE \[cpu\]: ALL PASS" "$TMP/g18_cpu.log"; then
    G18=1; echo "  bvh cpu: ALL PASS"
  else
    G18=0; echo "  bvh cpu: expected ALL PASS:"; grep "^\[FAIL\]\|^RESULT" "$TMP/g18_cpu.log" || tail -8 "$TMP/g18_cpu.log"
  fi
  if wait_gpu_idle; then
    "${BVH_BORROW[@]}" "$PY" -u "$ROOT/tests/robostudio/gate_bvh.py" metal >"$TMP/g18_metal.log" 2>&1
    if (( $? == 0 )) && grep -q "RESULT: BVH GATE \[metal\]: ALL PASS" "$TMP/g18_metal.log"; then
      echo "  bvh metal: ALL PASS"
    else
      G18=0; echo "  bvh metal: expected ALL PASS:"; grep "^\[FAIL\]\|^RESULT" "$TMP/g18_metal.log" || tail -8 "$TMP/g18_metal.log"
    fi
  else
    G18=0; echo "  bvh metal: gpu-busy"
  fi
  (( G18 )) && gate_pass 18 "qd-native BVH (cpu+metal)" || gate_fail 18 "qd-native BVH (cpu+metal)" "see arms above"
fi
# 19 CPU head-declaration state per-iteration (mujocoqd/cpu-head-state-per-iteration) ---
# Head-declared scalars written AND read inside an UNSERIALIZED top-level loop
# must rebuild per iteration inside the loop task (offload.cpp
# rebuild_head_state_per_iteration), not live in a launch-shared global temp
# (pre-fix: parallel race). Serialized range-fors keep their pre-fix
# deterministic launch-shared accumulation - ADJUDICATED R3 (2026-10-18):
# serialize=sequential-task semantics, loop-carried carry is the declared
# contract (_cpu_paths.py docstring); the spec's serialize-form expectation
# was wrong, corrected in the spec status section. Five forms x 20 launches:
# head all-1, serialize deterministic 1..N, body all-1, read-only head
# unchanged (promotion path bit-identical), computed-offset store
# deterministic. Plus the write-only disjoint-slice head tensor consumed
# post-loop (test_ret_4k class) which must stay on the promotion path. CPU only.
"$PY" -u "$ROOT/tests/robostudio/gate_cpu_head_state.py" >"$TMP/g19.log" 2>&1 \
  && grep -q "^RESULT: PASS" "$TMP/g19.log" \
  && gate_pass 19 "cpu head state per-iteration" \
  || { gate_fail 19 "cpu head state per-iteration" "output below"; tail -16 "$TMP/g19.log"; }

# 20 bool-cast C truthiness (mujocoqd/a34-bool-cast) ---------------------------
# bool(<float>) must compile to C truthiness (NaN -> True, unordered compare)
# on both backends; int/ptr path unchanged; implicit truthiness stays loudly
# rejected. Pre-fix wheels abort (cpu) / loudly reject (metal) - the gate
# process itself failing IS the regression signal (see ledger A34).
G20=0
"$PY" -u "$ROOT/tests/robostudio/probe_bool_cast.py" cpu >"$TMP/g20_cpu.log" 2>&1 \
  && grep -q "BOOL CAST GATE \\[cpu\\]: ALL PASS" "$TMP/g20_cpu.log" && G20=1 \
  || { echo "  bool-cast cpu arm: expected ALL PASS:"; tail -12 "$TMP/g20_cpu.log"; }
if wait_gpu_idle; then
  "$PY" -u "$ROOT/tests/robostudio/probe_bool_cast.py" metal >"$TMP/g20_metal.log" 2>&1 \
    && grep -q "BOOL CAST GATE \\[metal\\]: ALL PASS" "$TMP/g20_metal.log" && G20=$((G20 + 1)) \
    || { echo "  bool-cast metal arm: expected ALL PASS:"; tail -12 "$TMP/g20_metal.log"; }
else
  echo "  bool-cast metal arm: skipped (gpu-busy)"
fi
(( G20 == 2 )) && gate_pass 20 "bool-cast C truthiness (cpu+metal)" || gate_fail 20 "bool-cast C truthiness (cpu+metal)" "see arms above"

echo "UNION GATE: $PASS/$TOTAL PASS"
if (( NFAIL > 0 )); then echo "(logs kept in $TMP)"; trap - EXIT; exit 1; fi
exit 0
