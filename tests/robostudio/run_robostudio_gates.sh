#!/bin/zsh
# robostudio-main union gate runner — one command, all robostudio gate families.
#
# Reproduces (in order) the exact checks run at the 2026-10-11 consolidation:
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

[[ -x "$PY" ]] || { echo "FATAL: no venv python at $PY (create .venv-consolidated, see BUILD.md)" >&2; exit 1; }

PASS=0; NFAIL=0
gate_pass() { echo "gate $1/6 [$2]: PASS"; (( PASS = PASS + 1 )); }
gate_fail() { echo "gate $1/6 [$2]: FAIL -- $3"; (( NFAIL = NFAIL + 1 )); }

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

echo "UNION GATE: $PASS/6 PASS"
if (( NFAIL > 0 )); then echo "(logs kept in $TMP)"; trap - EXIT; exit 1; fi
exit 0
