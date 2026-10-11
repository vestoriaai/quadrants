#!/bin/zsh
# R13 (curobo_alignment_plan §7): GPU busy-check before heavy benches.
# Usage: scripts/_gpu_busy.sh   -> exit 0 if GPU idle, 1 if busy (prints who).
# Apple GPU has no MPS-style isolation: concurrent metal processes pollute
# each other's timings.  Run this before any timing-sensitive bench; the
# plan's rule is <=1 heavy metal bench at a time.

procs=$(ps aux | grep -E "arch[= ]metal|qd\.init|metal_gate|bench_.*metal|sweep_bd" \
        | grep -v grep | grep -v "_gpu_busy" || true)
if [[ -n "$procs" ]]; then
  echo "GPU BUSY — metal/quadrants processes running:"
  echo "$procs"
  exit 1
fi
echo "GPU idle"
exit 0
