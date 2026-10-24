"""render-perf: interleaved A/B for the no-sync BVH hypothesis.

A = batwt bvh.py (no per-stage sync) via /tmp/a53b_venv
B = pristine dev142 bvh via _port.venv
Alternate A/B x3 at the 64envs shadows=on cell, gpu_wait median each round.
Rules out time-varying GPU contention; a stable A~2xB gap across interleave =
real effect of sync removal.

Usage: python render_perf_interleave.py   (from mujoqo worktree root)
"""

import os
import subprocess
import time

HERE = os.getcwd()
MATRIX = "_probe/render_perf_matrix.py"
A_PY = "/tmp/a53b_venv/bin/python"
B_PY = "/Users/mini0/Code/mujoqo/_port.venv/bin/python"


def run(python_bin):
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    env["QD_ARCH"] = "metal"
    t0 = time.perf_counter()
    proc = subprocess.run(
        [python_bin, MATRIX, ], capture_output=True, text=True, env=env, timeout=900,
    )
    wall = time.perf_counter() - t0
    row = ""
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT") and "shadows=1" in line:
            row = line.split("kernel_account=")[-1]
    return row, wall


for rnd in range(3):
    ra, wa = run(A_PY)
    rb, wb = run(B_PY)
    print(f"round {rnd}: A(no-sync bvh) 64/1 kernel={ra} (wall {wa:.0f}s) | "
          f"B(dev142 bvh) 64/1 kernel={rb} (wall {wb:.0f}s)", flush=True)
