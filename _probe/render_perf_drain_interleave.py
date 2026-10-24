"""Interleaved pristine/no-sync bvh measurement: submit vs mega_drain split.

Swaps /tmp/a53b_venv's quadrants/bvh.py between the two variants and runs the
drain-split probe alternately x3. Usage: python render_perf_drain_interleave.py
"""

import os
import shutil
import subprocess
import sys

SP_BVH = "/tmp/a53b_venv/lib/python3.12/site-packages/quadrants/bvh.py"
PY = "/Users/mini0/Code/mujoqo/_port.venv/bin/python"
PROBE = "/Users/mini0/Code/quadrants-rpwt/_probe/render_perf_drainsplit.py"
CWD = "/Users/mini0/Code/mujoqo-metal"


def run(tag):
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    env["QD_ARCH"] = "metal"
    proc = subprocess.run([PY, PROBE], capture_output=True, text=True, env=env, cwd=CWD, timeout=900)
    for line in proc.stdout.splitlines():
        if line.startswith("DRAIN"):
            print(f"{tag}: {line}", flush=True)
            return
    print(f"{tag}: NO OUTPUT rc={proc.returncode} {proc.stderr[-300:]}", flush=True)


for rnd in range(3):
    shutil.copy("/tmp/bvh_pristine.py", SP_BVH)
    run(f"r{rnd} pristine(448 sync)")
    shutil.copy("/tmp/bvh_nosync.py", SP_BVH)
    run(f"r{rnd} nosync   (3 sync)  ")
