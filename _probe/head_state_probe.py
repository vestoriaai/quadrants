"""Head-declaration state probe: where do pre-loop scalars live on the qd CPU backend?

Defect line: mujocoqd/cpu-head-state-per-iteration (docs/cpu_head_state_development_spec.md).

Forms (N=16, default multithreaded CPU):
  k_head_acc   : `acc = 0` BEFORE the parallel loop, `acc += 1` inside it.
                 Pre-fix  : one launch-shared global-temp slot, unsynchronised RMW
                            -> monotonic 1..16 (serialize: deterministic 1..16).
                 Post-fix : per-iteration rebuild -> all 1s (warp per-program
                            semantics for the loops the thread pool runs
                            concurrently).
  k_head_acc_ser: same head form with qd.loop_config(serialize=True) on the loop.
                 Post-fix : STILL monotonic 1..16 - serialize=True keeps its
                 pre-fix deterministic launch-shared accumulation semantics.
                 ADJUDICATED R3 (2026-10-18): serialize = sequential-task
                 semantics, loop-carried carry is the declared contract
                 (_cpu_paths.py docstring); see
                 tests/robostudio/gate_cpu_head_state.py and the spec status
                 section.
  k_body_acc   : control, `acc = 0` declared inside the loop (always per-iteration).

Run: <venv>/bin/python _probe/head_state_probe.py
Exit 0 iff every form matches the post-fix expectation (pass --pre-fix to instead
assert the pre-fix defect signature, used for the baseline record).
"""
import sys

import numpy as np
import quadrants as qd

qd.init(arch=qd.cpu, offline_cache=False, cfg_optimization=False)


@qd.kernel
def k_head_acc(out: qd.types.ndarray(dtype=int, ndim=1), __n0: int):
    acc = 0  # head: before the parallel loop
    for __t in range(__n0):
        acc += 1
        out[__t] = acc


@qd.kernel
def k_body_acc(out: qd.types.ndarray(dtype=int, ndim=1), __n0: int):
    for __t in range(__n0):
        acc = 0  # control: declared in-body (per-iteration)
        acc += 1
        out[__t] = acc


@qd.kernel
def k_head_acc_ser(out: qd.types.ndarray(dtype=int, ndim=1), __n0: int):
    acc = 0  # head + serialized outer loop (timing deterministic pre-fix,
    # but still launch-shared: NOT warp semantics)
    qd.loop_config(serialize=True)
    for __t in range(__n0):
        acc += 1
        out[__t] = acc


N = 16
LAUNCHES = 8  # repeat launches: pre-fix head form is nondeterministic across runs


def run(kernel):
    o = qd.ndarray(dtype=qd.i32, shape=(N,))
    outs = []
    for _ in range(LAUNCHES):
        o.from_numpy(np.zeros(N, dtype=np.int32))
        kernel(o, N)
        outs.append(o.to_numpy().copy())
    return outs


def classify(r):
    if np.all(r == 1):
        return "per-iteration (warp semantics)"
    if len(set(r)) > 2 and r[0] == 1 and np.all(np.diff(r) == 1):
        return "monotonic 1..N (single shared storage)"
    resets = [i for i in range(1, len(r)) if r[i] < r[i - 1]]
    if resets:
        return f"per-task sawtooth (reset at {resets})"
    return "other"


head_runs = run(k_head_acc)
body_runs = run(k_body_acc)
ser_runs = run(k_head_acc_ser)

head, body, ser = head_runs[0], body_runs[0], ser_runs[0]
print("head  :", head.tolist())
print("body  :", body.tolist())
print("serial:", ser.tolist())
print("head  ->", classify(head))
print("body  ->", classify(body))
print("serial->", classify(ser))

pre_fix = "--pre-fix" in sys.argv[1:]
failures = []
if pre_fix:
    # Defect signature: head (parallel or serialized) leaves launch-shared state.
    if not all(np.array_equal(r, np.arange(1, N + 1, dtype=np.int32)) for r in head_runs):
        pass  # racy: exact values may vary run to run; classification below is the pin
    if "shared storage" not in classify(head):
        failures.append(f"head: want monotonic shared-storage pre-fix, got {classify(head)}")
    if "shared storage" not in classify(ser):
        failures.append(f"serial: want monotonic shared-storage pre-fix, got {classify(ser)}")
    if not all(np.array_equal(r, np.ones(N, dtype=np.int32)) for r in body_runs):
        failures.append(f"body: want all 1 pre-fix, got {classify(body)}")
else:
    # Post-fix contract: head/body per-iteration; serialize keeps the pre-fix
    # deterministic launch-shared accumulation (see gate 19 header).
    if not all(np.array_equal(r, np.ones(N, dtype=np.int32)) for r in head_runs):
        failures.append(f"head: want all 1 in every launch, got {classify(head)}")
    if not all(np.array_equal(r, np.arange(1, N + 1, dtype=np.int32)) for r in ser_runs):
        failures.append(f"serial: want deterministic 1..{N} in every launch, got {classify(ser)}")
    if not all(np.array_equal(r, np.ones(N, dtype=np.int32)) for r in body_runs):
        failures.append(f"body: want all 1 in every launch, got {classify(body)}")

if failures:
    for f in failures:
        print("[FAIL]", f)
    print("RESULT: FAIL")
    sys.exit(1)
print("RESULT: PASS")
