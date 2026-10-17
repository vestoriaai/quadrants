"""Gate 19: CPU parallel-kernel head-declaration state is per-iteration.

Defect line mujocoqd/cpu-head-state-per-iteration
(docs/cpu_head_state_development_spec.md).

Pre-fix, a kernel-root alloca that a top-level loop stores to is promoted by
the offload cross-task promotion into a single launch-shared global temp slot:
parallel tasks race on it (silent nondeterministic numerical corruption), and a
serialized task only makes the shared-slot timing deterministic (still
launch-shared accumulation, not warp semantics). The fix (offload.cpp
rebuild_head_state_per_iteration) rebuilds storing head state per iteration
inside the loop task: alloca + loop-free prefix-store clones at task entry.
Read-only cross-task promotion is untouched (pinned by the read-only form).

Forms:
  H1 head           : `acc = 0` before the parallel loop, `acc += 1` inside.
                      Post-fix: all 1s in every launch (pre-fix: monotonic 1..N,
                      racy for loop ranges above the CPU chunking threshold).
  H2 head serialize : same head form with qd.loop_config(serialize=True).
                      Post-fix: STILL monotonic 1..16. ADJUDICATED R3
                      (2026-10-18): serialize = sequential-task semantics,
                      loop-carried carry is the declared contract
                      (_cpu_paths.py docstring: "one thread, program order, so
                      the loop-carried accumulator is exact"); the spec's
                      serialize-form expectation (all 1s) was wrong and is
                      corrected in the spec status section. The rebuild
                      therefore covers write-and-read head state in
                      UNSERIALIZED top-level loops only - exactly the loops the
                      CPU thread pool runs concurrently, i.e. the racy defect
                      class. Criterion boundary: write-only head state is NOT
                      rebuilt (premise: no in-loop reads, consumer after the
                      loop - the test_ret_4k class, pinned by H6); a future
                      write-only-with-in-loop-read form must be rejected loudly
                      or caught by a lint rule, not silently missed (recorded
                      in the ledger limitations).
  H3 body control   : acc declared in-body; must stay all 1s (per-iteration).
  H4 read-only head : head var only read inside the loop; must still see the
                      loop-entry value - the read-only promotion path stays
                      bit-identical.
  H5 computed-offset: the real defect shape from the field (mujoqo _flex_edges):
                      head-declared cursor rebound inside the parallel loop,
                      then used to compute the store offset. Post-fix: every
                      iteration stores to its own slot, deterministically.
  H6 write-only head : head-declared vector written at disjoint slices inside
                      the loop and consumed only AFTER the loop (the upstream
                      test_ret_4k class). Legal and race-free pre-fix -> must
                      stay on the launch-shared promotion path (write-and-read
                      is the rebuild criterion; write-only is untouched).

CPU only (the defect is arch-independent IR; the metal path runs the same
transform - covered indirectly by the metal arms of the other union gates;
metal end-to-end validation is out of scope for this line).

Run: <venv>/bin/python tests/robostudio/gate_cpu_head_state.py
Exit 0 iff every form passes; prints "RESULT: PASS".
"""
import sys

import numpy as np
import quadrants as qd

qd.init(arch=qd.cpu)

N = 64
LAUNCHES = 20


@qd.kernel
def k_head_acc(out: qd.types.ndarray(dtype=int, ndim=1), __n0: int):
    acc = 0  # head: before the parallel loop
    for __t in range(__n0):
        acc += 1
        out[__t] = acc


@qd.kernel
def k_head_acc_ser(out: qd.types.ndarray(dtype=int, ndim=1), __n0: int):
    acc = 0  # head + serialized outer loop (post-fix per-iteration by design)
    qd.loop_config(serialize=True)
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
def k_readonly_head(out: qd.types.ndarray(dtype=int, ndim=1), __n0: int):
    k = 42  # head var, only read inside the loop: promotion path unchanged
    for __t in range(__n0):
        out[__t] = k + 1


@qd.kernel
def k_computed_offset(out: qd.types.ndarray(dtype=int, ndim=1),
                      base_in: qd.types.ndarray(dtype=int, ndim=1), __n0: int):
    base = 0  # head-declared cursor, rebound inside the loop
    for __t in range(__n0):
        base = base_in[__t]
        out[base] = __t


vec_n = qd.types.vector(N, qd.i32)


@qd.kernel
def k_writeonly_head_ret() -> vec_n:
    ret = vec_n(0)  # head var, disjoint slice writes only, consumed post-loop
    for i in range(N):
        ret[i] = i
    return ret


def run(kernel, args, out_shape=(N,)):
    o = qd.ndarray(dtype=qd.i32, shape=out_shape)
    outs = []
    for _ in range(LAUNCHES):
        o.from_numpy(np.zeros(out_shape, dtype=np.int32))
        kernel(o, *args)
        outs.append(o.to_numpy().copy())
    return outs


failures = []
ones = np.ones(N, dtype=np.int32)

# H1 head
head_runs = run(k_head_acc, (N,))
if not all(np.array_equal(r, ones) for r in head_runs):
    failures.append(f"H1 head: want all 1 in every launch, got {head_runs[0].tolist()}")

# H2 head + serialize: keeps pre-fix deterministic accumulation (1..16);
# serialize semantics unchanged by the fix - see the SPEC DEVIATION note above
arange = np.arange(1, N + 1, dtype=np.int32)
ser_runs = run(k_head_acc_ser, (N,))
if not all(np.array_equal(r, arange) for r in ser_runs):
    failures.append(f"H2 serialize: want deterministic 1..{N} in every launch, got {ser_runs[0].tolist()}")

# H3 body control
body_runs = run(k_body_acc, (N,))
if not all(np.array_equal(r, ones) for r in body_runs):
    failures.append(f"H3 body: want all 1 in every launch, got {body_runs[0].tolist()}")

# H4 read-only head keeps the promotion read path
ro_runs = run(k_readonly_head, (N,))
want_ro = np.full(N, 43, dtype=np.int32)
if not all(np.array_equal(r, want_ro) for r in ro_runs):
    failures.append(f"H4 read-only head: want all 43, got {ro_runs[0].tolist()}")

# H5 computed-offset store (field-recorded defect shape)
rng = np.random.default_rng(7)
perm = rng.permutation(N).astype(np.int32)  # collision-free slot permutation
base_arr = qd.ndarray(dtype=qd.i32, shape=(N,))
base_arr.from_numpy(perm)
co_runs = run(k_computed_offset, (base_arr, N))
for li, r in enumerate(co_runs):
    want = np.zeros(N, dtype=np.int32)
    want[perm] = np.arange(N, dtype=np.int32)
    if not np.array_equal(r, want):
        failures.append(f"H5 computed-offset launch {li}: slot contents corrupted (race)")
        break

# determinism across launches for the previously racy forms
for name, runs in (("H1 head", head_runs), ("H2 serialize", ser_runs), ("H5 computed-offset", co_runs)):
    if any(not np.array_equal(r, runs[0]) for r in runs):
        failures.append(f"{name}: outputs differ across launches (nondeterministic)")

# H6 write-only head tensor keeps the launch-shared promotion path
wo = k_writeonly_head_ret()
want_wo = np.arange(N, dtype=np.int32)
if not np.array_equal(wo.to_numpy(), want_wo):
    failures.append(f"H6 write-only head: want ret[i]==i, got {wo.to_numpy().tolist()[:8]}...")

if failures:
    for f in failures:
        print("[FAIL]", f)
    print("RESULT: FAIL")
    sys.exit(1)
print(f"RESULT: PASS ({LAUNCHES} launches x 5 forms + write-only form, N={N})")
