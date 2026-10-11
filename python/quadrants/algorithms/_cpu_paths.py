"""CPU-backend serial implementations of the ``qd.algorithms`` device staircases.

The GPU staircases (``_scan.py`` / ``_reduce.py`` / ``_select.py``) are built on block-tier
primitives (SharedArray + block barriers + subgroup reductions), which have no CPU-backend
mapping: the CPU backend has max shared memory 0 and no block/subgroup model over its LLVM
thread pool.  These serial paths give the CPU backend the same public APIs with the same
numerics, at serial-loop speed — sufficient for CPU-side SAP-broadphase-scale workloads
(n ~ 1e5 scans run in microseconds-to-low-milliseconds on an M-series performance core).

All variants open with ``loop_config(serialize=True)``: one thread, program order, so the
loop-carried accumulator is exact.  The accumulator is seeded with a dtype-exact zero
(``arr[0] - arr[0]``) rather than a python literal so i32/u32/f32 arrays all type-check
without casts.  Dispatch happens at kernel-compile time via ``qd.static(is_cpu())`` inside
the public entry points (call sites keep their exact signatures; the GPU staircase code is
never touched).

mujocoqd/compiler-line C5.  Gate: ``tests/robostudio/probe_algorithms_cpu.py``.
"""

from quadrants.lang.impl import current_cfg
from quadrants.lang.kernel_impl import func as _func
from quadrants.lang.misc import cpu as _arch_cpu, loop_config
from quadrants.types.annotations import template
from quadrants.types.primitive_types import i32


def is_cpu() -> bool:
    """Host-side predicate, evaluated inside ``qd.static`` during kernel compilation."""
    return current_cfg().arch == _arch_cpu


@_func
def _cpu_exclusive_scan_add(arr: template(), out: template(), n: i32):
    """Serial exclusive prefix sum: ``out[i] = sum(arr[0:i])``."""
    loop_config(serialize=True)
    if n > 0:
        run = arr[0] - arr[0]
        for i in range(n):
            out[i] = run
            run = run + arr[i]


@_func
def _cpu_reduce_add(arr: template(), out: template(), n: i32):
    """Serial reduction: ``out[0] = sum(arr[0:n])``."""
    loop_config(serialize=True)
    if n > 0:
        run = arr[0] - arr[0]
        for i in range(n):
            run = run + arr[i]
        out[0] = run


@_func
def _cpu_select(arr: template(), flags: template(), out: template(), num_out: template(), n: i32):
    """Serial stream compaction: selected ``arr[i]`` pack into a dense prefix of ``out``;
    ``num_out[0]`` receives the live count."""
    loop_config(serialize=True)
    idx = 0
    for i in range(n):
        if flags[i] != 0:
            out[idx] = arr[i]
            idx = idx + 1
    num_out[0] = idx
