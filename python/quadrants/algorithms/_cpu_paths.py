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

from quadrants.lang.any_array import AnyArray
from quadrants.lang.impl import current_cfg, static
from quadrants.lang.kernel_impl import func as _func
from quadrants.lang.misc import cpu as _arch_cpu, loop_config
from quadrants.lang.ops import bit_cast
from quadrants.types.annotations import template
from quadrants.types.primitive_types import f32, f64, i32, i64, u32, u64


# Count normalization (trace-time, must stay inline in the func body): the frozen contract
# reads counts from a 0-d ndarray (``n[()]`` — traced as an AST subscript, since AnyArray is
# not python-subscriptable); the CPU path additionally accepts a host ``int`` or an i32
# scalar kernel arg (already a device value). The ``static`` guard matters: a raw if or a
# conditional expression would trace both arms.
#     if static(isinstance(n, int)):
#         nc = n
#     elif static(isinstance(n, AnyArray)):
#         nc = n[()]
#     else:
#         nc = n


@_func
def _cpu_twiddle(keys: template(), n: template(), key_dtype: template(), key_width: template()):
    """Forward twiddle in place on ``keys[0:n]``: map key values to monotone-unsigned bit patterns
    (stored in the key dtype). Same transform as the GPU ``_radix_twiddle``: signed ints XOR the
    sign bit; floats flip the sign bit on positives and all bits on negatives (so ordering matches
    ``numpy.sort``, NaN last); unsigned is the identity."""
    loop_config(serialize=True)
    for i in range(n):
        if static(key_width == 32):
            v = bit_cast(keys[i], u32)
            if static(key_dtype == i32):
                w = v ^ u32(0x80000000)
            elif static(key_dtype == f32):  # input sign bit picks the mask
                w = v ^ u32(0x80000000)
                if (v & u32(0x80000000)) != u32(0):
                    w = v ^ u32(0xFFFFFFFF)
            else:  # u32: identity
                w = v
            keys[i] = bit_cast(w, key_dtype)
        else:  # 64-bit
            v = bit_cast(keys[i], u64)
            if static(key_dtype == i64):
                w = v ^ u64(0x8000000000000000)
            elif static(key_dtype == f64):
                w = v ^ u64(0x8000000000000000)
                if (v & u64(0x8000000000000000)) != u64(0):
                    w = v ^ u64(0xFFFFFFFFFFFFFFFF)
            else:  # u64: identity
                w = v
            keys[i] = bit_cast(w, key_dtype)


@_func
def _cpu_untwiddle(keys: template(), n: template(), key_dtype: template(), key_width: template()):
    """Inverse twiddle in place on ``keys[0:n]`` (masks selected from the *output* sign bit —
    the exact inverse of :func:`_cpu_twiddle`)."""
    loop_config(serialize=True)
    for i in range(n):
        if static(key_width == 32):
            v = bit_cast(keys[i], u32)
            if static(key_dtype == i32):
                w = v ^ u32(0x80000000)
            elif static(key_dtype == f32):
                w = v ^ u32(0xFFFFFFFF)
                if (v & u32(0x80000000)) != u32(0):
                    w = v ^ u32(0x80000000)
            else:  # u32: identity
                w = v
            keys[i] = bit_cast(w, key_dtype)
        else:  # 64-bit
            v = bit_cast(keys[i], u64)
            if static(key_dtype == i64):
                w = v ^ u64(0x8000000000000000)
            elif static(key_dtype == f64):
                w = v ^ u64(0xFFFFFFFFFFFFFFFF)
                if (v & u64(0x8000000000000000)) != u64(0):
                    w = v ^ u64(0x8000000000000000)
            else:  # u64: identity
                w = v
            keys[i] = bit_cast(w, key_dtype)


@_func
def _cpu_radix_pass(src: template(), dst: template(), vsrc: template(), vdst: template(),
                    scratch: template(), n: template(), shift: i32, has_values: template(),
                    key_width: template()):
    """One stable LSD byte pass of the serial radix sort: counting sort of ``src[0:n]`` into
    ``dst[0:n]`` by digit ``(pattern >> shift) & 0xFF`` (values move in lock-step).

    ``pattern`` is the key's monotone-unsigned bit pattern (the caller has already twiddled).
    ``scratch[0:256]`` holds the digit counters: zeroed, histogrammed, exclusive-scanned into
    per-digit base offsets, then consumed by an ascending-``i`` scatter — which is what makes
    the pass stable. Scratch dtype is ``u32`` regardless of key width (counts are ``<= n``).
    """
    # `loop_config(serialize=True)` directives the NEXT loop only, so every loop of the pass
    # carries its own: without these, the CPU backend parallelizes the remaining top-level
    # range-fors across its thread pool and the counting sort races (intermittent corruption).
    loop_config(serialize=True)
    for d in range(256):
        scratch[d] = u32(0)
    loop_config(serialize=True)
    for i in range(n):
        if static(key_width == 32):
            d = i32((bit_cast(src[i], u32) >> u32(shift)) & u32(255))
        else:
            d = i32((bit_cast(src[i], u64) >> u64(shift)) & u64(255))
        scratch[d] = scratch[d] + u32(1)
    loop_config(serialize=True)
    run = u32(0)
    for d in range(256):
        b = scratch[d]
        scratch[d] = run
        run = run + b
    loop_config(serialize=True)
    for i in range(n):
        if static(key_width == 32):
            d = i32((bit_cast(src[i], u32) >> u32(shift)) & u32(255))
        else:
            d = i32((bit_cast(src[i], u64) >> u64(shift)) & u64(255))
        t = bit_cast(scratch[d], i32)
        scratch[d] = scratch[d] + u32(1)
        dst[t] = src[i]
        if static(has_values):
            vdst[t] = vsrc[i]


@_func
def _cpu_seg_radix_pass(src: template(), dst: template(), vsrc: template(), vdst: template(),
                        scratch: template(), lo: i32, hi: i32, shift: i32, has_values: template(),
                        key_width: template()):
    """Slice-bounded sibling of :func:`_cpu_radix_pass`: counting sort of ``src[lo:hi]`` into
    ``dst[lo:hi]`` (used by the segmented sort, one call per digit pass per segment)."""
    loop_config(serialize=True)
    for d in range(256):
        scratch[d] = u32(0)
    loop_config(serialize=True)
    for i in range(lo, hi):
        if static(key_width == 32):
            d = i32((bit_cast(src[i], u32) >> u32(shift)) & u32(255))
        else:
            d = i32((bit_cast(src[i], u64) >> u64(shift)) & u64(255))
        scratch[d] = scratch[d] + u32(1)
    loop_config(serialize=True)
    run = u32(0)
    for d in range(256):
        b = scratch[d]
        scratch[d] = run
        run = run + b
    loop_config(serialize=True)
    for i in range(lo, hi):
        if static(key_width == 32):
            d = i32((bit_cast(src[i], u32) >> u32(shift)) & u32(255))
        else:
            d = i32((bit_cast(src[i], u64) >> u64(shift)) & u64(255))
        t = lo + bit_cast(scratch[d], i32)  # bases are segment-relative; offset back into [lo, hi)
        scratch[d] = scratch[d] + u32(1)
        dst[t] = src[i]
        if static(has_values):
            vdst[t] = vsrc[i]


@_func
def _cpu_radix_sort(keys: template(), tmp_keys: template(), values: template(), tmp_values: template(),
                    scratch: template(), n: template(), has_values: template(), end_bit: template(),
                    key_dtype: template(), key_width: template()):
    """Serial LSD radix sort, in place on ``keys``/``values`` (``tmp_*`` = ping-pong working copies).

    This is the CPU-backend counterpart of the GPU staircase in ``_radix_sort.py`` with the **same
    externally visible semantics**: keys are twiddled to monotone-unsigned bit patterns, then
    ``end_bit // 8`` stable byte passes run LSB-first over the patterns, ping-ponging
    ``keys <-> tmp_keys`` (values in lock-step); an even pass count lands the result back in
    ``keys``. Bits at/above ``end_bit`` are never examined (stable input order among ties),
    matching the GPU ``end_bit`` contract. ``scratch[0:256]`` is the per-pass digit-counter
    array — inside the ``sort_scratch_slots`` footprint (>= 257 for every ``n >= 1``), so the CPU
    path honours the same caller-owned scratch contract as the GPU path. O(4n) at 32-bit keys.

    Why serial radix and not the GPU staircase: the latter is block-tier (``SharedArray`` + block
    barriers + subgroup collectives), which the CPU backend cannot lower at any layer — barrier
    ops raise at trace time, the subgroup size reads 0, and a shared alloca lowers to an
    addrspace(3) LLVM global with no CPU backing, failing kernel JIT (see the algorithms ledger
    entry for the full evidence chain). The serial form avoids every block-tier construct.
    """
    num_passes = static(end_bit // 8)
    if static(isinstance(n, int)):
        nc = n
    elif static(isinstance(n, AnyArray)):
        nc = n[()]
    else:
        nc = n
    _cpu_twiddle(keys, nc, key_dtype, key_width)
    # Byte passes, LSB first — same ping-pong parity as the GPU staircase (_emit_pass): pass 0
    # moves keys -> tmp_keys, so an even pass count (the frozen contract) lands the result back
    # in ``keys``; an odd one lands it in ``tmp_keys`` and the finalize copies it over.
    for p in static(range(num_passes)):
        if static(p % 2 == 0):
            _cpu_radix_pass(keys, tmp_keys, values, tmp_values, scratch, nc, p * 8, has_values, key_width)
        else:
            _cpu_radix_pass(tmp_keys, keys, tmp_values, values, scratch, nc, p * 8, has_values, key_width)
    if static(num_passes % 2 == 0):
        _cpu_untwiddle(keys, nc, key_dtype, key_width)
    else:
        # Odd pass count (not reachable under the frozen even-pass contract; kept correct anyway):
        # patterns sit in tmp_keys (values in tmp_values), so untwiddle them back into place.
        _cpu_untwiddle(tmp_keys, nc, key_dtype, key_width)
        if static(has_values):
            loop_config(serialize=True)
            for i in range(nc):
                values[i] = tmp_values[i]
        loop_config(serialize=True)
        for i in range(nc):
            keys[i] = tmp_keys[i]


@_func
def _cpu_seg_radix_sort(keys: template(), tmp_keys: template(), values: template(), tmp_values: template(),
                        scratch: template(), segment_ids: template(), n: template(), num_segments: template(),
                        has_values: template(), end_bit: template(), key_dtype: template(),
                        key_width: template()):
    """Serial per-segment LSD radix sort — CPU counterpart of ``sort_segmented``.

    Segments are the contiguous ranges ``[segment_ids[j], segment_ids[j+1])`` for ``j <
    num_segments`` — the caller supplies ``num_segments + 1`` start offsets, the last being the
    end sentinel (conventionally ``n``; this is the warp ``segmented_sort_pairs`` / CUB array
    shape). Each segment is sorted by key with the same twiddled stable byte passes as
    :func:`_cpu_radix_sort` (values ride along), and segments keep their order. Empty segments
    (``ids[j] == ids[j+1]``) are no-ops. The even pass count lands every segment back in place in
    ``keys``, so no segment bookkeeping survives the passes. Serial over segments: the segment
    loop must carry its own ``serialize`` directive like every other loop here.
    """
    num_passes = static(end_bit // 8)
    if static(isinstance(n, int)):
        nc = n
    elif static(isinstance(n, AnyArray)):
        nc = n[()]
    else:
        nc = n
    if static(isinstance(num_segments, int)):
        nseg = num_segments
    elif static(isinstance(num_segments, AnyArray)):
        nseg = num_segments[()]
    else:
        nseg = num_segments
    _cpu_twiddle(keys, nc, key_dtype, key_width)
    loop_config(serialize=True)
    for j in range(nseg):
        lo = segment_ids[j]
        hi = segment_ids[j + 1]  # sentinel form: ids has num_segments + 1 entries
        for p in static(range(num_passes)):
            if static(p % 2 == 0):
                _cpu_seg_radix_pass(keys, tmp_keys, values, tmp_values, scratch, lo, hi, p * 8, has_values, key_width)
            else:
                _cpu_seg_radix_pass(tmp_keys, keys, tmp_values, values, scratch, lo, hi, p * 8, has_values, key_width)
    _cpu_untwiddle(keys, nc, key_dtype, key_width)


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
def _cpu_exclusive_scan_minmax(arr: template(), out: template(), n: i32, identity, is_min: template()):
    """Serial exclusive min/max scan; ``identity`` is the compile-time identity constant
    (``_typed_min_identity`` / ``_typed_max_identity`` for the dtype, mirrored from the GPU
    staircase)."""
    loop_config(serialize=True)
    if n > 0:
        run = identity
        for i in range(n):
            out[i] = run
            if is_min:
                run = min(run, arr[i])
            else:
                run = max(run, arr[i])


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
