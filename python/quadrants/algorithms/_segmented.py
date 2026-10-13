# type: ignore
"""Segmented key-value sort (``qd.algorithms.sort_segmented``) — C4 (mujocoqd/algorithms-2).

Sorts every segment of a key/value array independently by key (stable). Segments are the
contiguous ranges ``[segment_ids[j], segment_ids[j+1])`` for ``j < num_segments``; the caller
supplies ``num_segments + 1`` non-decreasing i32 start offsets with ``segment_ids[0] == 0`` and
the end sentinel ``segment_ids[num_segments] == n`` (the exact array shape warp's
``utils.segmented_sort_pairs`` / CUB ``DeviceSegmentedSort`` consume — the mujoco SAP broadphase
already builds it). Empty segments (``ids[j] == ids[j+1]``) are no-ops, and segments keep their
order. Start offsets (rather than per-element segment ids) are what make empty segments
expressible.

**Why not the "segment id << 32 | key" composite key** (the original sketch in the requirements):
packging two 32-bit halves needs a 64-bit key, and Metal has no 64-bit keys / atomics — the
composite is unimplementable on the short-term target backend. The implementation below needs no
wide composite: it is a lexicographic LSD radix sort over the pair (run-rank, key), where the
"run-rank" — the index of the run of segments containing each element — is derived once with the
same flags + exclusive-scan machinery ``select`` uses, and byte-digit passes run over the *key*
first (minor) and the *rank* second (major). Stable byte passes make that exactly "per-segment
stable key sort, segments in order", on 32-bit digits only, with the same twiddle / ``end_bit``
/ dtype semantics as :func:`qd.algorithms.sort`.

GPU phases (one ``@qd.func`` chain, fixed launch topology for a given compile-time
``(end_bit, log256_max_n)``):

1. ``_seg_zero_flags`` + ``_seg_mark_starts``: 1 at every segment start in ``tmp_seg``.
2. ``_emit_select_scan``: exclusive scan of the flags -> run-rank in ``scratch[0:n]`` (u32),
   scan partials stacked above ``n`` (reuses ``select``'s staircase verbatim).
3. key digit passes (``end_bit // 8``, even per the frozen ``sort`` contract), then rank digit
   passes (``log256_max_n`` — a rank is ``< n <= 256 ** D`` so ``D`` bytes always cover it).
   Every pass moves keys, values AND rank in lock-step (the rank ping-pongs
   ``scratch[0:n] <-> tmp_seg``); key passes histogram the key, rank passes the rank.
4. parity finalize: an even total pass count lands the result in ``keys``/``values`` (twiddle
   undone in place); an odd count adds one copy-back phase.

Both digit-pass families share one hist/scatter pair parameterised by a compile-time ``sel``
flag (which array supplies the digit); the tile-histogram region sits at ``scratch[n:]`` —
above the live rank — because the histogram and the rank coexist in scratch between passes.

CPU path: ``_cpu_paths._cpu_seg_radix_sort`` — per-segment serial byte passes, no rank needed.

**Scratch**: caller-owned ``u32``, :func:`sort_segmented_scratch_slots` ``(n, log256_max_n)``
slots (= ``n`` rank slots + the sort histogram/staircase footprint; the rank scan's partials
reuse the same space sequentially). No on-device sufficiency check (graph-capture contract).
"""

from quadrants.lang.impl import current_cfg, static
from quadrants.lang.kernel_impl import func as _func
from quadrants.lang.misc import cpu as _arch_cpu, loop_config
from quadrants.lang.ops import atomic_add, bit_cast
from quadrants.lang.simt import block as _block
from quadrants.types.annotations import template
from quadrants.types.primitive_types import i32, u32, u64

from ._cpu_paths import _cpu_seg_radix_sort
from ._radix_sort import (
    RADIX_BITS,
    RADIX_DIGITS,
    _key_width_bits,
    _radix_twiddle,
    _TWIDDLE_KEY_DTYPES,
)
from ._reduce import BLOCK_DIM, _validate_log256_max_n
from ._scan import _emit_exclusive_scan_add
from ._select import _emit_select_scan


@_func
def _seg_zero_flags(flags: template(), n: i32):
    """Zero ``flags[0:n]`` (i32). Its own offloaded launch: the mark phase's scattered writes
    must not race these fills (they hit the same slots)."""
    loop_config(block_dim=BLOCK_DIM)
    for i in range(n):
        flags[i] = 0


@_func
def _seg_mark_starts(segment_ids: template(), flags: template(), num_segments: i32, n: i32):
    """Mark every segment start: ``flags[segment_ids[j]] = 1`` for ``j < num_segments`` (writes
    past ``n`` are skipped so an ``ids[j] == n`` end-sentinel is harmless). Distinct ``j`` write
    distinct slots (ids are non-decreasing), so no write-write race beyond duplicated 1s."""
    loop_config(block_dim=BLOCK_DIM)
    for j in range(num_segments):
        s = segment_ids[j]
        if s < n:
            flags[s] = 1


@_func
def _seg_rank_fixup(scratch: template(), flags: template(), n: i32):
    """Turn the exclusive scan of the start flags into run ranks in place:
    ``rank[i] = excl[i] + flags[i] - 1`` (== the number of segment starts at or before ``i``,
    minus one — correct because ``ids[0] == 0`` guarantees at least one start at or before every
    element). The raw exclusive scan counts run 0's own start against its non-first elements,
    which shifts every rank by +1 (observed as the whole array rotated one slot)."""
    loop_config(block_dim=BLOCK_DIM)
    for i in range(n):
        scratch[i] = scratch[i] + bit_cast(flags[i], u32) - u32(1)


@_func
def _seg_hist(src: template(), scratch: template(), n: i32, num_blocks: i32, base: i32,
              bit_start: i32, sel: template(), key_width: template()):
    """Per-block digit histogram into ``scratch[base : base + num_blocks * 256]`` (digit-major),
    ``sel == 1`` extracting the digit from the key (``key_width`` bits) and ``sel == 0`` from the
    run-rank (u32 patterns, possibly stored through an i32 buffer — read via ``bit_cast``).

    Same body contract as ``_radix_hist``: the extra all-ones "dump" slot keeps the shared-memory
    ``atomic_add`` in uniform control flow (see the rationale there), and out-of-range lanes take
    digit 256 (the dump slot) so they never pollute the histogram.
    """
    loop_config(block_dim=BLOCK_DIM)
    total_threads = num_blocks * BLOCK_DIM
    for i in range(total_threads):
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        hist = _block.SharedArray((RADIX_DIGITS + 1,), i32)
        hist[tid] = i32(0)
        if tid == 0:
            hist[RADIX_DIGITS] = i32(0)
        _block.sync()
        digit = i32(RADIX_DIGITS)  # dump slot for out-of-range lanes (unconditional first assignment)
        if i < n:
            if static(sel == 1):
                if static(key_width == 32):
                    k32 = bit_cast(src[i], u32)
                    digit = i32((k32 >> u32(bit_start)) & u32(RADIX_DIGITS - 1))
                else:
                    k64 = bit_cast(src[i], u64)
                    digit = i32((k64 >> u64(bit_start)) & u64(RADIX_DIGITS - 1))
            else:
                r32 = bit_cast(src[i], u32)
                digit = i32((r32 >> u32(bit_start)) & u32(RADIX_DIGITS - 1))
        atomic_add(hist[digit], i32(1))
        _block.sync()
        scratch[base + tid * num_blocks + block_id] = bit_cast(hist[tid], u32)


@_func
def _seg_scatter(keys_in: template(), keys_out: template(), values_in: template(), values_out: template(),
                 rank_in: template(), rank_out: template(), scratch: template(), n: i32, num_blocks: i32,
                 base: i32, bit_start: i32, key_dtype: template(), has_values: template(),
                 key_width: template(), sel: template()):
    """Per-block radix rank + scatter moving keys, values AND rank in lock-step. ``sel == 1``
    ranks by the key digit (the minor key passes), ``sel == 0`` by the rank digit (the major
    passes) — both consume an 8-bit digit via ``block.radix_rank_match_atomic_or``, reading the
    per-(digit, block) global offset from ``scratch[base + ...]``."""
    loop_config(block_dim=BLOCK_DIM)
    total_threads = num_blocks * BLOCK_DIM
    for i in range(total_threads):
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        bins = _block.SharedArray((RADIX_DIGITS,), i32)
        excl_prefix = _block.SharedArray((RADIX_DIGITS,), i32)
        block_offsets = _block.SharedArray((RADIX_DIGITS,), i32)
        if static(sel == 1):
            if static(key_width == 32):
                key = u32(0xFFFFFFFF)
                if i < n:
                    key = bit_cast(keys_in[i], u32)
                rank = _block.radix_rank_match_atomic_or(key, BLOCK_DIM, RADIX_BITS, bit_start, RADIX_BITS, bins, excl_prefix)
                digit = i32((key >> u32(bit_start)) & u32(RADIX_DIGITS - 1))
            else:
                # 64-bit keys: extract the digit from the full-width key, then hand the
                # pre-shifted 8-bit digit to the u32 rank helper with bit_start=0
                # (mirrors the key_width branch in _seg_hist; the helper's key contract
                # is u32, see block.radix_rank_match_atomic_or).
                k64 = u64(0xFFFFFFFF)
                if i < n:
                    k64 = bit_cast(keys_in[i], u64)
                digit = i32((k64 >> u64(bit_start)) & u64(RADIX_DIGITS - 1))
                rank = _block.radix_rank_match_atomic_or(u32(digit), BLOCK_DIM, RADIX_BITS, i32(0), RADIX_BITS, bins, excl_prefix)
        else:
            rpat = u32(0xFFFFFFFF)
            if i < n:
                rpat = bit_cast(rank_in[i], u32)
            rank = _block.radix_rank_match_atomic_or(rpat, BLOCK_DIM, RADIX_BITS, bit_start, RADIX_BITS, bins, excl_prefix)
            digit = i32((rpat >> u32(bit_start)) & u32(RADIX_DIGITS - 1))
        if tid < RADIX_DIGITS:
            global_off = bit_cast(scratch[base + tid * num_blocks + block_id], i32)
            block_offsets[tid] = global_off - excl_prefix[tid]
        _block.sync()
        if i < n:
            dst = block_offsets[digit] + rank
            keys_out[dst] = keys_in[i]
            if static(has_values):
                values_out[dst] = values_in[i]
            rank_out[dst] = rank_in[i]


def _emit_seg_pass(keys, tmp_keys, values, tmp_values, rank_a, rank_b, scratch, count, num_blocks,
                   hist_base, hist_len, p, key_dtype, has_values, key_width, log256_max_n, sel):
    """Emit one segmented digit pass at kernel-compile time (histogram -> scan -> scatter).

    ``p`` is a Python int; all ping-pong selections are made here in plain Python and the field
    handles go straight into the ``@qd.func`` calls. ``rank_a`` / ``rank_b`` are the two rank
    buffers (``scratch[0:n]`` slice owner and ``tmp_seg``); every pass moves the rank in lock-step
    so its ping-pong parity stays coherent across the key and rank pass families. The histogram
    digit source follows ``sel`` (1 = key, 0 = rank); the histogram/scan region lives above the
    live rank at ``hist_base``.
    """
    bit_start = p * RADIX_BITS
    src = keys if (p % 2 == 0) else tmp_keys
    dst = tmp_keys if (p % 2 == 0) else keys
    vsrc = values if (p % 2 == 0) else tmp_values
    vdst = tmp_values if (p % 2 == 0) else values
    rsrc = rank_a if (p % 2 == 0) else rank_b
    rdst = rank_b if (p % 2 == 0) else rank_a
    digit_src = src if (sel == 1) else rsrc
    _seg_hist(digit_src, scratch, count, num_blocks, hist_base, bit_start, sel, key_width)
    _emit_exclusive_scan_add(scratch, hist_base, hist_len, log256_max_n - 1)
    _seg_scatter(src, dst, vsrc, vdst, rsrc, rdst, scratch, count, num_blocks, hist_base, bit_start,
                 key_dtype, has_values, key_width, sel)


def sort_segmented_scratch_slots(n, log256_max_n: int = None):
    """Minimum u32 scratch slots :func:`sort_segmented` needs for a length-``n`` input.

    Layout: ``scratch[0:n]`` holds the run-rank (and rank ping-pong half), the sort's tile
    histograms + scan staircase live above it, and the rank scan's partials reuse that same
    region sequentially (phases never overlap), so the total is ``n + sort_scratch_slots(n,
    log256_max_n)``. Size against the provisioned upper bound on the count, exactly like
    :func:`qd.algorithms.sort_scratch_slots`. Multiply by 4 for bytes.
    """
    from ._radix_sort import sort_scratch_slots

    return n + sort_scratch_slots(n, log256_max_n)


@_func(requires_top_level=True)
def sort_segmented(
    keys: template(),
    tmp_keys: template(),
    values: template(),
    tmp_values: template(),
    segment_ids: template(),
    tmp_seg: template(),
    scratch: template(),
    n: template(),
    num_segments: template(),
    key_dtype: template(),
    has_values: template(),
    end_bit: template(),
    log256_max_n: template(),
):
    """Segmented stable key-value sort: every contiguous segment (``segment_ids`` = start
    offsets, ``segment_ids[0] == 0``, non-decreasing, last ``<= n``) is sorted by key ascending
    with stable tie order, values riding along; segments keep their order. Empty segments
    (``ids[j] == ids[j+1]``) are no-ops.

    Call at the **top level** of your own ``@qd.kernel`` — same composition contract as
    :func:`qd.algorithms.sort` (never nest in ordinary runtime ``for`` / ``if`` / ``while``).
    Semantics per key dtype / ``end_bit`` / twiddle / NaN ordering match ``sort`` exactly
    (``end_bit`` must be a positive even multiple of 8, ``<=`` key width; key dtype one of
    ``{u32, i32, f32, u64, i64, f64}`` — 64-bit not on Metal). ``values`` / ``tmp_values``
    follow the ``sort`` placeholder convention when ``has_values=False``.

    Args:
        keys / tmp_keys: 1-D key tensor + same-shape ping-pong work buffer (distinct buffers);
            sorted in place on return.
        values / tmp_values: key-value payload (any scalar dtype) or placeholders.
        segment_ids: 1-D ``i32`` start-offset array with ``num_segments + 1`` entries
            (``segment_ids[0] == 0``, non-decreasing, ``segment_ids[num_segments] == n``).
        tmp_seg: 1-D ``i32`` work buffer, shape ``>= n``; garbage on return (it carries the
            start flags and one ping-pong half of the run-rank).
        scratch: caller-owned ``u32``, :func:`sort_segmented_scratch_slots` ``(capacity_n,
            log256_max_n)`` slots; no on-device sufficiency check.
        n / num_segments: 0-d ``i32`` ndarrays (device-resident counts), read once as ``x[()]``.
        key_dtype / has_values / end_bit / log256_max_n: compile-time, as for ``sort``.

    On the CPU backend this dispatches to the serial per-segment radix
    (:func:`_cpu_paths._cpu_seg_radix_sort`) with the same external semantics.
    """
    if static(current_cfg().arch == _arch_cpu):
        key_width = static(_key_width_bits(key_dtype))
        _cpu_seg_radix_sort(keys, tmp_keys, values, tmp_values, scratch, segment_ids, n, num_segments,
                            has_values, end_bit, key_dtype, key_width)
    else:
        _validate_log256_max_n(log256_max_n)
        key_width = static(_key_width_bits(key_dtype))
        num_key_passes = static(end_bit // RADIX_BITS)
        needs_twiddle = static(key_dtype in _TWIDDLE_KEY_DTYPES)
        total_passes = static(num_key_passes + log256_max_n)
        count = n[()]
        nseg = num_segments[()]
        num_blocks = (count + (BLOCK_DIM - 1)) // BLOCK_DIM
        hist_base = count  # rank lives in scratch[0:count]; tile histograms start above it
        hist_len = num_blocks * RADIX_DIGITS
        if static(needs_twiddle):
            _radix_twiddle(keys, count, key_dtype, key_width, True)
        _seg_zero_flags(tmp_seg, count)
        _seg_mark_starts(segment_ids, tmp_seg, nseg, count)
        _emit_select_scan(tmp_seg, scratch, count, log256_max_n)  # exclusive scan of flags
        _seg_rank_fixup(scratch, tmp_seg, count)  # -> run-rank in scratch[0:count]
        for p in static(range(num_key_passes)):
            _emit_seg_pass(keys, tmp_keys, values, tmp_values, scratch, tmp_seg, scratch, count, num_blocks,
                           hist_base, hist_len, p, key_dtype, has_values, key_width, log256_max_n, 1)
        for p in static(range(log256_max_n)):
            _emit_seg_pass(keys, tmp_keys, values, tmp_values, scratch, tmp_seg, scratch, count, num_blocks,
                           hist_base, hist_len, p + num_key_passes, key_dtype, has_values, key_width,
                           log256_max_n, 0)
        if static(needs_twiddle):
            if static(total_passes % 2 == 0):
                _radix_twiddle(keys, count, key_dtype, key_width, False)
            else:
                _radix_twiddle(tmp_keys, count, key_dtype, key_width, False)
        if static(total_passes % 2 != 0):
            # odd pass total: results sit in tmp_*; one copy-back phase restores the in-place contract
            for i in range(count):
                keys[i] = tmp_keys[i]
                if static(has_values):
                    values[i] = tmp_values[i]


__all__ = [
    "sort_segmented",
    "sort_segmented_scratch_slots",
]
