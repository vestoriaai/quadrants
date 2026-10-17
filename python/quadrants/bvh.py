"""qd-native generic AABB BVH (the wp.Bvh/wp.Mesh-query counterpart).

Pure qd kernels + qd arrays: zero native, zero codegen changes. Algorithm-level
translation of warp's native LBVH (bvh.cu) and traversal skeleton (bvh.h):
compute_morton_codes / build_leaves / build_hierarchy / mark_packed_leaf_nodes /
bvh_refit_kernel and morton3<1024> / bvh_query_next_impl / bvh_query_test.

Deviations vs warp (qd-frontend necessity, all pinned by
_probe/bvh_local_array_probe.py):

- warp hides the tree behind a uint64 id (device BVH struct); qd.func cannot
  take struct/pointer args, so every kernel-level entry point takes the tree
  arrays explicitly. The host ``Bvh`` handle bundles them.
- warp keeps iterator state (stack + counters) inside a bvh_query_t passed by
  reference; qd structs cannot hold ndarrays and in-kernel fixed-length local
  arrays are unavailable, so state lives in caller-owned 2-D ndarray rows:
  ``stack[slot, depth]`` and ``state[slot, k] = [count, prim_cur, prim_end,
  result_prim]``.
- BVHPackedNodeHalf packs {x, y, z, bitfield(i:31|b:1)}; the bitfield is stored
  as its raw u32 bit pattern reinterpreted into the vec4 w channel via
  ``bit_cast`` (byte-identical to warp; payload roundtrip probe-verified).
- warp sorts (u64 group<<32 | morton) keys with CUB radix sort; groups are not
  implemented here (single group, upper key bits reserved 0), so the key is the
  plain 30-bit morton code in an i32 array. The sort is the Batcher odd-even
  merge network re-stated with ndarray args (the runtime's composable radix
  sort is GPU-only on the current wheel and ``_kernels.sort_stage`` is
  field-only); its tie order is network-dependent, not stable - LBVH is valid
  for any tie permutation and query result sets are tree-shape independent.
- build_hierarchy / refit node-completion chains run serialized (one loop
  iteration at a time): the chains hand work across iterations, so a serial
  order is the deterministic-correct schedule. Build is a one-off host call;
  per-phase cost stays O(n).

Traversal: three unrolled query funcs (qd has no templates), all manual
single-exit with scalar ``-> i32`` returns (runs on runtimes without the
compound early-return slots):

    bvh_query_init(state, stack, slot, root)          # push the root
    found = bvh_query_aabb_next(node_lowers, node_uppers, primitive_indices,
                                item_lowers, item_uppers, stack, state, slot,
                                q_lower, q_upper)
    while found == 1:
        idx = state[slot, 3]                          # the primitive hit
        ...
        found = bvh_query_aabb_next(...)              # advance

``bvh_query_ray_next`` takes (start, dir, max_dist) with warp's half-open
``t < max_dist`` bound; ``bvh_query_sphere_next`` takes (center, radius). The
``*_batch`` kernels are usage references that collect up to ``max_hits``
candidates per query slot.

Requires ``qd.init`` before ``Bvh`` construction. Items come in as caller-owned
vec3 ndarrays (``lower``/``upper``); triangle/AABB extraction stays with the
consumer (mujoco_qd mesh side), matching warp's generic-BVH split.
"""

import numpy as np

import quadrants as qd
from quadrants.lang.misc import loop_config
from quadrants.lang.ops import atomic_add, bit_cast
from quadrants.types.primitive_types import f32, i32, u32

# traversal stack depth; mark_packed_leaf mutes nodes deeper than this so a
# query can never overflow the stack (warp uses 32; 64 per the spec)
_STACK_DEPTH = 64

# warp uses FLT_MAX; qd has no inf constant - far larger than any real slab t
_INF_F32 = 1.0e30

_f32_1d = qd.types.ndarray(dtype=qd.f32, ndim=1)
_vec3_1d = qd.types.ndarray(dtype=qd.math.vec3, ndim=1)
_vec4_1d = qd.types.ndarray(dtype=qd.math.vec4, ndim=1)
_i32_1d = qd.types.ndarray(dtype=qd.i32, ndim=1)
_i32_2d = qd.types.ndarray(dtype=qd.i32, ndim=2)
_vec3_t = qd.math.vec3
_vec4_t = qd.math.vec4


# ---------------------------------------------------------------- morton3 ----

@qd.func
def _bvh_part1by2(n: u32) -> u32:
    """Spread the 10 low bits of ``n`` with two zeros between them (bvh.h:277)."""
    m = n
    m = (m ^ (m << 16)) & u32(0xFF0000FF)
    m = (m ^ (m << 8)) & u32(0x0300F00F)
    m = (m ^ (m << 4)) & u32(0x030C30C3)
    m = (m ^ (m << 2)) & u32(0x09249249)
    return m


@qd.func
def _bvh_morton3(x: f32, y: f32, z: f32) -> u32:
    """10 bits per axis interleaved (bvh.h:288, morton3<1024>)."""
    ux = i32(qd.math.clamp(x * 1024.0, 0.0, 1023.0))
    uy = i32(qd.math.clamp(y * 1024.0, 0.0, 1023.0))
    uz = i32(qd.math.clamp(z * 1024.0, 0.0, 1023.0))
    return (_bvh_part1by2(u32(uz)) << 2) | (_bvh_part1by2(u32(uy)) << 1) | _bvh_part1by2(u32(ux))


@qd.func
def _bvh_clz32(x: i32) -> i32:
    """Leading zeros of a non-negative i32; 32 for 0 (delta sentinel, bvh.cu:218)."""
    v = x
    c = 0
    while v != 0:
        v = v >> 1
        c = c + 1
    return 32 - c


# ------------------------------------------------------- packed node bits ----

@qd.func
def _bvh_node_i(n: _vec4_t) -> i32:
    """The 31-bit payload of a packed node's w channel (child index / prim range)."""
    return i32(bit_cast(n[3], u32) & u32(0x7FFFFFFF))


@qd.func
def _bvh_is_leaf(n: _vec4_t) -> i32:
    """The 1-bit leaf flag in bit31 of a packed node's w channel."""
    return i32(bit_cast(n[3], u32) >> u32(31))


# -------------------------------------------------------- query predicates ---

@qd.func
def _bvh_aabb_overlap(al: _vec3_t, au: _vec3_t, bl: _vec3_t, bu: _vec3_t) -> i32:
    """Closed-interval AABB/AABB overlap (warp intersect_aabb_aabb)."""
    hit = 1
    if (al[0] > bu[0]) or (al[1] > bu[1]) or (al[2] > bu[2]) \
            or (au[0] < bl[0]) or (au[1] < bl[1]) or (au[2] < bl[2]):
        hit = 0
    return hit


@qd.func
def _bvh_ray_aabb_t(pos: _vec3_t, rcp: _vec3_t, lo: _vec3_t, hi: _vec3_t) -> f32:
    """Ray slab test (warp intersect_ray_aabb); returns t, or _INF_F32 on miss.

    warp keeps the hit boolean separately with a half-open max_dist bound; the
    sentinel folds both into the scalar return (a real hit can never reach
    _INF_F32).
    """
    lmin = -_INF_F32
    lmax = _INF_F32

    l1 = (lo[0] - pos[0]) * rcp[0]
    l2 = (hi[0] - pos[0]) * rcp[0]
    lmin = qd.max(lmin, qd.min(l1, l2))
    lmax = qd.min(lmax, qd.max(l1, l2))

    l1 = (lo[1] - pos[1]) * rcp[1]
    l2 = (hi[1] - pos[1]) * rcp[1]
    lmin = qd.max(lmin, qd.min(l1, l2))
    lmax = qd.min(lmax, qd.max(l1, l2))

    l1 = (lo[2] - pos[2]) * rcp[2]
    l2 = (hi[2] - pos[2]) * rcp[2]
    lmin = qd.max(lmin, qd.min(l1, l2))
    lmax = qd.min(lmax, qd.max(l1, l2))

    t = _INF_F32
    if (lmax >= 0.0) and (lmax >= lmin):
        t = lmin
    return t


@qd.func
def _bvh_sphere_aabb_hit(c: _vec3_t, radius_sq: f32, lo: _vec3_t, hi: _vec3_t) -> i32:
    """Exact sphere/AABB overlap (warp intersect_sphere_aabb, RTCD branchless)."""
    dx = qd.max(qd.max(lo[0] - c[0], c[0] - hi[0]), 0.0)
    dy = qd.max(qd.max(lo[1] - c[1], c[1] - hi[1]), 0.0)
    dz = qd.max(qd.max(lo[2] - c[2], c[2] - hi[2]), 0.0)
    hit = 0
    if dx * dx + dy * dy + dz * dz <= radius_sq:
        hit = 1
    return hit


# ----------------------------------------------------------- build kernels ---

@qd.kernel
def _bvh_morton(
    item_lowers: _vec3_1d,
    item_uppers: _vec3_1d,
    grid_lower: _vec3_1d,
    grid_inv_edges: _vec3_1d,
    keys: _i32_1d,
    indices: _i32_1d,
    n: int,
):
    """warp compute_morton_codes (bvh.cu:184): centroid -> 30-bit morton key."""
    for i in range(n):
        lo = item_lowers[i]
        up = item_uppers[i]
        center = (lo + up) * 0.5
        local = (center - grid_lower[0]) * grid_inv_edges[0]
        keys[i] = i32(_bvh_morton3(local[0], local[1], local[2]))
        indices[i] = i


@qd.kernel
def _bvh_sort_stage(keys: _i32_1d, values: _i32_1d, n: int, p: int, k: int, invocations: int):
    """One Batcher odd-even merge stage over (keys, values) pairs.

    1:1 restatement of ``quadrants._kernels.sort_stage`` with ndarray args (the
    original is field-only); the host drives the canonical ``parallel_sort``
    schedule with a sync between stages.
    """
    for inv in range(invocations):
        j = k % p + inv * 2 * k
        for i in range(0, qd.min(k, n - j - k)):
            a = i + j
            b = i + j + k
            if int(a / (p * 2)) == int(b / (p * 2)):
                key_a = keys[a]
                key_b = keys[b]
                if key_a > key_b:
                    keys[a] = key_b
                    keys[b] = key_a
                    tmp = values[a]
                    values[a] = values[b]
                    values[b] = tmp


@qd.kernel
def _bvh_build_rest(
    keys: _i32_1d,
    indices: _i32_1d,
    item_lowers: _vec3_1d,
    item_uppers: _vec3_1d,
    node_lowers: _vec4_1d,
    node_uppers: _vec4_1d,
    node_parents: _i32_1d,
    range_lefts: _i32_1d,
    range_rights: _i32_1d,
    node_counts: _i32_1d,
    deltas: _i32_1d,
    root: _i32_1d,
    n: int,
    leaf_size: int,
):
    """Everything after the sort: deltas, leaves, counter reset, hierarchy, packing.

    warp compute_key_deltas (bvh.cu:218) + build_leaves (bvh.cu:228) +
    build_hierarchy (bvh.cu:261) + mark_packed_leaf_nodes (bvh.cu:402).
    The hierarchy chains hand completions across loop iterations, so that one
    phase runs serialized; the other phases are race-free and parallel.
    """
    # -- key deltas (sorted keys in; adjacent xor leading zeros)
    for i in range(n - 1):
        diff = keys[i] ^ keys[i + 1]
        deltas[i] = _bvh_clz32(diff)

    # -- leaf nodes [0, n): bounds + [start, end) range + leaf key ranges
    for i in range(n):
        item = indices[i]
        lo = item_lowers[item]
        up = item_uppers[item]
        w_lo = bit_cast(u32(i), f32)
        w_hi = bit_cast(u32(i) | (u32(1) << u32(31)), f32)
        node_lowers[i] = qd.math.vec4(lo[0], lo[1], lo[2], w_hi)
        node_uppers[i] = qd.math.vec4(up[0], up[1], up[2], w_lo)
        range_lefts[i] = i
        range_rights[i] = i

    # -- reset child counters (the build's atomic completion counters)
    max_nodes = 2 * n - 1
    for i in range(max_nodes):
        node_counts[i] = 0

    # -- bottom-up clustering; serialized for determinism (see module docstring)
    loop_config(serialize=True)
    for i in range(n):
        index = i
        doneqd = 0
        while doneqd == 0:
            left = range_lefts[index]
            right = range_rights[index]
            if left == 0 and right == n - 1:
                root[0] = index
                node_parents[index] = -1
                doneqd = 1
            else:
                # parent side: prefer the greater common prefix; parity
                # tie-break on sorted indices avoids ladders (bvh.cu:296)
                parent_right = 0
                if left == 0:
                    parent_right = 1
                else:
                    if (right != n - 1) and (deltas[right] >= deltas[left - 1]):
                        if deltas[right] == deltas[left - 1]:
                            parent_right = (indices[left - 1] % 2) ^ (indices[right] % 2)
                        else:
                            parent_right = 1
                parent = right + n
                if parent_right == 0:
                    parent = left + n - 1
                node_parents[index] = parent
                stage_w = bit_cast(u32(index), f32)
                stage4 = qd.math.vec4(0.0, 0.0, 0.0, stage_w)
                if parent_right == 1:
                    node_lowers[parent] = stage4
                    range_lefts[parent] = left
                else:
                    node_uppers[parent] = stage4
                    range_rights[parent] = right
                child_count = atomic_add(node_counts[parent], 1)
                if child_count == 1:
                    # second arrival: the parent is complete - union child bounds
                    bl = node_lowers[parent]
                    bu = node_uppers[parent]
                    lc = _bvh_node_i(bl)
                    rc = _bvh_node_i(bu)
                    lb = node_lowers[lc]
                    lu = node_uppers[lc]
                    rb = node_lowers[rc]
                    ru = node_uppers[rc]
                    nx = qd.min(lb[0], rb[0])
                    ny = qd.min(lb[1], rb[1])
                    nz = qd.min(lb[2], rb[2])
                    mx = qd.max(lu[0], ru[0])
                    my = qd.max(lu[1], ru[1])
                    mz = qd.max(lu[2], ru[2])
                    node_lowers[parent] = qd.math.vec4(nx, ny, nz, bl[3])
                    node_uppers[parent] = qd.math.vec4(mx, my, mz, bu[3])
                    index = parent
                else:
                    doneqd = 1

    # -- mark packed leaves: any node spanning <= leaf_size items (or deeper
    #    than the query stack) becomes a leaf; its children are muted forever
    for node_index in range(max_nodes):
        depth = 1
        parent = node_parents[node_index]
        while parent != -1:
            parent = node_parents[parent]
            depth = depth + 1
        left = range_lefts[node_index]
        right = range_rights[node_index] + 1
        mark = 0
        if right - left <= leaf_size:
            mark = 1
        if depth >= _STACK_DEPTH:
            mark = 1
        if mark == 1:
            bl = node_lowers[node_index]
            bu = node_uppers[node_index]
            w_lo = bit_cast(u32(left) | (u32(1) << u32(31)), f32)
            w_hi = bit_cast(u32(right), f32)
            node_lowers[node_index] = qd.math.vec4(bl[0], bl[1], bl[2], w_lo)
            node_uppers[node_index] = qd.math.vec4(bu[0], bu[1], bu[2], w_hi)


# -------------------------------------------------------- query iterator ----
# state[slot, 0..2] = [count, prim_cur, prim_end] (traversal cursor, mutated
# by the *_next funcs); state[slot, 3] = result primitive of the last hit.


@qd.func
def bvh_query_init(state: _i32_2d, stack: _i32_2d, slot: int, root: int):
    """Seed one query slot (warp bvh_query construction). root < 0 = empty tree."""
    count = 1
    if root < 0:
        count = 0
    stack[slot, 0] = root
    state[slot, 0] = count
    state[slot, 1] = 0
    state[slot, 2] = 0
    state[slot, 3] = -1


@qd.func
def bvh_query_aabb_next(
    node_lowers: _vec4_1d,
    node_uppers: _vec4_1d,
    primitive_indices: _i32_1d,
    item_lowers: _vec3_1d,
    item_uppers: _vec3_1d,
    stack: _i32_2d,
    state: _i32_2d,
    slot: int,
    q_lower: _vec3_t,
    q_upper: _vec3_t,
) -> i32:
    """One step of an AABB overlap query (warp bvh_query_next, AABB kind).

    Returns 1 and writes the hit primitive to ``state[slot, 3]``; returns 0
    when the query is exhausted. Call ``bvh_query_init`` first, then loop:
    ``found = bvh_query_aabb_next(...)`` / ``while found == 1: ...``.
    """
    count = state[slot, 0]
    prim_cur = state[slot, 1]
    prim_end = state[slot, 2]
    found = 0
    prim = -1
    doneqd = 1
    while doneqd == 1:
        if prim_cur < prim_end:
            # enumerating a packed leaf: test one primitive per step
            p = primitive_indices[prim_cur]
            prim_cur = prim_cur + 1
            il = item_lowers[p]
            iu = item_uppers[p]
            if _bvh_aabb_overlap(q_lower, q_upper, il, iu) == 1:
                found = 1
                prim = p
                doneqd = 0
        else:
            if count == 0:
                doneqd = 0
            else:
                count = count - 1
                node_index = stack[slot, count]
                nl = node_lowers[node_index]
                nu = node_uppers[node_index]
                nlo = qd.math.vec3(nl[0], nl[1], nl[2])
                nhi = qd.math.vec3(nu[0], nu[1], nu[2])
                if _bvh_aabb_overlap(q_lower, q_upper, nlo, nhi) == 1:
                    left = _bvh_node_i(nl)
                    right = _bvh_node_i(nu)
                    if _bvh_is_leaf(nl) == 1:
                        if right - left == 1:
                            # single-primitive leaf: its AABB already passed
                            p = primitive_indices[left]
                            found = 1
                            prim = p
                            doneqd = 0
                        else:
                            prim_cur = left
                            prim_end = right
                    else:
                        stack[slot, count] = left
                        count = count + 1
                        stack[slot, count] = right
                        count = count + 1
    state[slot, 0] = count
    state[slot, 1] = prim_cur
    state[slot, 2] = prim_end
    state[slot, 3] = prim
    return found


@qd.func
def bvh_query_ray_next(
    node_lowers: _vec4_1d,
    node_uppers: _vec4_1d,
    primitive_indices: _i32_1d,
    item_lowers: _vec3_1d,
    item_uppers: _vec3_1d,
    stack: _i32_2d,
    state: _i32_2d,
    slot: int,
    q_start: _vec3_t,
    q_dir: _vec3_t,
    max_dist: f32,
) -> i32:
    """One step of a ray query (warp bvh_query_ray_next; half-open ``t < max_dist``)."""
    rcp = qd.math.vec3(1.0 / q_dir[0], 1.0 / q_dir[1], 1.0 / q_dir[2])
    count = state[slot, 0]
    prim_cur = state[slot, 1]
    prim_end = state[slot, 2]
    found = 0
    prim = -1
    doneqd = 1
    while doneqd == 1:
        if prim_cur < prim_end:
            p = primitive_indices[prim_cur]
            prim_cur = prim_cur + 1
            il = item_lowers[p]
            iu = item_uppers[p]
            t = _bvh_ray_aabb_t(q_start, rcp, il, iu)
            if t < max_dist:
                found = 1
                prim = p
                doneqd = 0
        else:
            if count == 0:
                doneqd = 0
            else:
                count = count - 1
                node_index = stack[slot, count]
                nl = node_lowers[node_index]
                nu = node_uppers[node_index]
                nlo = qd.math.vec3(nl[0], nl[1], nl[2])
                nhi = qd.math.vec3(nu[0], nu[1], nu[2])
                t = _bvh_ray_aabb_t(q_start, rcp, nlo, nhi)
                if t < max_dist:
                    left = _bvh_node_i(nl)
                    right = _bvh_node_i(nu)
                    if _bvh_is_leaf(nl) == 1:
                        if right - left == 1:
                            p = primitive_indices[left]
                            found = 1
                            prim = p
                            doneqd = 0
                        else:
                            prim_cur = left
                            prim_end = right
                    else:
                        stack[slot, count] = left
                        count = count + 1
                        stack[slot, count] = right
                        count = count + 1
    state[slot, 0] = count
    state[slot, 1] = prim_cur
    state[slot, 2] = prim_end
    state[slot, 3] = prim
    return found


@qd.func
def bvh_query_sphere_next(
    node_lowers: _vec4_1d,
    node_uppers: _vec4_1d,
    primitive_indices: _i32_1d,
    item_lowers: _vec3_1d,
    item_uppers: _vec3_1d,
    stack: _i32_2d,
    state: _i32_2d,
    slot: int,
    q_center: _vec3_t,
    q_radius: f32,
) -> i32:
    """One step of a sphere query (warp bvh_query_sphere_next; exact node test)."""
    radius = qd.max(q_radius, 0.0)
    radius_sq = radius * radius
    count = state[slot, 0]
    prim_cur = state[slot, 1]
    prim_end = state[slot, 2]
    found = 0
    prim = -1
    doneqd = 1
    while doneqd == 1:
        if prim_cur < prim_end:
            p = primitive_indices[prim_cur]
            prim_cur = prim_cur + 1
            il = item_lowers[p]
            iu = item_uppers[p]
            if _bvh_sphere_aabb_hit(q_center, radius_sq, il, iu) == 1:
                found = 1
                prim = p
                doneqd = 0
        else:
            if count == 0:
                doneqd = 0
            else:
                count = count - 1
                node_index = stack[slot, count]
                nl = node_lowers[node_index]
                nu = node_uppers[node_index]
                nlo = qd.math.vec3(nl[0], nl[1], nl[2])
                nhi = qd.math.vec3(nu[0], nu[1], nu[2])
                if _bvh_sphere_aabb_hit(q_center, radius_sq, nlo, nhi) == 1:
                    left = _bvh_node_i(nl)
                    right = _bvh_node_i(nu)
                    if _bvh_is_leaf(nl) == 1:
                        if right - left == 1:
                            p = primitive_indices[left]
                            found = 1
                            prim = p
                            doneqd = 0
                        else:
                            prim_cur = left
                            prim_end = right
                    else:
                        stack[slot, count] = left
                        count = count + 1
                        stack[slot, count] = right
                        count = count + 1
    state[slot, 0] = count
    state[slot, 1] = prim_cur
    state[slot, 2] = prim_end
    state[slot, 3] = prim
    return found


# ------------------------------------------------------------ refit kernel ---

@qd.kernel
def _bvh_refit(
    node_parents: _i32_1d,
    node_counts: _i32_1d,
    primitive_indices: _i32_1d,
    node_lowers: _vec4_1d,
    node_uppers: _vec4_1d,
    item_lowers: _vec3_1d,
    item_uppers: _vec3_1d,
    n_leaves: int,
):
    """warp bvh_refit_kernel (bvh.cu:42): bottom-up bounds refresh, structure kept.

    One iteration per original leaf (possibly muted); each recomputes its bound
    from its primitive range and climbs while it is the completing child.
    Serialized: completion chains hand work across iterations (deterministic).
    """
    loop_config(serialize=True)
    for index in range(n_leaves):
        bl = node_lowers[index]
        if _bvh_is_leaf(bl) == 1:
            bu = node_uppers[index]
            parent = node_parents[index]
            parent_leaf = 0
            if parent != -1:
                parent_leaf = _bvh_is_leaf(node_lowers[parent])
            if parent == -1 or parent_leaf == 0:
                # valid leaf: recompute from its primitive range
                start = _bvh_node_i(bl)
                end = _bvh_node_i(bu)
                lo = qd.math.vec3(_INF_F32, _INF_F32, _INF_F32)
                hi = qd.math.vec3(-_INF_F32, -_INF_F32, -_INF_F32)
                for c in range(start, end):
                    p = primitive_indices[c]
                    pil = item_lowers[p]
                    piu = item_uppers[p]
                    nx = qd.min(lo[0], pil[0])
                    ny = qd.min(lo[1], pil[1])
                    nz = qd.min(lo[2], pil[2])
                    mx = qd.max(hi[0], piu[0])
                    my = qd.max(hi[1], piu[1])
                    mz = qd.max(hi[2], piu[2])
                    lo = qd.math.vec3(nx, ny, nz)
                    hi = qd.math.vec3(mx, my, mz)
                node_lowers[index] = qd.math.vec4(lo[0], lo[1], lo[2], bl[3])
                node_uppers[index] = qd.math.vec4(hi[0], hi[1], hi[2], bu[3])

            # climb: complete parents, refresh their bounds, move up
            doneqd = 0
            cur = index
            while doneqd == 0:
                parent = node_parents[cur]
                if parent == -1:
                    doneqd = 1
                else:
                    finished = atomic_add(node_counts[parent], 1)
                    if finished == 1:
                        pb = node_lowers[parent]
                        pbu = node_uppers[parent]
                        if _bvh_is_leaf(pb) == 1:
                            # packed leaf parent: children lost, recompute range
                            pp = node_parents[parent]
                            pp_leaf = 0
                            if pp != -1:
                                pp_leaf = _bvh_is_leaf(node_lowers[pp])
                            if pp == -1 or pp_leaf == 0:
                                start = _bvh_node_i(pb)
                                end = _bvh_node_i(pbu)
                                lo = qd.math.vec3(_INF_F32, _INF_F32, _INF_F32)
                                hi = qd.math.vec3(-_INF_F32, -_INF_F32, -_INF_F32)
                                for c in range(start, end):
                                    p = primitive_indices[c]
                                    pil = item_lowers[p]
                                    piu = item_uppers[p]
                                    nx = qd.min(lo[0], pil[0])
                                    ny = qd.min(lo[1], pil[1])
                                    nz = qd.min(lo[2], pil[2])
                                    mx = qd.max(hi[0], piu[0])
                                    my = qd.max(hi[1], piu[1])
                                    mz = qd.max(hi[2], piu[2])
                                    lo = qd.math.vec3(nx, ny, nz)
                                    hi = qd.math.vec3(mx, my, mz)
                                node_lowers[parent] = qd.math.vec4(lo[0], lo[1], lo[2], pb[3])
                                node_uppers[parent] = qd.math.vec4(hi[0], hi[1], hi[2], pbu[3])
                        else:
                            lc = _bvh_node_i(pb)
                            rc = _bvh_node_i(pbu)
                            lb = node_lowers[lc]
                            lu = node_uppers[lc]
                            rb = node_lowers[rc]
                            ru = node_uppers[rc]
                            nx = qd.min(lb[0], rb[0])
                            ny = qd.min(lb[1], rb[1])
                            nz = qd.min(lb[2], rb[2])
                            mx = qd.max(lu[0], ru[0])
                            my = qd.max(lu[1], ru[1])
                            mz = qd.max(lu[2], ru[2])
                            node_lowers[parent] = qd.math.vec4(nx, ny, nz, pb[3])
                            node_uppers[parent] = qd.math.vec4(mx, my, mz, pbu[3])
                        cur = parent
                    else:
                        # first child to arrive: parent not ready, terminate
                        doneqd = 1


# --------------------------------------------------------- batch queries ----
# Usage references: collect up to max_hits candidates per query slot. The
# candidate set of a query is independent of the tree shape (every primitive
# whose own bounds pass the predicate is enumerated, none beyond it).

@qd.kernel
def bvh_query_aabb_batch(
    root: _i32_1d,
    node_lowers: _vec4_1d,
    node_uppers: _vec4_1d,
    primitive_indices: _i32_1d,
    item_lowers: _vec3_1d,
    item_uppers: _vec3_1d,
    q_lower_a: _vec3_1d,
    q_upper_a: _vec3_1d,
    stack: _i32_2d,
    state: _i32_2d,
    max_hits: int,
    n_queries: int,
    hit_counts: _i32_1d,
    hit_indices: _i32_2d,
):
    for q in range(n_queries):
        bvh_query_init(state, stack, q, root[0])
        count = 0
        found = 1
        while found == 1:
            if count < max_hits:
                found = bvh_query_aabb_next(node_lowers, node_uppers, primitive_indices,
                                            item_lowers, item_uppers, stack, state, q,
                                            q_lower_a[q], q_upper_a[q])
                if found == 1:
                    hit_indices[q, count] = state[q, 3]
                    count = count + 1
            else:
                found = 0
        hit_counts[q] = count


@qd.kernel
def bvh_query_ray_batch(
    root: _i32_1d,
    node_lowers: _vec4_1d,
    node_uppers: _vec4_1d,
    primitive_indices: _i32_1d,
    item_lowers: _vec3_1d,
    item_uppers: _vec3_1d,
    q_start_a: _vec3_1d,
    q_dir_a: _vec3_1d,
    max_dist_a: _f32_1d,
    stack: _i32_2d,
    state: _i32_2d,
    max_hits: int,
    n_queries: int,
    hit_counts: _i32_1d,
    hit_indices: _i32_2d,
):
    for q in range(n_queries):
        bvh_query_init(state, stack, q, root[0])
        count = 0
        found = 1
        while found == 1:
            if count < max_hits:
                found = bvh_query_ray_next(node_lowers, node_uppers, primitive_indices,
                                           item_lowers, item_uppers, stack, state, q,
                                           q_start_a[q], q_dir_a[q], max_dist_a[q])
                if found == 1:
                    hit_indices[q, count] = state[q, 3]
                    count = count + 1
            else:
                found = 0
        hit_counts[q] = count


@qd.kernel
def bvh_query_sphere_batch(
    root: _i32_1d,
    node_lowers: _vec4_1d,
    node_uppers: _vec4_1d,
    primitive_indices: _i32_1d,
    item_lowers: _vec3_1d,
    item_uppers: _vec3_1d,
    q_center_a: _vec3_1d,
    q_radius_a: _f32_1d,
    stack: _i32_2d,
    state: _i32_2d,
    max_hits: int,
    n_queries: int,
    hit_counts: _i32_1d,
    hit_indices: _i32_2d,
):
    for q in range(n_queries):
        bvh_query_init(state, stack, q, root[0])
        count = 0
        found = 1
        while found == 1:
            if count < max_hits:
                found = bvh_query_sphere_next(node_lowers, node_uppers, primitive_indices,
                                              item_lowers, item_uppers, stack, state, q,
                                              q_center_a[q], q_radius_a[q])
                if found == 1:
                    hit_indices[q, count] = state[q, 3]
                    count = count + 1
            else:
                found = 0
        hit_counts[q] = count


# ---------------------------------------------------------------- host API ---

def _bvh_zeros(shape, dtype, np_dtype, elem=1):
    arr = qd.ndarray(dtype=dtype, shape=shape)
    logical = shape if isinstance(shape, tuple) else (shape,)
    np_shape = logical + (elem,) if elem > 1 else logical
    arr.from_numpy(np.zeros(np_shape, dtype=np_dtype))
    return arr


class Bvh:
    """Host handle over a qd-native LBVH (the wp.Bvh counterpart).

    Args:
        lower, upper: caller-owned vec3 qd ndarrays (shape (n,)) of the item
            bounds; mutated bounds are picked up by :meth:`refit`.
        leaf_size: pack any node spanning <= leaf_size items into one leaf
            (warp semantic; 1 = one item per leaf).

    After :meth:`build`, query from any qd kernel via the module-level funcs,
    passing the tree arrays as ``bvh.node_lowers`` etc.; refit after moving
    bounds. n == 0 builds an empty tree (every query immediately exhausts).
    """

    def __init__(self, lower, upper, leaf_size=1, num_items=None):
        # num_items overrides shape[0] because qd pads zero-length arrays to
        # width 1 (the shape cannot be trusted for the n=0 case)
        n = self._true_n(lower, num_items)
        if self._true_n(upper, num_items) != n:
            raise ValueError("bvh: lower/upper shapes differ")
        self.num_items = n
        self.leaf_size = int(leaf_size)
        self.lower = lower
        self.upper = upper
        cap = max(2 * n - 1, 1)
        buf_n = max(n, 1)
        self.node_lowers = _bvh_zeros(cap, qd.math.vec4, np.float32, elem=4)
        self.node_uppers = _bvh_zeros(cap, qd.math.vec4, np.float32, elem=4)
        self.node_parents = _bvh_zeros(cap, qd.i32, np.int32)
        self.range_lefts = _bvh_zeros(cap, qd.i32, np.int32)
        self.range_rights = _bvh_zeros(cap, qd.i32, np.int32)
        self.node_counts = _bvh_zeros(cap, qd.i32, np.int32)
        self.deltas = _bvh_zeros(buf_n, qd.i32, np.int32)
        self.primitive_indices = _bvh_zeros(buf_n, qd.i32, np.int32)
        self.keys = _bvh_zeros(buf_n, qd.i32, np.int32)
        self.root = _bvh_zeros(1, qd.i32, np.int32)
        self.root.fill(-1)
        self._grid_lower = _bvh_zeros(1, qd.math.vec3, np.float32, elem=3)
        self._grid_inv_edges = _bvh_zeros(1, qd.math.vec3, np.float32, elem=3)

    @staticmethod
    def _true_n(arr, num_items):
        if num_items is not None:
            return int(num_items)
        return int(arr.shape[0])

    def build(self):
        """Construct the LBVH from the current bounds (one-off; ~O(n log^2 n) sort)."""
        n = self.num_items
        if n == 0:
            self.root.fill(-1)
            return
        lo = self.lower.to_numpy().astype(np.float32)
        up = self.upper.to_numpy().astype(np.float32)
        grid_min = lo.min(axis=0)
        grid_max = up.max(axis=0)
        edges = (grid_max - grid_min) + np.float32(1.0e-4)
        inv_edges = (np.float32(1.0) / edges).astype(np.float32)
        self._grid_lower.from_numpy(grid_min.reshape(1, 3))
        self._grid_inv_edges.from_numpy(inv_edges.reshape(1, 3))
        _bvh_morton(self.lower, self.upper, self._grid_lower, self._grid_inv_edges,
                    self.keys, self.primitive_indices, n)
        qd.sync()
        # Batcher odd-even merge schedule (quadrants parallel_sort form)
        p = 1
        while p < n:
            k = p
            while k >= 1:
                invocations = int((n - k - k % p) / (2 * k)) + 1
                _bvh_sort_stage(self.keys, self.primitive_indices, n, p, k, invocations)
                qd.sync()
                k = int(k / 2)
            p = int(p * 2)
        _bvh_build_rest(self.keys, self.primitive_indices, self.lower, self.upper,
                        self.node_lowers, self.node_uppers, self.node_parents,
                        self.range_lefts, self.range_rights, self.node_counts,
                        self.deltas, self.root, n, self.leaf_size)
        qd.sync()

    def refit(self):
        """Refresh node bounds after mutating ``lower``/``upper``; structure kept."""
        if self.num_items == 0:
            return
        self.node_counts.fill(0)
        _bvh_refit(self.node_parents, self.node_counts, self.primitive_indices,
                   self.node_lowers, self.node_uppers, self.lower, self.upper,
                   self.num_items)
        qd.sync()
