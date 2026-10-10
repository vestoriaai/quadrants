# Overnight investigation report — quadrants compile-cache persistence & one-arch-per-process

Branch: `overnight/inv-cache-keys` (cut from `feat/readback` @ `7aaaa7d7`, which is exactly the
built wheel `dist/quadrants-1.3.4.dev7+g7aaaa7d73-...whl` used by the probes).
Environment: private venv `/Users/mini0/Code/quadrants/.venv-overnight` (py3.12 + that wheel +
numpy). All probes use **private** `offline_cache_file_path` dirs under
`/Users/mini0/Code/quadrants/overnight_inv/caches/` — no robostudio venv or shared cache touched.

Probes: `overnight_inv/probe_keys.py` (variants A/B/C/D/E/P), `overnight_inv/probe_phases.py`
(NS scaling + two-process timing). Raw outputs: `overnight_inv/out/`.

---

## Investigation 1 — why coll_scan-like class kernels never persist across processes

### TL;DR

The offline disk cache key for every kernel is computed in C++ from the kernel's AST
serialization, and that serialization embeds **raw per-process SNode allocation ids** for every
`qd.field` the kernel touches. SNode ids come from a process-global monotonic counter, so the key
is a function of the *field-creation history of the process*, not of the kernel source. Any
difference in what was constructed before the kernel's first launch (imports, instance count,
construction order, extra pads) → different key → guaranteed disk miss → full recompile.
Kernels that touch **no fields** (data via ndarray args) have source-only keys and persist/share
everywhere. This is "per-instance keying" in the literal sense: every `TrajKin` instance creates
fresh fields → fresh ids → its own key, even in one process.

Additionally, the Python-side fastcache (the only layer that can skip the expensive Python AST
transform across processes) is (a) gated on `is_pure`, and plain `@qd.kernel` is **not pure**, and
(b) even when forced on, it **hard-disables itself on any Field-typed member** at a kernel-read
path (`[FASTCACHE][INVALID_FUNC]`). So field-reading class kernels never get AST-skip persistence
at the Python layer either.

### Mechanism with file:line (tree = feat/readback @ 7aaaa7d7)

1. Compile path: `python/quadrants/lang/kernel.py:802` `prog.compile_kernel(...)` →
   `quadrants/program/program.cpp:164` → `KernelCompilationManager::load_or_compile`
   (`quadrants/compilation_manager/kernel_compilation_manager.cpp:64`).
2. Key: `make_kernel_key` (`kernel_compilation_manager.cpp:176-186`) →
   `get_hashed_offline_cache_key` (`quadrants/analysis/offline_cache_util.cpp:171-199`) =
   SHA256(compile_config ‖ device_caps ‖ parameter_list ‖ rets ‖ **AST serialization** ‖ autodiff).
   Disk artifacts are `<key>.qdc` under `<offline_cache_file_path>/kernel_compilation_manager/`
   (`make_filename`, `kernel_compilation_manager.cpp:31,141`).
3. **The AST serializer leaks SNode allocation ids**:
   - `quadrants/analysis/gen_offline_cache_key.cpp:526-536` — `emit(const SNode*)` writes
     `(snode->get_snode_tree_id(), snode->id)` into the body stream for every field reference;
   - `gen_offline_cache_key.cpp:424-451` (`emit_dependencies`) additionally hashes each referenced
     tree root via `get_hashed_offline_cache_key_of_snode`;
   - `quadrants/analysis/offline_cache_util.cpp:99-151` — that tree hash serializes the whole
     tree including `snode->id` (lines 103, 121).
4. **snode->id is a process-global counter**: `quadrants/ir/snode.cpp:191`
   `id = counter++;` (`static std::atomic<int> counter`, `snode.h:87`). Every `qd.field(...)`
   call allocates several ids (root/dense/place per axis). So the key depends on how many fields
   were created earlier in the same process, in what order — i.e. on instance construction
   history. (Parameter list itself is stable: `program/callable.h:20-27` serializes only
   is_array/total_dim/format/dtype/needs_grad/element_shape/ptype — no names, no ids.)
5. Helper `@qd.func` ids are **first-use order**, not pointer order: `emit(Function*)`
   (`gen_offline_cache_key.cpp:507-517`) assigns `real_funcs_.size()` at first insert during the
   deterministic AST walk (the `std::map<Function*, size_t>` is only the lookup structure, its
   pointer ordering never reaches the key). Initially suspected as a cross-process hazard;
   re-checked the code and probed it (variant D, two helpers): same key in both processes —
   deterministic, NOT a defect.
6. Python fastcache layer (would skip the AST transform cross-process):
   - Gate: `kernel.py:423` `if self.runtime.src_ll_cache and self.quadrants_callable and
     self.quadrants_callable.is_pure:` — plain `@qd.kernel` sets `is_pure=False`
     (`kernel_impl.py:282`), so **bspline_kernels' six kernels never even try the fastcache**.
   - Even with `is_pure=True`, `args_hasher.stringfy_obj_type` fails the whole call on
     Field-typed members: `python/quadrants/lang/_fast_caching/args_hasher.py:350-355` (ScalarField)
     and `:363-368` (MatrixField) → `_FAIL_FASTCACHE` →
     `[FASTCACHE][INVALID_FUNC] The pure function scan could not be fast cached, because one or
     more parameter types were invalid` (`src_hasher.py:175-179`). L2 is never written, artifact
     restore never attempted. Verified by forcing `is_pure=True` on variant A
     (`caches/s_A_pure`: only an L1 pruning entry is written, `python_side_cache/` has 1 file, no
     L2, no restore).

### Probe evidence (verbatim excerpts; full outputs under overnight_inv/out/)

All runs: private cache dirs; `cxx_key` is `CompileResult.cache_key` read back in Python.

Identical scripts persist; ANY perturbation of the preceding field-creation history re-keys:

```
== A proc1: cxx_key T6388871a06b7cec08c1713c2ad979f3215dfb4268a2fa659466818f32b6ec31f
            snode_ids s_ctr=2 s_rad=6 s_par=8 coef=10 grid=15 (tree 0)   wall 0.037s
== A proc2: cxx_key T6388871a... (same)  wall 0.014s   ← QD_SHOW_COMPILING prints no "Compiling kernel" → disk HIT
== A proc3 (--perturb 3: 3 unrelated fields created first):
            cxx_key T9b1b9f6e7f43668e075b12a0b214b93cb9eb28f8b1319a9ca3ef6533ad299e23  ← DIFFERENT → MISS
            snode_ids s_ctr=8 s_rad=12 s_par=14 coef=16 grid=21
```

Per-instance in one process (the literal "per-instance keys"):

```
== A --two-inst (one process):
inst1 cxx_key T6388871a...  (fields in tree 0, ids 2..15)  "Compiling kernel 'scan_c62_0'"
inst2 cxx_key T0c0bc2ee...  (fields in tree 1, ids 18..31) "Compiling kernel 'scan_c62_1'"
```

Discriminator — the same class/kernel shape WITHOUT field reads is immune to everything:

```
== B (no fields) proc1/proc2/proc3(+perturb)/two-inst/+perturb&two-inst:
   cxx_key T1ea8705d9bf15a1a992a2e3ac2c91e31924f02e7b7dbd9b6db199d4ec6dc2042  — ONE key, always.
== C (fields, no static NS):     perturb → key changes (Td5c97c26… → T399babdb…)  → id-sensitive
== E (spline_eval-like, Wp field): perturb → key changes (T1a52c279… → Tc2fd162e…)  → id-sensitive
== P (module-scope fields, the classic "persistent" case): identical procs hit; perturb → miss.
== D (two @qd.func helpers):     proc1 == proc2 (pointer-order hazard did NOT fire).
== A on METAL: same behavior as cpu (identical procs hit; perturb → new key).
```

NS scaling of the static unroll (variant A light body, metal; real coll_scan body is much heavier):

```
NS   p1 compile(s)  p2 fresh-process(s, disk HIT)
16   0.345          0.014
64   0.246          0.043
128  0.643          0.081
256  2.098          0.158
```
Even on a disk hit, every fresh process re-pays the Python-side AST expansion + key
serialization (p2 column grows with NS) — for non-pure kernels nothing skips that phase.

### Why T23 saw "spline_eval persists, coll_scan doesn't"

Both read fields; both are equally id-sensitive. Persistence happens **iff the entire
field-creation history before the first launch is identical** between two processes. The first
instance constructed after the same import/world preamble (master kin → spline_eval etc.) shares
that prefix in every entry script; group kins constructed afterwards (coll_scan) sit at
history-dependent positions — T23's three entry scripts built different amounts of stuff
(different group counts / NS pads / `set_pairs` calls) before them → new ids → new keys →
guaranteed 22-minute recompiles. Different NS across processes also re-keys (correctly — the
unrolled AST differs).

### Fix assessment (the "stable key derivation" question)

- **A small safe wheel patch that makes field-reading class kernels persist across processes
  with differing histories is NOT achievable**, because it would be *unsound*: the compiled
  artifact bakes the field layout (byte offsets inside the SNodeTree, `num_snode_trees` is the
  only tree-related field in SPIR-V `CompiledKernelData`, spirv/compiled_kernel_data.h:19-20) of
  the process that compiled it. Different history ⇒ different offsets ⇒ a shared artifact would
  access the wrong memory. Today's id-keyed misses are *correct* misses in exactly those cases.
- The principled wheel-level redesign is **layout-keyed snodes** (M): replace raw ids with
  (tree structural hash, offset-path) in `offline_cache_util.cpp:get_offline_cache_key_of_snode_impl`
  and `gen_offline_cache_key.cpp:emit(const SNode*)`. Same layout ⇒ safe share (ids differ only
  under destroy/recreate-style histories); different layout ⇒ miss (required). Benefit is real but
  modest for T23 (its histories differ ⇒ layouts differ ⇒ still misses, correctly).
- Small wheel patches worth considering:
  1. allow the Python fastcache to hash Field members structurally
     (name+dtype+shape+C++ layout hash folded into the L2 key) instead of `_FAIL_FASTCACHE`
     (args_hasher.py:350-368) — M-sized, needs a new export of the C++ snode-layout hash; gives
     AST-skip persistence for field-reading kernels *when the layout matches*;
  2. `real_funcs_` needs no fix — see mechanism item 5 (initial suspicion withdrawn).
- **The lever that actually fixes T23 is packages-level** (matches T25's plan):
  1. make the sphere loop dynamic (kill the NS mega-unroll);
  2. pass per-instance data (s_ctr/s_rad/s_par/s_top/coef/esdf grids) as **ndarray args**, not
     instance fields — then the kernel is variant-B-class: ONE source-derived key shared across
     instances, processes, and histories (probe-proven above: `T1ea8705d…` always).

### Patch status for Investigation 1

No "stable key" patch was landed — by design: the sound fix is the M-sized layout-key redesign
above, and the packages-level lever (ndarray args + dynamic loop) is the one that actually
delivers cross-process persistence. Landed instead: probes + mechanism (this report) on branch
`overnight/inv-cache-keys` (commit `9f9c2528`).

---

## Investigation 2 — one-arch-per-process root cause (upstream bug #1)

### TL;DR (updated vs the C0-D-era diagnosis)

Re-probed on 1.3.4.dev7 (this tree's wheel, private caches, `overnight_inv/probe_multiarch.py`):

| scenario (fresh process each) | result on dev7 |
|---|---|
| init(cpu) → family A launch | PASS |
| init(metal) → family A launch | PASS |
| init(cpu) A → **init(metal)** → fresh family B, never touch A | **PASS** (abandoning the old family is safe) |
| same, A = ndarray-arg-only kernel, re-launch A after re-init | **PASS** (kernels survive reset by design, recompile under the new program) |
| init(cpu) A → init(metal) → **re-launch A** (field members) | UB → **now loud QuadrantsRuntimeError** (was: garbage-shape QuadrantsIndexError / SIGSEGV per C0-D) |
| same with metal first | same |
| init(cpu) A → init(cpu) → re-launch A | same UB → loud error (double-init is the hazard, arch-independent) |
| cpu launch → create field → re-launch (the C0-D "realloc" shot) | **no crash on dev7** (multi-SNodeTree layout: each `finalize_snode_tree` → `Program::add_snode_tree` allocates a separate tree, `program/program.cpp:238-258`; no root realloc path) |

So the residual defect is **not arch-specific**: `qd.init` #2 (any arch) finalizes Program #1
(`impl.py:PyQuadrants.clear` → `Program::finalize`, which also resets the process-global
`HostMemoryPool`, `program/program.cpp:394-429`) while Python `Field` objects survive with raw
SNode pointers into that freed memory. The kernel objects survive reset **by design**
(`impl.py:reset` carries them over and `Kernel.reset()` clears materializations) and ndarrays are
invalidated (`Ndarray._reset`, `lang/_ndarray.py:113`) — **Fields were the one class left with no
staleness pattern**. Any later use = use-after-free; the failure shape (loud index error, silent
garbage, SIGSEGV) depends on what the freed pool memory now holds. Mixed arch merely guarantees a
re-init happened; same-arch double init is equally hazardous (probed).

Per-arch state audit (what init switches, what survives):
- Destroyed with the Program at `clear()`: SNodeTrees + field storage, KernelCompilationManager +
  disk-cache handle, device/allocator (metal: `create_metal_device()` shared_ptr owned by
  ProgramImpl — no process-global device singleton blocks two live Programs),
  launch contexts.
- Survives reset by design: class-level `Kernel` objects (materializations cleared, recompiled
  under the new program), `TemplateMapper` (bound to programs via weakref + destroy callback).
- Survives reset UNSAFELY (pre-patch): `Field` (raw SNode/SNodeTree pointers) — fixed by this
  branch; user `Ndarray`s are tracked (`WeakSet`) and neutered; numpy-side buffers are plain
  host memory (fine).
- Process-global C++: TypeFactory (arch-free), SNode id counter (monotonic, never reused →
  the Investigation-1 keying), profiler mode (reconfigured per init), `Stmt::reset_counter()` in
  finalize.

### Feasibility verdicts

- **S (landed, this branch, commit `80445172`)**: stale-field guard. `PyQuadrants.generation`
  bumped per reset; `Field` records its generation; host ops and `impl.subscript` (the single
  choke point where a template-member/template-arg field enters compiled IR) refuse stale fields
  with a loud `QuadrantsRuntimeError`; `reset()` warns when live fields are orphaned. 64 lines in
  `impl.py`/`field.py`/`matrix.py` + `tests/python/test_stale_field.py` (4 tests). The C0-D class
  of incidents becomes a deterministic error naming the cause, wheel-level, for all users.
- **M**: keep-alive routing — retain per-program bundles (`_prog`, `_arch`, `_root_fb`,
  `materialized`, fields builder state) and switch them atomically (context manager), with
  launches routed to the program that owns the kernel's materialization (needs exporting the
  Program backref from `KernelCxx`/`Callable.program`, and a launch-time ownership check).
  Blocks: the launch path consults `impl.get_runtime().prog` at `kernel.py:796-848`; swap must
  be serialized by the existing `compilation_lock`; disk-cache dir is per-path not per-arch (key
  already includes arch, so sharing the dir is safe).
- **L**: true per-arch program table (`qd.init(arch)` switches among live Programs instead of
  finalizing; every Field/Kernel/ndarray tagged with its owning program; cross-family data
  exchange via ndarrays/host copies — exactly the robostudio ik(cpu)+esdf(metal) shape).
  Architecturally unblocked (no device singletons), but touches ownership of every object class
  plus the launch hot path; needs the full test battery, not an overnight patch.

Until M/L land, the wheel contract remains one-arch-per-process — with the difference that the
enforcement now lives in the wheel (loud, deterministic) instead of in robostudio's `qd_guard`
discipline.

### Patch status for Investigation 2

Branch `overnight/inv-cache-keys`, commit `80445172` (stale-field guard, S) + `9f9c2528`
(investigation 1 probes/report). Regression evidence on the patched source:
- multiarch matrix: base_cpu/base_metal/abandon/abandon_nd/realloc_cpu/realloc_metal PASS
  unchanged; reuse/same_arch/reuse_metal_cpu now fail LOUDLY with
  `QuadrantsRuntimeError: This qd.Field was created under a program that has been finalized...`
  (`overnight_inv/out/map_*.log`).
- Investigation-1 probes byte-identical keys (patch is compile-neutral):
  variant A `T6388871a…`, variant B two-inst `T1ea8705d…` unchanged.
- repo test battery (venv pytest + xdist `-n 0`): test_field/test_reset_ndarrays/
  test_fields_builder/test_all_field_shortcut 235 passed; data-oriented + template-mapper +
  kernel-arg-errors 82 passed; test_cache/test_offline_cache 57 passed 1 skipped; matrix/arg
  batches 180 passed 2 skipped, 80 passed 1 xfailed; new test_stale_field.py 4 passed.

No wheel was installed anywhere (dev-only venv `/Users/mini0/Code/quadrants/.venv-overnight`;
source tree runs via `PYTHONPATH=python` with the wheel's `_lib/core/*.so` and `_lib/runtime/`
copied in untracked for testing only).
