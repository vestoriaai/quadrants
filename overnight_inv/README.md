# Overnight investigation probes (branch overnight/inv-cache-keys)

Environment: private venv ../.venv-overnight (wheel dist/quadrants-1.3.4.dev7+g7aaaa7d73-*.whl).
All caches are PRIVATE dirs under caches/ — never point these at shared quadrants caches.

- probe_keys.py      — investigation 1: cross-process/offline-cache key stability.
                       Variants A (field+static-NS class kernel), B (field-free), C (fields, no
                       static NS), D (two helpers), E (spline_eval-like), P (module fields);
                       --perturb N shifts snode-id history, --two-inst, --pure, --arch cpu|metal.
                       Prints CompileResult.cache_key + per-field SNode ids.
- probe_phases.py    — NS scaling: p1 compile process vs p2 fresh process (disk hit).
- probe_multiarch.py — investigation 2: double qd.init / mixed-arch scenario matrix
                       (base/abandon/reuse/same_arch/abandon_nd/realloc_{cpu,metal}).
- out/               — verbatim probe logs.
Run with the patched source tree: PYTHONPATH=<repo>/python (test-only copies of the wheel's
_lib/core/*.so and _lib/runtime/ live untracked in the source tree).
