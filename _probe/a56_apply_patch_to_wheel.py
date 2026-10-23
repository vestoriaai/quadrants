"""Apply the A56 atexit-dump patch to an installed quadrants wheel copy.

Exact-string replacement with assertion on every anchor (no silent no-ops).
Usage: python apply_a56_patch.py <site-packages>/quadrants/lang/impl.py
"""

import sys

path = sys.argv[1]
with open(path) as f:
    src = f.read()

anchor_import = "import numbers\nimport threading\nimport weakref"
patch_import = "import atexit\nimport numbers\nimport threading\nimport weakref"

anchor_grt = (
    "def get_runtime() -> PyQuadrants:\n"
    "    return pyquadrants\n"
)
patch_grt = (
    "def get_runtime() -> PyQuadrants:\n"
    "    return pyquadrants\n"
    "\n"
    "\n"
    "# A56: interpreter shutdown does not reliably destroy the Program -- co-imported heavy\n"
    "# frameworks (torch is pulled in transitively by ``import quadrants`` whenever it is\n"
    "# installed) change module-clear ordering so ``~Program`` (and the kernel-cache dump that\n"
    "# lives in ``Program::finalize``) never runs. The offline cache then misses on the next\n"
    "# process: the Python-side L1/L2 entries are written eagerly, but the C++ artifacts they\n"
    "# name are never dumped. Flush the dump via atexit instead. Only the dump, not finalize():\n"
    "# other exit hooks may still use the runtime, and ``KernelCompilationManager::dump`` is\n"
    "# idempotent (it moves out and clears ``caching_kernels_``), so the finalize-time dump\n"
    "# no-ops after ours.\n"
    "_exit_cache_dump_registered = False\n"
    "\n"
    "\n"
    "def _register_exit_cache_dump() -> None:\n"
    "    global _exit_cache_dump_registered\n"
    "    if _exit_cache_dump_registered:\n"
    "        return\n"
    "    _exit_cache_dump_registered = True\n"
    "\n"
    "    def _dump_caches_at_exit():\n"
    "        # Late-bind through the module global: ``reset()`` swaps in a fresh PyQuadrants\n"
    "        # (and Program) after this hook was registered.\n"
    "        prog = getattr(pyquadrants, \"_prog\", None)\n"
    "        if prog is not None:\n"
    "            try:\n"
    "                prog.dump_cache_data_to_disk()\n"
    "            except Exception:  # noqa: BLE001 - shutdown must not raise\n"
    "                pass\n"
    "\n"
    "    atexit.register(_dump_caches_at_exit)\n"
)

anchor_cp = (
    "    def create_program(self):\n"
    "        if self._prog is None:\n"
    "            self._prog = _qd_core.Program()\n"
)
patch_cp = (
    "    def create_program(self):\n"
    "        if self._prog is None:\n"
    "            self._prog = _qd_core.Program()\n"
    "            _register_exit_cache_dump()\n"
)

for name, anchor in [("import", anchor_import), ("get_runtime", anchor_grt), ("create_program", anchor_cp)]:
    assert anchor in src, f"anchor not found in installed wheel: {name}"
    assert src.count(anchor) == 1, f"anchor not unique: {name}"

src = src.replace(anchor_import, patch_import)
src = src.replace(anchor_grt, patch_grt)
src = src.replace(anchor_cp, patch_cp)

with open(path, "w") as f:
    f.write(src)

import ast

ast.parse(src)
print("PATCH_OK", path)
