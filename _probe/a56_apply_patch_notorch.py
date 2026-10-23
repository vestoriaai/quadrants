"""Apply the A56 atexit-dump patch to an installed quadrants wheel copy (short anchors).

Same patch as a56_apply_patch_to_wheel.py, kept separate so the torch-venv
version's exact anchors cannot drift. Usage: python apply_a56_patch_notorch.py
<site-packages>/quadrants/lang/impl.py
"""

import sys

path = sys.argv[1]
with open(path) as f:
    src = f.read()

pairs = [
    (
        "import numbers\nimport threading\nimport weakref",
        "import atexit\nimport numbers\nimport threading\nimport weakref",
    ),
    (
        "    def create_program(self):\n        if self._prog is None:\n            self._prog = _qd_core.Program()\n",
        "    def create_program(self):\n"
        "        if self._prog is None:\n"
        "            self._prog = _qd_core.Program()\n"
        "            _register_exit_cache_dump()\n",
    ),
    (
        "def get_runtime() -> PyQuadrants:\n    return pyquadrants\n",
        "def get_runtime() -> PyQuadrants:\n"
        "    return pyquadrants\n"
        "\n"
        "\n"
        "# A56: see _register_exit_cache_dump docstring in the source tree.\n"
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
        "        prog = getattr(pyquadrants, \"_prog\", None)\n"
        "        if prog is not None:\n"
        "            try:\n"
        "                prog.dump_cache_data_to_disk()\n"
        "            except Exception:  # noqa: BLE001\n"
        "                pass\n"
        "\n"
        "    atexit.register(_dump_caches_at_exit)\n",
    ),
]

for i, (anchor, patch) in enumerate(pairs):
    assert anchor in src, f"anchor {i} not found"
    assert src.count(anchor) == 1, f"anchor {i} not unique"
    src = src.replace(anchor, patch)

with open(path, "w") as f:
    f.write(src)

import ast

ast.parse(src)
print("PATCH_OK", path)
