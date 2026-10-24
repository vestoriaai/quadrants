import subprocess
import sys

from tests import test_utils


@test_utils.test()
def test_pyi_stubs(tmpdir):
    test_code = """
import quadrants._lib.core.quadrants_python
reveal_type(quadrants._lib.core.quadrants_python)
"""
    test_file = tmpdir / "tmp_mypy_test.py"
    test_file.write(test_code)

    # A58: two environment quirks made this check report "Unknown" even though
    # the .pyi ships next to the .so. (1) The pip-installed pyright resolves
    # module search paths from the *real* interpreter behind sys.executable
    # (it follows the venv symlink to the base python) -- pin --pythonpath to
    # the venv interpreter so site-packages come from the venv under test.
    # (2) Run from the tmpdir: with cwd at the repo root pyright treats the
    # source tree's python/quadrants as an import root and resolves the module
    # to the stub-less source layout.
    res = subprocess.check_output(
        [sys.executable, "-m", "pyright", "--pythonpath", sys.executable, test_file],
        cwd=str(tmpdir),
    ).decode("utf-8")
    assert "unknown" not in res.lower()
