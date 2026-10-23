"""Verify the modified [tool.scikit-build] settings actually resolve to the
intended values (catches TOML key typos that would silently land nowhere).

Run with an interpreter that has scikit_build_core importable:
    python _probe/verify_pyproject_settings.py <path-to-pyproject.toml>
"""

import sys
import tomllib

path = sys.argv[1]

from scikit_build_core.settings.skbuild_read_settings import SettingsReader  # noqa: E402

with open(path, "rb") as f:
    pyproject = tomllib.load(f)

reader = SettingsReader(pyproject, {}, state="wheel", verify_conf=False)
bt = reader.settings.cmake.build_type
strip = reader.settings.install.strip
print(f"cmake.build-type = {bt!r}")
print(f"install.strip    = {strip!r}")
assert bt == "RelWithDebInfo", f"build-type not picked up: {bt!r}"
assert strip is False, f"strip not picked up: {strip!r}"
print("SETTINGS_OK")
