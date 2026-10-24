"""Apply the A53 sync-flush patch (batwt source) to an installed wheel copy.

Usage: python a53_apply_to_wheel.py <venv-python>
Patches <venv>/site-packages/quadrants/lang/{shell.py,impl.py} in place with
anchored exact-string replacements.
"""

import ast
import pathlib
import sys

venv = pathlib.Path(sys.argv[1])
sp = next(venv.glob("lib/python*/site-packages"))
shell_p = sp / "quadrants" / "lang" / "shell.py"
impl_p = sp / "quadrants" / "lang" / "impl.py"

# --- shell.py ---
s = shell_p.read_text()
old = '''    @functools.wraps(old_call)
    def new_call(*args, **kwargs):
        ret = old_call(*args, **kwargs)
        # print's in kernel won't take effect until qd.sync(), discussion:
        # https://github.com/taichi-dev/taichi/pull/1303#discussion_r444897102
        print(_qd_core.pop_python_print_buffer(), end="")
        return ret

    return new_call'''
new = '''    # A53: kernel prints take effect at qd.sync() -- PyQuadrants.sync() pops
    # the buffer and writes it to sys.__stdout__ (the real fd). The per-call
    # pop+print that used to live here wrote to the *wrapped* sys.stdout
    # (invisible to capfd) and contradicted the sync-only semantics.
    @functools.wraps(old_call)
    def new_call(*args, **kwargs):
        return old_call(*args, **kwargs)

    return new_call'''
assert old in s and s.count(old) == 1
shell_p.write_text(s.replace(old, new))

# --- impl.py: sys import ---
s = impl_p.read_text()
old = "import atexit\nimport numbers\nimport threading\nimport weakref\n"
new = "import atexit\nimport numbers\nimport sys\nimport threading\nimport weakref\n"
assert old in s and s.count(old) == 1
s = s.replace(old, new)

old = '''    def sync(self):
        if is_python_backend():
            return
        self.materialize()
        assert self._prog is not None
        try:
            self._prog.synchronize()
        except Exception as e:
            wrapped = handle_exception_from_cpp(e)
            if wrapped is e:
                raise
            raise wrapped from None'''
new = '''    def sync(self):
        if is_python_backend():
            return
        self.materialize()
        assert self._prog is not None
        try:
            self._prog.synchronize()
        except Exception as e:
            wrapped = handle_exception_from_cpp(e)
            if wrapped is e:
                raise
            raise wrapped from None
        self._flush_python_print_buffer()

    @staticmethod
    def _flush_python_print_buffer():
        # A53: kernel prints land in the C++ python print buffer whenever a
        # wrapped sys.stdout was detected at import (shell.py) -- flush them to
        # sys.__stdout__ (the real fd) here, restoring the documented
        # "prints take effect at qd.sync()" semantics for capfd/capsys/Jupyter
        # alike. No-op when the buffer path is disabled (C++ already wrote to
        # the real fd). Late import guarded: shell imports kernel machinery,
        # so import it lazily to avoid an import cycle.
        from quadrants.lang.shell import pybuf_enabled

        if not pybuf_enabled:
            return
        try:
            buf = _qd_core.pop_python_print_buffer()
        except Exception:  # noqa: BLE001 - flush must never break sync
            return
        if buf:
            sys.__stdout__.write(buf)
            sys.__stdout__.flush()'''
assert old in s and s.count(old) == 1
s = s.replace(old, new)
impl_p.write_text(s)

ast.parse(shell_p.read_text())
ast.parse(impl_p.read_text())
print("A53_PATCH_OK")
