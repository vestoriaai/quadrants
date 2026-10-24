# type: ignore

import functools
import os
import sys

from quadrants._lib import core as _qd_core
from quadrants._logging import info

pybuf_enabled = False
_env_enable_pybuf = os.environ.get("QD_ENABLE_PYBUF", "1")
if not _env_enable_pybuf or int(_env_enable_pybuf):
    # When using in Jupyter / IDLE, the sys.stdout will be their wrapped ones.
    # While sys.__stdout__ should always be the raw console stdout.
    pybuf_enabled = sys.stdout is not sys.__stdout__

_qd_core.toggle_python_print_buffer(pybuf_enabled)


def _shell_pop_print(old_call):
    if not pybuf_enabled:
        # zero-overhead! (C++ writes kernel prints straight to the real fd)
        return old_call

    info("Graphical python shell detected, using wrapped sys.stdout")

    # A53: kernel prints go to the C++ python print buffer and take effect at
    # qd.sync() (taichi #1303 semantics) -- PyQuadrants.sync() pops the buffer
    # and writes it to sys.__stdout__ (the real fd), so pytest capfd/capsys and
    # Jupyter both see it. The per-call pop+print that used to live here wrote
    # to the *wrapped* sys.stdout (invisible to capfd) and contradicted the
    # documented sync-only semantics. Keeping the wrapper (instead of removing
    # the decoration) preserves the wrapped-shell detection log line.
    @functools.wraps(old_call)
    def new_call(*args, **kwargs):
        return old_call(*args, **kwargs)

    return new_call
