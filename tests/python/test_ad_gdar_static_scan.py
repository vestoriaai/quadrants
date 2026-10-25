"""A63: GDAR static scan for ndarray arguments + graceful skip of the check-bit checker.

Pins four behaviours:
* safe ndarray kernel   -> compile+launch under validation, no DEFINITE/POTENTIAL lines;
* same-index read-then-overwrite -> one DEFINITE warning line;
* distinct-index read/write aliasing -> one POTENTIAL warning line;
* mixed field+ndarray kernel -> no crash (the A63 repro: pre-fix, the check-bit
  checker aborted compilation with `validation.cpp:... is<MatrixPtrStmt>`).

Warnings go to C-level stderr (spdlog); assertions capture it via fd dup2.
"""

import contextlib
import os
import tempfile

import numpy as np
import pytest

import quadrants as qd
from tests import test_utils

def _launch_validated(kernel, *args):
    """Launch under a validation-mode tape (dummy scalar loss, no adjoint eval)
    so the kernel compiles as kCheckAutodiffValid and the A63 static scan runs."""
    dummy = qd.field(dtype=qd.f32, shape=(), needs_grad=True)
    with qd.ad.Tape(loss=dummy, validation=True, clear_gradients=False) as tape:
        kernel(*args)
    tape.gradient_evaluated = True  # skip eval_on_exit; scan already emitted at compile


DEFINITE = "[gdar-static-scan]"
DEFINITE_MARK = "DEFINITE"
POTENTIAL_MARK = "POTENTIAL"


@contextlib.contextmanager
def capture_stderr_fd():
    """Capture C-level stderr (spdlog bypasses Python sys.stderr); yields a
    stateless handle; use ``readout(tmp)`` INSIDE the with-block."""
    with tempfile.TemporaryFile(mode="w+") as tmp:
        saved = os.dup(2)
        os.dup2(tmp.fileno(), 2)
        try:
            yield tmp
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            tmp.seek(0)


def _read(tmp):
    """Must be called inside the capture context (file closes on exit)."""
    pos = tmp.tell()
    tmp.seek(0)
    out = tmp.read()
    tmp.seek(pos)
    return out


@test_utils.test(debug=True)
def test_scan_safe_kernel_no_warnings():
    x = qd.ndarray(dtype=qd.f32, shape=4)
    y = qd.ndarray(dtype=qd.f32, shape=4)
    x.from_numpy(np.array([1, 2, 3, 4], dtype=np.float32))

    @qd.kernel
    def k(x: qd.types.ndarray(dtype=qd.f32, ndim=1), y: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        for i in range(4):
            y[i] = x[i] * 2.0  # read x, write y: disjoint args -> safe

    with capture_stderr_fd() as tmp:
        _launch_validated(k, x, y)
        qd.sync()
        out = _read(tmp)
    assert DEFINITE_MARK not in out
    assert POTENTIAL_MARK not in out
    assert DEFINITE in out  # summary line present => scan ran


@test_utils.test(debug=True)
def test_scan_definite_same_index_read_then_overwrite():
    x = qd.ndarray(dtype=qd.f32, shape=4)
    y = qd.ndarray(dtype=qd.f32, shape=4)
    x.from_numpy(np.array([1, 2, 3, 4], dtype=np.float32))
    y.from_numpy(np.array([1, 1, 1, 1], dtype=np.float32))

    @qd.kernel
    def k(x: qd.types.ndarray(dtype=qd.f32, ndim=1), y: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        for i in range(4):
            y[i] = x[i] * y[i]  # read y[i] then overwrite y[i]: DEFINITE

    with capture_stderr_fd() as tmp:
        _launch_validated(k, x, y)
        qd.sync()
        out = _read(tmp)
    assert DEFINITE_MARK in out
    assert DEFINITE in out


@test_utils.test(debug=True)
def test_scan_potential_distinct_index_aliasing():
    x = qd.ndarray(dtype=qd.f32, shape=4)
    y = qd.ndarray(dtype=qd.f32, shape=8)
    x.from_numpy(np.array([1, 2, 3, 4], dtype=np.float32))
    y.from_numpy(np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32))

    @qd.kernel
    def k(x: qd.types.ndarray(dtype=qd.f32, ndim=1), y: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        for i in range(4):
            x[i] = y[i + 1]  # read y[i+1], write x[i]: different args -> safe
            y[i * 2] = x[i]  # write y[i*2]: y both read and written, indices are distinct Stmts -> POTENTIAL

    with capture_stderr_fd() as tmp:
        _launch_validated(k, x, y)
        qd.sync()
        out = _read(tmp)
    assert POTENTIAL_MARK in out
    assert DEFINITE_MARK not in out


@test_utils.test(debug=True)
def test_scan_mixed_field_and_ndarray_no_crash():
    """A63 repro: pre-fix, the check-bit checker QD_ASSERT-crashed on ndarray
    kernels; with the graceful skip, field check-bit path and ndarray scan coexist."""
    f = qd.field(dtype=qd.f32, shape=4, needs_grad=True)
    x = qd.ndarray(dtype=qd.f32, shape=4, needs_grad=True)
    x.from_numpy(np.array([1, 2, 3, 4], dtype=np.float32))
    loss = qd.field(dtype=qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def k(f: qd.types.template(), x: qd.types.ndarray(dtype=qd.f32, ndim=1),
          loss: qd.types.template()):
        for i in range(4):
            loss[None] += f[i] * x[i]

    f.from_numpy(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    with qd.ad.Tape(loss=loss, validation=True):
        k(f, x, loss)  # must not raise (pre-fix: validation.cpp is<MatrixPtrStmt> abort)
    assert abs(loss.to_numpy() - 10.0) < 1e-5
