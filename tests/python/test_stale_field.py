import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@qd.data_oriented
class _Fam:
    def __init__(self) -> None:
        self.s_rad = qd.field(dtype=qd.f32, shape=(4,))
        self.s_rad.from_numpy(np.ones(4, np.float32) * 0.5)
        self.n = 4

    @qd.kernel
    def scan(self, fr: qd.types.ndarray(ndim=1, dtype=qd.f32),
             out: qd.types.ndarray(ndim=1, dtype=qd.f32), n: int) -> None:
        for i in range(n):
            acc = 0.0
            for s in qd.static(range(4)):
                acc += self.s_rad[s] * fr[0]
            out[i] = acc

    @qd.kernel
    def scan_nd(self, fr: qd.types.ndarray(ndim=1, dtype=qd.f32),
                out: qd.types.ndarray(ndim=1, dtype=qd.f32), n: int) -> None:
        for i in range(n):
            acc = 0.0
            for s in qd.static(range(4)):
                acc += fr[0] * (s + 1)
            out[i] = acc * self.n


@test_utils.test(arch=[qd.cpu])
def test_stale_field_host_ops_raise() -> None:
    """A field created before a re-init must refuse host access instead of UAF (``Field._check_not_stale``)."""
    qd.init(arch=qd.cpu)
    f = qd.field(dtype=qd.f32, shape=(3,))
    f.from_numpy(np.ones(3, np.float32))
    qd.init(arch=qd.cpu)  # re-init finalizes the previous program
    with pytest.raises(qd.lang.exception.QuadrantsRuntimeError, match="finalized"):
        _ = f[0]
    with pytest.raises(qd.lang.exception.QuadrantsRuntimeError, match="finalized"):
        f[0] = 1.0
    with pytest.raises(qd.lang.exception.QuadrantsRuntimeError, match="finalized"):
        f.from_numpy(np.zeros(3, np.float32))
    with pytest.raises(qd.lang.exception.QuadrantsRuntimeError, match="finalized"):
        _ = f.to_numpy()


@test_utils.test(arch=[qd.cpu])
def test_stale_field_member_kernel_launch_raises() -> None:
    """Re-launching a class kernel whose member fields predate the re-init must raise loudly.

    Before the guard this was use-after-free: the recompile resolved ``self.s_rad`` into the
    finalized program's freed memory (garbage shapes / SIGSEGV depending on allocator state).
    """
    qd.init(arch=qd.cpu)
    a = _Fam()
    fr = np.array([2.0], np.float32)
    out = np.zeros(4, np.float32)
    a.scan(fr, out, 4)
    qd.sync()
    qd.init(arch=qd.cpu)  # re-init; object ``a`` (and its fields) survive in Python
    # the launcher wraps compile-path errors in QuadrantsCompilationError; the message
    # names the finalized program either way
    with pytest.raises(
        (qd.lang.exception.QuadrantsCompilationError, qd.lang.exception.QuadrantsRuntimeError),
        match="finalized",
    ):
        a.scan(fr, out, 4)


@test_utils.test(arch=[qd.cpu])
def test_fresh_objects_after_reinit_untouched() -> None:
    """Objects created under the NEW program keep working (the abandon pattern stays legal)."""
    qd.init(arch=qd.cpu)
    _old = _Fam()
    qd.init(arch=qd.cpu)
    b = _Fam()
    out = np.zeros(4, np.float32)
    b.scan(np.array([2.0], np.float32), out, 4)
    qd.sync()
    assert out[0] == pytest.approx(4.0)


@test_utils.test(arch=[qd.cpu])
def test_kernel_not_reading_stale_members_still_compiles() -> None:
    """A class kernel that reads no fields is unaffected even when its owner holds stale fields."""
    qd.init(arch=qd.cpu)
    a = _Fam()
    qd.init(arch=qd.cpu)
    out = np.zeros(4, np.float32)
    a.scan_nd(np.array([1.0], np.float32), out, 4)
    qd.sync()
    assert out[0] == pytest.approx(40.0)
