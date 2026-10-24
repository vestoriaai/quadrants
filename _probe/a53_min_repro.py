import io
import sys

# simulate pytest capture: wrap sys.stdout BEFORE importing quadrants, exactly
# like pytest's sys-level capture does before collecting test modules
fake = io.StringIO()
_real = sys.stdout
sys.stdout = fake

import quadrants as qd  # noqa: E402
from quadrants._lib import core as _qd_core  # noqa: E402
import quadrants.lang.shell as shell_mod  # noqa: E402

print("pybuf_enabled:", shell_mod.pybuf_enabled, file=_real)
sys.stdout = _real

qd.init(arch=qd.cpu, debug=True)
a = qd.field(qd.f32, 4)


@qd.kernel
def k():
    a[0] = 1.0
    print("MARKER_XYZ")


k()
buf = _qd_core.pop_python_print_buffer()
sys.__stdout__.write(f"after-call pop len: {len(buf)} content={buf!r}\n")
sys.__stdout__.flush()
qd.sync()
sys.__stdout__.write(f"fake.stdout after sync: {fake.getvalue()!r}\n")
sys.__stdout__.flush()
