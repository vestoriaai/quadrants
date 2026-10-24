import sys

import quadrants as qd

qd.init(arch=qd.cpu, debug=True)
a = qd.field(qd.f32, 4)


@qd.kernel
def k():
    a[0] = 1.0
    print("MARKER_DIRECT")


k()
qd.sync()
print("DONE", file=sys.stderr)
