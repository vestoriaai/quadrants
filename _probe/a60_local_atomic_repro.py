"""Minimal local-atomic race repro: kernel-local scalar atomic_add.

qd.atomic_add(ck, step) where ck is a kernel-local scalar must behave as a
serialized read-modify-write across threads (each of the 128 unrolled
iterations must observe a distinct old value). Observed on cpu+metal (dev152):
two iterations read the same old value -> one returned value is duplicated.

Usage: python a60_local_atomic_repro.py [arch]
"""

import sys

import numpy as np

import quadrants as qd

arch = getattr(qd, sys.argv[1] if len(sys.argv) > 1 else "cpu")
qd.init(arch=arch, debug=True)

n = 128
step = 42
x = qd.field(qd.i32)
y = qd.field(qd.i32)
c = qd.field(qd.i32)
qd.root.dense(qd.i, n).place(x, y)
qd.root.place(c)


@qd.kernel
def func():
    ck = 0
    for i in range(n):
        x[i] = qd.atomic_add(c[None], step)
        y[i] = qd.atomic_add(ck, step)


func()

cval = c[None]
xs = sorted(x.to_numpy())
ys = sorted(y.to_numpy())
print(f"c={cval} (expect {n * step})")
print("x ok:", xs == [i * step for i in range(n)])
print("y ok:", ys == [i * step for i in range(n)])
if ys != [i * step for i in range(n)]:
    bad = [(i, v) for i, v in enumerate(ys) if v != i * step]
    print("y bad (idx, value):", bad[:8])
    sys.exit(1)
print("LOCAL_ATOMIC_OK")
