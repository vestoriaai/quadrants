import quadrants as qd

qd.init(arch=qd.cpu)

fld = qd.field(dtype=qd.i32, shape=())
fld1 = qd.field(dtype=qd.i32, shape=(4,))


@qd.kernel
def to_field():
    n = 0
    for i in range(10):
        n += 1
    fld[None] = n


@qd.kernel
def to_field1d():
    n = 0
    for i in range(10):
        n += 1
    fld1[0] = n


@qd.kernel
def to_ndarray(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
    n = 0
    for i in range(10):
        n += 1
    out[0] = n


@qd.kernel
def ret_only() -> qd.i32:
    n = 0
    for i in range(10):
        n += 1
    return n


to_field(); print(f"0-D field store after loop   -> {fld[None]} (want 10)")
to_field1d(); print(f"1-D field store after loop   -> {fld1[0]} (want 10)")
arr = qd.ndarray(dtype=qd.i32, shape=[4]); to_ndarray(arr); print(f"ndarray store after loop     -> {arr.to_numpy()[0]} (want 10)")
print(f"kernel return after loop     -> {ret_only()} (want 10)")
